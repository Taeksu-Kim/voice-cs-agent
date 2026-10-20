"""도구 스택 조립 — 에이전트가 쓸 도구 레지스트리와 콜 인프라(픽스처)를 한 쌍으로 만든다 (docs/08).

- remote_stack: 도구 서버(MCP)·도메인 API가 따로 떠 있을 때 (scripts/run_tools.py) — 평가·웹 서버
- local_stack:  세 층을 한 프로세스에 (MCP는 메모리 연결, 도메인 API는 ASGI 직접 호출) — 테스트.
                경계·검사·실패 형태는 remote와 같다. 목업 로직은 도메인 API 한 곳에만 있다
- offline_registry: 실행 없이 카탈로그 메타데이터만 (저장된 통화 다시 채점 — scripts/rejudge.py)
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from csagent.agent.contracts import CallContext
from csagent.agent.state import CallState
from csagent.agent.tools.builtin import builtin_specs
from csagent.agent.tools.registry import ToolRegistry
from csagent.agent.tools.remote import McpToolRegistry, SyncMcpClient, ToolServerUnavailable, spec_from_tool
from csagent.callinfra import Fixtures, HttpFixtures, StoreFixtures, close_call, open_call
from csagent.config import ROOT, Settings, load_settings


@dataclass
class ToolStack:
    tools: McpToolRegistry
    fixtures: Fixtures

    def open(self, ctx: CallContext) -> CallState:
        return open_call(ctx, self.fixtures)

    def close(self, state: CallState) -> None:
        close_call(state, self.fixtures)

    def versions(self) -> dict[str, Any]:
        """평가 실행 기록용 (docs/08 9절): 카탈로그 해시 + 코드 커밋 (도구 서버·도메인 API가 같은 저장소)."""
        return {"catalog": self.tools.catalog_digest, **git_version(), "tool_warnings": list(self.tools.warnings)}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout


def git_version() -> dict[str, Any]:
    """커밋 + 커밋 안 된 변경. dirty면 diff_sha(변경 내용 해시)를 붙인다 — 같은 커밋 위의 다른 수정을 구분한다."""
    try:
        head = _git("rev-parse", "--short", "HEAD").strip()
        diff = _git("diff", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return {"commit": "", "dirty": None}
    out: dict[str, Any] = {"commit": head, "dirty": bool(diff)}
    if diff:
        out["diff_sha"] = hashlib.sha1(diff.encode()).hexdigest()[:12]
    return out


def model_versions(s: Optional[Settings] = None) -> dict[str, Any]:
    """설정의 모델 이름 + LLM 서버가 실제로 서빙 중인 모델 id (설정과 서빙이 어긋난 걸 잡는다)."""
    s = s or load_settings()
    out: dict[str, Any] = {"llm": s.llm_model, "embedder": s.embedder_model, "reranker": s.reranker_model}
    try:
        out["llm_served"] = [m["id"] for m in httpx.get(f"{s.llm_url}/models", timeout=3).json()["data"]]
    except (httpx.HTTPError, KeyError, ValueError):
        out["llm_served"] = None
    return out


def record_versions(stack: "ToolStack", out_dir: Path) -> dict[str, Any]:
    """실행 하나의 버전 기록 (평가 run_meta·서버 server_meta). dirty면 변경 내용을 code_<diff_sha>.diff로 남긴다
    — 커밋만으로는 그때 코드·지시문을 되살릴 수 없다."""
    meta = {**stack.versions(), "models": model_versions()}
    if meta.get("diff_sha"):
        path = out_dir / f"code_{meta['diff_sha']}.diff"
        if not path.exists():
            path.write_text(_git("diff", "HEAD"), encoding="utf-8")
    return meta


def remote_stack(s: Optional[Settings] = None) -> ToolStack:
    """기동 헬스체크 (docs/08 8절): 도구 서버 tools/list, 도메인 API /health — 하나라도 안 되면 ToolServerUnavailable."""
    s = s or load_settings()
    fixtures = HttpFixtures(s.domain_url)
    try:
        if not fixtures.health().get("fixtures"):
            raise ToolServerUnavailable(f"도메인 API가 픽스처 경로 없이 떠 있음: {s.domain_url} (평가·개발 모드로 띄울 것)")
    except httpx.HTTPError as e:
        raise ToolServerUnavailable(f"도메인 API 연결 실패: {s.domain_url} ({type(e).__name__})") from e
    tools = McpToolRegistry(SyncMcpClient(s.tool_server_url, timeout_s=s.tool_timeout_s))
    return ToolStack(tools, fixtures)


def local_stack(catalog_dir: Optional[str | Path] = None) -> ToolStack:
    from csagent.domain.app import create_app
    from csagent.domain.store import FixtureStore
    from csagent.toolserver.server import DEFAULT_CATALOG, ToolServer

    store = FixtureStore()
    server = ToolServer(catalog_dir or DEFAULT_CATALOG, "http://domain",
                        transport=httpx.ASGITransport(app=create_app(store)))
    return ToolStack(McpToolRegistry(SyncMcpClient(server.server)), StoreFixtures(store))


def offline_registry(catalog_dir: Optional[str | Path] = None) -> ToolRegistry:
    """실행 함수 없이 메타데이터만 — 채점(절차 감사)이 위험도·실행 조건만 보면 될 때."""
    from csagent.toolserver.catalog import load_catalog, to_tool
    from csagent.toolserver.server import DEFAULT_CATALOG

    specs = [spec_from_tool(to_tool(e), None) for e in load_catalog(catalog_dir or DEFAULT_CATALOG).entries]
    return ToolRegistry([s for s in specs if not isinstance(s, str)] + builtin_specs())
