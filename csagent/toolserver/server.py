"""도구 서버 (MCP) — 카탈로그를 tools/list로 제공하고 tools/call을 도메인 API로 라우팅한다 (docs/08 3·6절).

    POST /mcp       MCP (streamable HTTP, 무상태, JSON 응답)
    POST /reload    카탈로그 다시 읽기 — 실패하면 이전 카탈로그 유지
    GET  /health    카탈로그 해시·도구 수

- 도메인 검증·응답 가공은 하지 않는다. 도메인 API 응답 본문({ok, data|error})을 structuredContent로 그대로 넘긴다
- 컨텍스트 파라미터(customer_id 등)는 요청 _meta의 csagent/context에서 꺼내 도메인 API 요청에 합친다 — 모델 인자로는 받지 않는다
- 시간 초과·연결 실패도 같은 {ok:false, error:{code}} 형식으로 돌려준다 (에이전트가 기존과 같은 실패 결과로 바꾼다, docs/08 8절)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from mcp import types
from mcp.server.lowlevel import Server
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from csagent.toolserver.catalog import META, Catalog, CatalogError, load_catalog, to_tool

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CATALOG = ROOT / "catalog"


def _result(body: dict[str, Any], domain_ms: Optional[float] = None) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(body, ensure_ascii=False))],
        structured_content=body, is_error=not body.get("ok", False),
        meta={f"{META}domain_ms": round(domain_ms, 1)} if domain_ms is not None else None)


def _error(code: str, message: str, domain_ms: Optional[float] = None) -> types.CallToolResult:
    return _result({"ok": False, "error": {"code": code, "message": message}}, domain_ms)


class ToolServer:
    def __init__(self, catalog_dir: str | Path = DEFAULT_CATALOG, domain_url: str = "http://127.0.0.1:8020",
                 timeout_s: float = 3.0, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.catalog_dir = Path(catalog_dir)
        self.domain_url = domain_url.rstrip("/")
        self.timeout_s = timeout_s
        self.transport = transport          # 테스트: httpx.ASGITransport(app=도메인 API)
        self.catalog: Catalog = load_catalog(self.catalog_dir)
        self.server = Server("csagent-tools", version="1",
                             on_list_tools=self._list_tools, on_call_tool=self._call_tool)

    def reload(self) -> Catalog:
        """다시 읽기 — 검사에 실패하면 CatalogError, 이전 카탈로그는 그대로 둔다."""
        self.catalog = load_catalog(self.catalog_dir)
        return self.catalog

    async def _list_tools(self, ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[to_tool(e) for e in self.catalog.entries if e.status == "active"],
            meta={f"{META}catalog": self.catalog.digest})

    async def _call_tool(self, ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        entry = self.catalog.get(params.name)
        if entry is None or entry.status != "active":
            return _error("UNKNOWN_TOOL", f"unknown_tool:{params.name}")
        meta = dict(params.meta or {})
        context = meta.get(f"{META}context") or {}
        missing = [k for k in entry.context_params() if not context.get(k)]
        if missing:
            return _error("CONTEXT_MISSING", f"컨텍스트 파라미터 없음: {', '.join(missing)}")
        args = dict(params.arguments or {})
        hidden = [k for k in args if k in entry.context_params()]
        if hidden:   # 모델이 컨텍스트 파라미터를 직접 넣으면 받지 않는다 (다른 고객 지정 차단)
            return _error("INVALID_ARGS", f"모델이 채울 수 없는 인자: {', '.join(hidden)}")
        body = {**args, **{k: context[k] for k in entry.context_params()}}
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(base_url=self.domain_url, timeout=self.timeout_s,
                                         transport=self.transport) as http:
                r = await http.post(entry.endpoint, json=body)
            ms = (time.perf_counter() - t0) * 1000
            data = r.json()
        except httpx.TimeoutException:
            return _error("BACKEND_TIMEOUT", f"{entry.name} 백엔드 응답 없음 (timeout)", (time.perf_counter() - t0) * 1000)
        except (httpx.HTTPError, ValueError) as e:
            return _error("BACKEND_UNAVAILABLE", f"{entry.name} 백엔드 연결 실패 ({type(e).__name__})",
                          (time.perf_counter() - t0) * 1000)
        if not isinstance(data, dict) or "ok" not in data:
            return _error("BACKEND_BAD_RESPONSE", f"{entry.name} 응답 형식 오류 (HTTP {r.status_code})", ms)
        return _result(data, ms)

    def http_app(self) -> Starlette:
        async def reload(_: Request) -> JSONResponse:
            try:
                cat = self.reload()
            except CatalogError as e:
                return JSONResponse({"ok": False, "error": {"code": "CATALOG_INVALID", "message": str(e)},
                                     "catalog": self.catalog.digest}, status_code=400)
            return JSONResponse({"ok": True, "catalog": cat.digest, "tools": len(cat.entries)})

        async def health(_: Request) -> JSONResponse:
            return JSONResponse({"ok": True, "catalog": self.catalog.digest, "tools": len(self.catalog.entries),
                                 "domain_url": self.domain_url})

        return self.server.streamable_http_app(
            stateless_http=True, json_response=True,
            custom_starlette_routes=[Route("/reload", reload, methods=["POST"]),
                                     Route("/health", health, methods=["GET"])])


def create_default_app() -> Starlette:
    """uvicorn factory — 환경변수 CSAGENT_CATALOG_DIR · DOMAIN_URL · TOOL_TIMEOUT_S."""
    return ToolServer(os.environ.get("CSAGENT_CATALOG_DIR", str(DEFAULT_CATALOG)),
                      os.environ.get("DOMAIN_URL", "http://127.0.0.1:8020"),
                      float(os.environ.get("TOOL_TIMEOUT_S", "3"))).http_app()
