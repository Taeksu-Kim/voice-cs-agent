"""MCP 도구 레지스트리 — 도구 서버의 tools/list로 도구 목록을 받고 tools/call로 실행한다 (docs/08).

에이전트 쪽이 맡는 것
- 2차 검사 (4-3): _meta의 csagent/* 값이 없거나 틀린 도구는 노출하지 않는다 (fail closed) — warnings에 남김
- 통화별 스냅샷 (4-1): 통화의 첫 조회 때 그 시점 목록을 state에 고정. reload는 새 통화부터 반영
- 컨텍스트 주입 (4-2): source=context 파라미터(customer_id)는 모델 인자가 아니라 통화 컨텍스트에서 채운다
- 효과 반영 (4-2): effects=authenticate 도구의 결과로 인증 상태를 바꾼다 (도구 이름은 모른다)
- 근거 검증 (7-2): grounded_by — 이번 통화에서 그 도구 결과에 나온 값만 허용 (precheck 자리, 확인 질문 전에 돈다)
- 실패 결과 형태 유지 (8절): 시간 초과는 예외로 올려 registry.run이 기존과 같은 「TimeoutError: …」 결과로 바꾼다
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Callable, Optional

from mcp import Client, types

from csagent.agent.state import CallState
from csagent.agent.tools.builtin import builtin_specs
from csagent.agent.tools.registry import ToolRegistry, ToolSpec

META = "csagent/"
RISKS = ("read", "notify", "execute", "handoff")
EFFECTS = ("authenticate",)
BOOL_KEYS = ("requires_auth", "requires_confirmation", "requires_contact", "asks_customer")


class ToolServerUnavailable(RuntimeError):
    pass


class SyncMcpClient:
    """동기 에이전트 루프에서 MCP 클라이언트를 쓰기 위한 다리.
    백그라운드 스레드의 이벤트 루프에서 세션 하나를 유지하고, 호출은 그 루프로 넘긴다 (여러 세션 스레드에서 동시 호출 가능)."""

    def __init__(self, target: Any, timeout_s: float = 3.0, connect_timeout_s: float = 10.0) -> None:
        self.timeout_s = timeout_s
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop: Optional[asyncio.Event] = None
        self._client: Optional[Client] = None
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._main(target), self._loop)
        if not self._ready.wait(connect_timeout_s) or self._error is not None:
            self.close()
            raise ToolServerUnavailable(f"도구 서버 연결 실패: {target!r} ({self._error!r})")

    async def _main(self, target: Any) -> None:
        self._stop = asyncio.Event()
        try:
            async with Client(target, read_timeout_seconds=self.timeout_s + 1) as client:
                self._client = client
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:   # 연결 실패 — 생성자가 기다리다 알아챈다
            self._error = e
            self._ready.set()

    def _run(self, coro: Any, timeout_s: float) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout_s)

    def list_tools(self) -> types.ListToolsResult:
        return self._run(self._client.list_tools(cache_mode="bypass"), self.timeout_s + 2)

    def call_tool(self, name: str, args: dict[str, Any], meta: dict[str, Any]) -> types.CallToolResult:
        # 도구 서버가 도메인 API 시간 초과를 먼저 돌려주게, 에이전트 쪽 대기는 조금 길게
        return self._run(self._client.call_tool(name, args, meta=meta), self.timeout_s + 2)

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._loop.call_soon_threadsafe(self._loop.stop)


def _seen_values(state: CallState, tool: str, path: str) -> list[Any]:
    """이번 통화 trace에서 tool의 성공 결과를 path(예: cards.card_id)로 따라가 나온 값들 (순서 유지)."""
    out: list[Any] = []

    def walk(node: Any, keys: list[str]) -> None:
        if isinstance(node, list):
            for x in node:
                walk(x, keys)
        elif not keys:
            if node not in out:
                out.append(node)
        elif isinstance(node, dict) and keys[0] in node:
            walk(node[keys[0]], keys[1:])

    for e in state.trace.of("tool_result"):
        if e.data.get("name") == tool and e.data.get("ok"):
            walk(e.data, path.split("."))
    return out


def grounding_check(grounded: dict[str, str]) -> Callable[[CallState, dict], Optional[str]]:
    def check(state: CallState, args: dict) -> Optional[str]:
        for param, ref in grounded.items():
            tool, path = ref.split(".", 1)
            seen = _seen_values(state, tool, path)
            value = args.get(param)
            if value not in seen:
                avail = ", ".join(map(str, seen)) if seen else f"아직 없음 — 먼저 {tool}로 조회"
                return f"{param}는 {tool} 결과의 {path.rsplit('.', 1)[-1]} 중 하나여야 합니다 (받은 값: {value}, 가능: {avail})"
        return None
    return check


def _apply_effects(state: CallState, effects: tuple[str, ...], data: dict[str, Any]) -> None:
    if "authenticate" in effects:
        if data.get("status") == "success":
            state.authenticated = True
        else:
            state.auth_failures += 1


def _remote_fn(client: SyncMcpClient, name: str, context_params: tuple[str, ...],
               effects: tuple[str, ...]) -> Callable[..., dict]:
    def fn(state: CallState, **args: Any) -> dict:
        context = {p: getattr(state.ctx, p, "") for p in context_params}
        missing = [p for p, v in context.items() if not v]
        if missing:
            raise RuntimeError(f"통화 컨텍스트 없음: {', '.join(missing)} (통화를 콜 인프라로 열었는지 확인)")
        t0 = time.perf_counter()
        try:
            res = client.call_tool(name, args, {f"{META}context": context})
        except TimeoutError:
            state.trace.add(state.turn, "tool_hop", {"name": name, "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                                                     "timeout": True})
            raise TimeoutError(f"{name} 백엔드 응답 없음 (timeout)") from None
        state.trace.add(state.turn, "tool_hop", {"name": name, "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                                                 "domain_ms": (res.meta or {}).get(f"{META}domain_ms")})
        body = res.structured_content
        if body is None and res.content:
            body = json.loads(getattr(res.content[0], "text", "") or "{}")
        body = body or {}
        if body.get("ok"):
            data = dict(body.get("data") or {})
            _apply_effects(state, effects, data)
            return data
        err = body.get("error") or {}
        code, message = err.get("code", "UNKNOWN"), err.get("message", "")
        if code == "BACKEND_TIMEOUT":
            raise TimeoutError(message)          # 기존 장애 주입과 같은 「TimeoutError: …」 결과가 된다
        if code == "BACKEND_UNAVAILABLE":
            raise ConnectionError(message)
        return {"error": message, "code": code}  # 도메인 거부 (domain_rejected) — 구조화 코드를 함께 남긴다
    return fn


def spec_from_tool(tool: types.Tool, client: Optional[SyncMcpClient]) -> ToolSpec | str:
    """MCP 도구 정의 → ToolSpec. 2차 검사에 걸리면 경고 문자열 (그 도구는 노출하지 않는다)."""
    meta = tool.meta or {}
    risk = meta.get(f"{META}risk")
    if risk not in RISKS:
        return f"{tool.name}: _meta {META}risk 없음·잘못됨 ({risk!r}) — 제외"
    flags = {k: meta.get(f"{META}{k}", False) for k in BOOL_KEYS}
    bad = [k for k, v in flags.items() if not isinstance(v, bool)]
    if bad:
        return f"{tool.name}: _meta {bad}는 true/false여야 함 — 제외"
    schema = tool.input_schema or {}
    props: dict[str, Any] = schema.get("properties") or {}
    order = [k for k in (meta.get(f"{META}param_order") or list(props)) if k in props]
    targets = tuple(meta.get(f"{META}target_keys") or ())
    grounded = dict(meta.get(f"{META}grounded_by") or {})
    effects = tuple(meta.get(f"{META}effects") or ())
    context_params = tuple(meta.get(f"{META}context_params") or ())
    if not set(targets) <= set(props) or not set(grounded) <= set(props):
        return f"{tool.name}: target_keys·grounded_by가 inputSchema에 없는 인자를 가리킴 — 제외"
    if not set(effects) <= set(EFFECTS):
        return f"{tool.name}: 모르는 effects {sorted(set(effects) - set(EFFECTS))} — 제외"
    if set(context_params) & set(props):
        return f"{tool.name}: context 파라미터가 inputSchema에 노출됨 — 제외"
    return ToolSpec(
        tool.name, tool.description or "", {k: props[k].get("description", "") for k in order},
        risk=risk, requires_auth=flags["requires_auth"], requires_confirmation=flags["requires_confirmation"],
        requires_contact=flags["requires_contact"], asks_customer=flags["asks_customer"], target_keys=targets,
        choices={k: tuple(p["enum"]) for k, p in props.items() if p.get("enum")},
        required=tuple(schema.get("required") or ()),
        precheck=grounding_check(grounded) if grounded else None,
        fn=_remote_fn(client, tool.name, context_params, effects) if client is not None else None)


class McpToolRegistry(ToolRegistry):
    """도구 서버(MCP) 도구 + 에이전트 내장 도구. 내장 도구와 이름이 겹치는 원격 도구는 받지 않는다."""

    def __init__(self, client: SyncMcpClient, builtins: Optional[list[ToolSpec]] = None) -> None:
        super().__init__()
        self.client = client
        self._builtins = builtins if builtins is not None else builtin_specs()
        self._lock = threading.Lock()
        self.catalog_digest = ""
        self.warnings: list[str] = []
        self.reload()

    def reload(self) -> dict[str, Any]:
        """tools/list를 다시 받는다 — 진행 중인 통화는 자기 스냅샷을 계속 쓴다."""
        res = self.client.list_tools()
        builtin_names = {s.name for s in self._builtins}
        specs: dict[str, ToolSpec] = {}
        warnings: list[str] = []
        for tool in res.tools:
            if tool.name in builtin_names:
                warnings.append(f"{tool.name}: 내장 도구와 이름이 겹침 — 제외")
                continue
            spec = spec_from_tool(tool, self.client)
            if isinstance(spec, str):
                warnings.append(spec)
            else:
                specs[spec.name] = spec
        for s in self._builtins:
            specs[s.name] = s
        with self._lock:
            self._specs = specs
            self.catalog_digest = str((res.meta or {}).get(f"{META}catalog", ""))
            self.warnings = warnings
        return {"catalog": self.catalog_digest, "tools": len(specs), "warnings": warnings}

    def for_call(self, state: CallState) -> ToolRegistry:
        snap = getattr(state, "tool_snapshot", None)
        if snap is None:
            with self._lock:
                snap = ToolRegistry(list(self._specs.values()))
            state.tool_snapshot = snap
        return snap

    def missing_for(self, workers: dict[str, Any]) -> list[str]:
        """워커가 허용했는데 목록에 없는 도구 (리로드 경고용)."""
        names = set(self._specs)
        return sorted({f"{w}:{t}" for w, wk in workers.items() for t in (wk.policy.tools or ()) if t not in names})
