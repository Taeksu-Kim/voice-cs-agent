"""카탈로그 — 서비스별 YAML(catalog/<서비스>.yaml)을 읽어 검사하고 MCP 도구 정의로 바꾼다 (docs/08 4절).

읽을 때 하는 1차 검사: 필수 필드, 허용된 risk·source·effect 값, 이름 중복, target_keys·grounded_by가 가리키는 대상.
하나라도 틀리면 CatalogError — 서버는 이전 카탈로그를 유지한다 (일부만 반영하지 않는다).
에이전트는 tools/list를 받을 때 2차 검사를 한다 (서버를 믿지 않는 마지막 방어선).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from mcp import types

RISKS = ("read", "notify", "execute", "handoff")
SOURCES = ("model", "context")
EFFECTS = ("authenticate",)
STATUSES = ("active", "deprecated")
META = "csagent/"   # _meta 키 접두어


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    source: str
    required: bool = False
    description: str = ""
    grounded_by: str = ""     # "<도구>.<결과 경로>" — 이번 통화에서 그 도구 결과에 나온 값만 허용 (docs/08 7-2)
    enum: tuple[str, ...] = ()   # 허용값 — inputSchema enum으로 나가 에이전트 형식 검증이 막는다


@dataclass(frozen=True)
class Entry:
    name: str
    service: str              # 파일 이름 = 서비스 (소유 단위)
    owner: str
    version: int
    status: str
    description: str
    endpoint: str
    params: tuple[Param, ...]
    risk: str
    requires_auth: bool = False
    requires_confirmation: bool = False
    requires_contact: bool = False
    asks_customer: bool = False
    target_keys: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    approved_by: str = ""

    def model_params(self) -> list[Param]:
        return [p for p in self.params if p.source == "model"]

    def context_params(self) -> list[str]:
        return [p.name for p in self.params if p.source == "context"]


@dataclass(frozen=True)
class Catalog:
    entries: tuple[Entry, ...]
    digest: str               # 카탈로그 파일 해시 — 평가 실행 기록에 남긴다 (docs/08 9절)

    def get(self, name: str) -> Entry | None:
        return next((e for e in self.entries if e.name == name), None)


def _bool(raw: dict, key: str, where: str) -> bool:
    v = raw.get(key, False)
    if not isinstance(v, bool):
        raise CatalogError(f"{where}: {key}는 true/false여야 합니다 (받은 값: {v!r})")
    return v


def _param(name: str, raw: Any, where: str) -> Param:
    if not isinstance(raw, dict):
        raise CatalogError(f"{where}.{name}: 파라미터 정의는 매핑이어야 합니다")
    source = raw.get("source")
    if source not in SOURCES:
        raise CatalogError(f"{where}.{name}: source는 {'|'.join(SOURCES)} 중 하나 (받은 값: {source!r})")
    if source == "context" and raw.get("grounded_by"):
        raise CatalogError(f"{where}.{name}: context 파라미터에는 grounded_by를 둘 수 없습니다")
    enum = raw.get("enum") or []
    if not isinstance(enum, list) or not all(isinstance(v, str) and v for v in enum) or len(set(enum)) != len(enum):
        raise CatalogError(f"{where}.{name}: enum은 겹치지 않는 문자열 목록이어야 합니다")
    if enum and source == "context":
        raise CatalogError(f"{where}.{name}: context 파라미터에는 enum을 둘 수 없습니다")
    return Param(name, str(raw.get("type", "string")), source, bool(raw.get("required", False)),
                 str(raw.get("description", "")), str(raw.get("grounded_by", "") or ""), tuple(enum))


def _entry(raw: Any, service: str) -> Entry:
    if not isinstance(raw, dict) or not raw.get("name"):
        raise CatalogError(f"{service}: 항목에 name이 없습니다")
    where = f"{service}:{raw['name']}"
    for key in ("description", "endpoint", "risk", "owner"):
        if not raw.get(key):
            raise CatalogError(f"{where}: {key}가 없습니다")
    if raw["risk"] not in RISKS:
        raise CatalogError(f"{where}: risk는 {'|'.join(RISKS)} 중 하나 (받은 값: {raw['risk']!r})")
    status = raw.get("status", "active")
    if status not in STATUSES:
        raise CatalogError(f"{where}: status는 {'|'.join(STATUSES)} 중 하나 (받은 값: {status!r})")
    if not str(raw["endpoint"]).startswith("/"):
        raise CatalogError(f"{where}: endpoint는 /로 시작해야 합니다")
    params = tuple(_param(n, p, where) for n, p in (raw.get("params") or {}).items())
    model = {p.name for p in params if p.source == "model"}
    targets = tuple(raw.get("target_keys") or ())
    if not set(targets) <= model:
        raise CatalogError(f"{where}: target_keys {sorted(set(targets) - model)}는 model 파라미터가 아닙니다")
    effects = tuple(raw.get("effects") or ())
    if not set(effects) <= set(EFFECTS):
        raise CatalogError(f"{where}: 모르는 effects {sorted(set(effects) - set(EFFECTS))}")
    return Entry(
        name=raw["name"], service=service, owner=raw["owner"], version=int(raw.get("version", 1)), status=status,
        description=str(raw["description"]), endpoint=raw["endpoint"], params=params, risk=raw["risk"],
        requires_auth=_bool(raw, "requires_auth", where), requires_confirmation=_bool(raw, "requires_confirmation", where),
        requires_contact=_bool(raw, "requires_contact", where), asks_customer=_bool(raw, "asks_customer", where),
        target_keys=targets, effects=effects, approved_by=str(raw.get("approved_by", "")))


def load_catalog(directory: str | Path) -> Catalog:
    directory = Path(directory)
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise CatalogError(f"카탈로그 파일이 없습니다: {directory}")
    digest = hashlib.sha256()
    entries: list[Entry] = []
    for f in files:
        data = f.read_bytes()
        digest.update(f.name.encode() + b"\0" + data)
        raw = yaml.safe_load(data) or []
        if not isinstance(raw, list):
            raise CatalogError(f"{f.name}: 최상위는 항목 목록이어야 합니다")
        entries.extend(_entry(r, f.stem) for r in raw)
    names = [e.name for e in entries]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise CatalogError(f"도구 이름 중복: {dup}")
    for e in entries:
        for p in e.params:
            if p.grounded_by and p.grounded_by.split(".", 1)[0] not in names:
                raise CatalogError(f"{e.service}:{e.name}.{p.name}: grounded_by가 가리키는 도구가 없습니다 ({p.grounded_by})")
    return Catalog(tuple(entries), digest.hexdigest()[:12])


def to_tool(entry: Entry) -> types.Tool:
    """카탈로그 항목 → MCP 도구 정의. context 파라미터는 inputSchema에서 빠진다 (모델에 안 보임)."""
    props = {p.name: {"type": p.type, **({"description": p.description} if p.description else {}),
                      **({"enum": list(p.enum)} if p.enum else {})}
             for p in entry.model_params()}
    required = [p.name for p in entry.model_params() if p.required]
    schema: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        schema["required"] = required
    grounded = {p.name: p.grounded_by for p in entry.model_params() if p.grounded_by}
    return types.Tool(
        name=entry.name, description=entry.description, input_schema=schema,
        # 표준 힌트 — 다른 MCP 클라이언트와 의미를 맞추기 위한 것. 게이트 판단에는 쓰지 않는다
        annotations=types.ToolAnnotations(read_only_hint=entry.risk == "read",
                                          destructive_hint=entry.risk == "execute"),
        meta={
            f"{META}risk": entry.risk,
            f"{META}requires_auth": entry.requires_auth,
            f"{META}requires_confirmation": entry.requires_confirmation,
            f"{META}requires_contact": entry.requires_contact,
            f"{META}asks_customer": entry.asks_customer,
            f"{META}target_keys": list(entry.target_keys),
            f"{META}context_params": entry.context_params(),
            f"{META}grounded_by": grounded,
            f"{META}effects": list(entry.effects),
            f"{META}param_order": [p.name for p in entry.model_params()],
            f"{META}version": entry.version,
            f"{META}owner": entry.owner,
        })
