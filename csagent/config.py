"""설정은 이 파일 한 곳에서만 읽는다 (.env → 환경변수 → 기본값).

WSL 주의: vLLM을 WSL에서 띄우면 localhost가 안 붙는 경우가 있다 — .env에 WSL IP를 적는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


@dataclass(frozen=True)
class Settings:
    raw_dir: Path
    processed_dir: Path
    milvus_uri: str
    collection: str
    bm25_stats: Path
    embedder_url: str
    embedder_model: str
    embed_dim: int
    reranker_url: str
    reranker_model: str
    llm_url: str
    llm_model: str
    voice_url: str


def load_settings() -> Settings:
    _load_dotenv(ROOT / ".env")
    env = os.environ.get
    data = ROOT / "data"
    return Settings(
        raw_dir=data / "raw",
        processed_dir=data / "processed",
        milvus_uri=env("MILVUS_URI", str(data / "milvus.db")),
        collection=env("COLLECTION", "knowledge"),
        bm25_stats=data / "processed" / "kiwi_bm25_stats.json",
        embedder_url=env("EMBEDDER_URL", "http://localhost:8001/v1"),
        embedder_model=env("EMBEDDER_MODEL", "qwen3-emb"),
        embed_dim=int(env("EMBED_DIM", "1024")),
        # 8002는 Qwen3.8-27B 서버가 쓴다 — 리랭커는 8012
        reranker_url=env("RERANKER_URL", "http://localhost:8012"),
        reranker_model=env("RERANKER_MODEL", "qwen3-reranker"),
        llm_url=env("LLM_URL", "http://localhost:8002/v1"),
        llm_model=env("LLM_MODEL", "qwen3.8-27b-redhat-int4"),
        voice_url=env("VOICE_URL", "http://localhost:8013"),   # scripts/voice_server.py (STT·TTS)
    )
