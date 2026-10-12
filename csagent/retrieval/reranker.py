"""2단계 리랭커 (질의-문서 쌍 채점).

⚠ Qwen3-Reranker는 공식 채팅 템플릿이 필수 — raw 텍스트를 넣으면 순위가 무의미해진다
(agentic_rag 실측). format_pairs가 그 템플릿을 만든다.
"""

from __future__ import annotations

from typing import Protocol, Sequence

_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = (
    "Given a customer's question to a customer service, "
    "judge whether the help-center passage answers it"
)


def format_pairs(query: str, docs: Sequence[str],
                 instruction: str = DEFAULT_INSTRUCTION) -> tuple[str, list[str]]:
    return (f"{_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n",
            [f"<Document>: {d}{_SUFFIX}" for d in docs])


class Reranker(Protocol):
    def rerank(self, query: str, docs: Sequence[str]) -> list[float]: ...


class HttpReranker:
    """vLLM /score 엔드포인트 (Qwen3-Reranker-0.6B, sequence-classification 변환 서빙)."""

    def __init__(self, base_url: str, model: str, *, instruction: str = DEFAULT_INSTRUCTION,
                 doc_chars: int = 1000, query_chars: int = 600, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.instruction = instruction
        # 쌍 길이가 서빙 한도를 넘으면 요청이 무한 대기한다(실측) → 클라이언트에서 자른다
        self.doc_chars, self.query_chars = doc_chars, query_chars
        self.timeout = timeout

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        import requests  # lazy

        if not docs:
            return []
        q, d = format_pairs(query[:self.query_chars], [x[:self.doc_chars] for x in docs],
                            self.instruction)
        r = requests.post(f"{self.base_url}/score",
                          json={"model": self.model, "text_1": q, "text_2": d},
                          timeout=self.timeout)
        r.raise_for_status()
        return [x["score"] for x in sorted(r.json()["data"], key=lambda x: x["index"])]


class FakeReranker:
    """질의 단어가 문서에 들어 있는 비율 (테스트 전용)."""

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        toks = query.split()
        return [sum(t in d for t in toks) / max(len(toks), 1) for d in docs]
