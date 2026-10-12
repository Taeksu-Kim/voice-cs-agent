import pytest

from csagent.knowledge.models import Chunk
from csagent.retrieval.embedders import FakeDenseEmbedder, KiwiBM25
from csagent.retrieval.filters import SearchFilter
from csagent.retrieval.fusion import rrf
from csagent.retrieval.models import Hit
from csagent.retrieval.reranker import FakeReranker, format_pairs
from csagent.retrieval.retriever import HybridRetriever
from csagent.retrieval.store import MilvusStore


def _ws_tokenizer(text: str) -> list[str]:
    return text.split()


def test_rrf_merges_and_orders():
    a = [Hit("x", 0.9), Hit("y", 0.8)]
    b = [Hit("y", 5.0), Hit("z", 4.0)]
    fused = rrf([a, b], k=60)
    assert [h.chunk_id for h in fused][0] == "y"  # 양쪽에 다 나온 것이 1위
    assert {h.chunk_id for h in fused} == {"x", "y", "z"}


def test_bm25_query_ignores_unknown_tokens():
    bm = KiwiBM25(tokenizer=_ws_tokenizer).fit(["배송 기간 안내", "송금 한도 안내"])
    [q] = bm.encode_queries(["배송 우주선"])
    assert len(q) == 1  # 「우주선」은 코퍼스에 없어 제외


def test_reranker_template_has_markers():
    q, d = format_pairs("배송?", ["문서"])
    assert "<Instruct>:" in q and "<Query>: 배송?" in q and d[0].startswith("<Document>: 문서")


def _chunk(cid, source, title, text, audience, domains, discontinued=False):
    return Chunk(cid, cid.split("#")[0], source, title, "", text,
                 meta={"audience": audience, "companies": ["토스"], "domains": domains,
                       "discontinued": discontinued})


@pytest.fixture
def retriever(tmp_path):
    chunks = [
        _chunk("faq:1#0", "faq", "쇼핑 반품", "반품 은 앱 에서 접수", "consumer", ["쇼핑"]),
        _chunk("faq:2#0", "faq", "송금 한도", "송금 한도 는 1일 5천만 원", "consumer", ["송금·계좌"]),
        _chunk("faq:3#0", "faq", "카드 반품 환불", "카드 결제 반품 환불", "consumer", ["카드"]),
        _chunk("faq:4#0", "faq", "종료 상품", "반품 종료 상품", "consumer", ["쇼핑"], discontinued=True),
        _chunk("shop:a#0", "shopping_doc", "반품 관리", "반품 요청 은 셀러 가 처리", "seller", ["주문·배송·CS"]),
    ]
    texts = [c.embedding_text() for c in chunks]
    dense, bm = FakeDenseEmbedder(dim=16), KiwiBM25(tokenizer=_ws_tokenizer).fit(texts)
    store = MilvusStore(str(tmp_path / "t.db"), "kb", dim=16)
    store.recreate()
    store.insert(chunks, dense.encode_docs(texts), bm.encode_docs(texts))
    return HybridRetriever(store, dense, bm, FakeReranker(), prefetch=5)


def test_store_roundtrip(retriever):
    assert retriever.store.count() == 5


def test_sparse_search_finds_keyword(retriever):
    hits = retriever.search("송금 한도", k=1, mode="sparse", rerank=False)
    assert hits[0].chunk_id == "faq:2#0" and hits[0].title == "송금 한도"


def test_filter_expr():
    expr = SearchFilter(audience="consumer", companies=("토스뱅크",)).to_expr()
    assert 'audience == "consumer"' in expr and 'array_contains_any(companies, ["토스뱅크"])' in expr
    assert "discontinued == false" in expr


def test_audience_filter_excludes_seller_docs_and_discontinued(retriever):
    hits = retriever.search("반품", k=5, flt=SearchFilter(audience="consumer"))
    ids = {h.chunk_id for h in hits}
    assert "shop:a#0" not in ids and "faq:4#0" not in ids
    assert {"faq:1#0", "faq:3#0"} <= ids
