from csagent.knowledge.chunking import Chunker
from csagent.knowledge.cleaning import is_not_found_page, mark_app_links, strip_gitbook_noise
from csagent.knowledge.models import Document


def test_gitbook_trailer_and_tags_removed():
    md = (
        "> For the complete documentation index, see [llms.txt](x).\n\n"
        "# CS 처리하기\n\n{% hint style=\"info\" %}\n본문 \\[결제완료] 상태\n{% endhint %}\n"
        "<figure><img src=\"a.png\"></figure>\n\n---\n\n# Agent Instructions\nGET https://x?ask="
    )
    out = strip_gitbook_noise(md)
    assert "Agent Instructions" not in out and "GET https" not in out
    assert "{%" not in out and "<figure" not in out and "complete documentation" not in out
    assert "[결제완료] 상태" in out


def test_not_found_page():
    assert is_not_found_page("# Page Not Found\n...")
    assert not is_not_found_page("# CS 처리하기\n")


def test_app_link_marked():
    assert mark_app_links("안내\n- 주문 내역 바로가기") == "안내\n[앱 링크: 주문 내역]"


def test_faq_is_single_chunk():
    doc = Document("faq:1", "faq", "배송은 얼마나 걸리나요?", "평균 3-5일이에요.")
    [c] = Chunker().split(doc)
    assert c.chunk_id == "faq:1#0" and c.section == ""
    assert c.embedding_text().startswith("배송은 얼마나 걸리나요?\n")


def test_doc_split_by_headings_keeps_path():
    body = ("# 광고 신청하기\n도입\n\n## 캠페인 만들기\n### 광고 세트\n" + "가" * 300
            + "\n### 예산\n" + "나" * 300)
    chunks = Chunker().split(Document("shop:a", "shopping_doc", "광고 신청하기", body))
    assert [c.section for c in chunks] == ["", "캠페인 만들기 > 광고 세트", "캠페인 만들기 > 예산"]
    assert chunks[0].text == "도입"  # 문서 제목(#)은 본문에서 떼어진다


def test_q_lines_stay_inside_category_section():
    body = "### 오늘 출발\nQ1. 발송 기한은?\n답1\n\nQ2. 페널티는?\n답2"
    [c] = Chunker().split(Document("shop:p", "shopping_doc", "페널티 FAQ", body))
    assert c.section == "오늘 출발" and "Q1. 발송 기한은?" in c.text and "Q2. 페널티는?" in c.text


def test_long_text_split_within_limit():
    para = "\n".join("줄" * 100 for _ in range(30))  # ~3,000자
    parts = Chunker(max_chars=1200).split(Document("faq:2", "faq", "긴 답변", para))
    assert len(parts) >= 3 and all(len(p.text) <= 1200 for p in parts)


def test_long_table_split_repeats_header():
    table = "| 항목 | 점수 |\n| --- | --- |\n" + "\n".join(f"| 위반{i} | {i} |" for i in range(200))
    parts = Chunker(max_chars=500).split(Document("shop:t", "shopping_doc", "페널티", "## 표\n" + table))
    assert len(parts) > 1
    assert all(p.text.startswith("| 항목 | 점수 |\n| --- | --- |") for p in parts)


def test_text_after_table_does_not_get_header():
    body = "## 섹션\n" + "| a | b |\n| --- | --- |\n| 1 | 2 |\n\n" + "본문 " * 400
    parts = Chunker(max_chars=500).split(Document("shop:u", "shopping_doc", "x", body))
    assert not any(p.text.startswith("| a | b |") for p in parts[1:])


def test_html_table_cells_separated():
    out = strip_gitbook_noise("<table><tr><td>이런 상황</td><td>이렇게 동작해요</td></tr></table>")
    assert "이런 상황 | 이렇게 동작해요" in out


def test_links_and_hard_breaks_cleaned():
    out = strip_gitbook_noise("[[4. 광고비 정산 하기](/ads/x)](/y) 참고\\\n다음 줄")
    assert out == "4. 광고비 정산 하기 참고\n다음 줄"


def test_list_inside_table_cell_stays_on_one_row():
    md = "| 항목 | 기준 |\n| - | - |\n| 거부 | <ul><li>A를 거부</li><li>B를 거부</li></ul> |"
    out = strip_gitbook_noise(md)
    assert "| 거부 | A를 거부 / B를 거부 |" in out


def test_length_split_prefers_q_boundary():
    answer = "\n\n".join(["문단 " * 90] * 3)  # 여러 문단짜리 답변 (~810자) — 두 개면 1,200자 초과
    body = f"### 오늘 출발\nQ1. 첫 질문\n{answer}\n\nQ2. 둘째 질문\n{answer}"
    parts = Chunker(max_chars=1200).split(Document("shop:q", "shopping_doc", "FAQ", body))
    assert [p.text.split("\n")[0] for p in parts] == ["Q1. 첫 질문", "Q2. 둘째 질문"]
