"""토스 공개 지식 수집 — 음성 상담 에이전트 지식베이스용.

- support.toss.im FAQ (계열사 워크스페이스별, 조회수 포함)
- shopping-docs.toss.im 판매자·광고주 문서 (GitBook, 페이지별 Markdown)

원문은 data/raw/ 에만 저장하고 공개 레포에 올리지 않는다.
사용: python scripts/collect_toss.py
"""
import html
import json
import re
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://support.toss.im/"}

FAQ_API = "https://faq-editor-api.toss.im/api/v1/workspaces/{ws}/faq/?page=1&page_size=300"
FAQ_WORKSPACES = {1: "토스", 5: "토스뱅크", 29: "토스플레이스", 17: "토스모바일", 21: "토스페이"}
SHOPPING_SITEMAP = "https://shopping-docs.toss.im/sitemap.xml"


def fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def html_to_text(s: str) -> str:
    s = re.sub(r"<br\s*/?>|</p>|</li>", "\n", s)
    s = re.sub(r"<li[^>]*>", "- ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s).replace("\xa0", " ")
    return re.sub(r"\n\s*\n+", "\n", s).strip()


def collect_faq() -> int:
    rows = []
    for ws, company in FAQ_WORKSPACES.items():
        url = FAQ_API.format(ws=ws)
        while url:
            page = json.loads(fetch(url))["success"]
            for r in page["results"]:
                rows.append({
                    "id": r["id"],
                    "company": company,
                    "title": r["title"],
                    "answer": html_to_text(r["description"]),
                    "answer_html": r["description"],
                    "category_ids": r["category_set"],
                    "tag_ids": r["tag_set"],
                    "view_count": r["count"],
                    "created": r["created_time"],
                    "updated": r["updated_time"],
                })
            url = page["next"]
            time.sleep(0.5)
    out = RAW / "toss_faq.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def collect_shopping_docs() -> int:
    subs = re.findall(r"<loc>(.*?)</loc>", fetch(SHOPPING_SITEMAP).decode())
    urls = []
    for s in subs:
        urls += re.findall(r"<loc>(.*?)</loc>", fetch(s).decode())
    # 루트(랜딩)는 .md 버전이 없고, undefined/.pfx는 sitemap의 깨진 항목
    urls = sorted({u for u in urls if "undefined" not in u and not u.endswith(".pfx")
                   and u.rstrip("/") != "https://shopping-docs.toss.im"})

    out_dir = RAW / "shopping_docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for u in urls:
        path = u.replace("https://shopping-docs.toss.im", "").strip("/") or "index"
        try:
            md = fetch(u.rstrip("/") + ".md").decode("utf-8")
        except Exception as e:
            index.append({"url": u, "error": str(e)})
            continue
        fname = path.replace("/", "__") + ".md"
        (out_dir / fname).write_text(md, encoding="utf-8")
        title = next((l.lstrip("# ").strip() for l in md.splitlines() if l.startswith("# ")), path)
        index.append({"url": u, "file": fname, "title": title, "section": path.split("/")[0], "chars": len(md)})
        time.sleep(0.3)
    (out_dir / "_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return sum(1 for i in index if "file" in i)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    n_faq = collect_faq()
    n_docs = collect_shopping_docs()
    (RAW / "_collected.json").write_text(
        json.dumps({"date": date.today().isoformat(), "faq": n_faq, "shopping_docs": n_docs}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"FAQ {n_faq}건, 쇼핑 문서 {n_docs}페이지 → {RAW}")
