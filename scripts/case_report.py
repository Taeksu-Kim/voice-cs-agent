"""케이스 리포트 — 탐지 → triage → APO(후보·테스트·최종 제안)를 한 장으로 묶는다. 새로 판단하지 않고 산출물만 모은다.

    python scripts/case_report.py runs/cases/<시각>/<케이스>

결과: <케이스 폴더>/report.md. 사람이 반영을 결정할 때, 반영한 뒤 확인할 때 이것 하나를 본다.
순서: 결론 → 1 탐지 → 2 원인 분석 → 3 APO(후보·테스트 표·출력 예시·제안·diff) → 4 반영 후 확인(apo_test.py verify) → 5 사람이 할 일
읽는 것: case.json · trace.md · triage.md · runs/apo/*_<케이스>_i<번호>/(source.json · <후보>/diff.patch · results.jsonl · proposal.md)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.config import ROOT
from scripts.apo_test import table
from scripts.triage_save import ReportError, parse

NEXT = {"apo": "지시문 수정 후보 테스트 (APO)", "human": "사람 검토 (코드·가이드·문서)", "judge_set": "판정기 검증 세트에 추가",
        "none": "조치 없음"}


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def _experiment(case_dir: Path, issue: int) -> Path | None:
    """이 케이스·문제의 가장 최근 APO 실험 폴더."""
    found = sorted((ROOT / "runs" / "apo").glob(f"*_{case_dir.name}_i{issue}")) if (ROOT / "runs" / "apo").exists() else []
    return found[-1] if found else None


def _changed(diff: str) -> tuple[int, int]:
    lines = diff.splitlines()
    return (sum(l.startswith("-") and not l.startswith("---") for l in lines),
            sum(l.startswith("+") and not l.startswith("+++") for l in lines))


def _target_cells(exp: Path) -> dict[str, str]:
    """arm → target 결과 (마지막 실행)."""
    if not (exp / "results.jsonl").exists():
        return {}
    out = {}
    for line in (exp / "results.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] == "repro" and r["test"] == "target" and not r.get("error"):
            out[r["arm"]] = f"{r['fails']}/{r['n']}"
    return out


def _outputs(exp: Path) -> list[str]:
    """테스트별·arm별 실제 출력 첫 문장 (실험 내용 자체 — 숫자만으로 고르지 않도록)."""
    rows = [json.loads(l) for l in (exp / "results.jsonl").open(encoding="utf-8")]
    latest = {(r["arm"], r["test"]): r for r in rows if r["kind"] == "repro" and r.get("texts")}
    if not latest:
        return []
    out = ["<details><summary>출력 예시 (테스트·arm별 첫 반복)</summary>", ""]
    for t in dict.fromkeys(t for _, t in latest):
        out.append(f"**{t}**")
        for (arm, test), r in latest.items():
            if test == t:
                out.append(f"- `{arm}` ({r['fails']}/{r['n']} 실패): {r['texts'][0][:200]}")
        out.append("")
    return out + ["</details>", ""]


def _recommendation(exp: Path) -> str | None:
    """제안서의 「추천: …」 제목 줄 (없으면 None)."""
    if not (exp / "proposal.md").exists():
        return None
    m = re.search(r"^#+ *추천[:：]? *(.+)$", (exp / "proposal.md").read_text(encoding="utf-8"), re.M)
    return m[1].strip() if m else None


def _proposal_status(exp: Path) -> str:
    text = (exp / "proposal.md").read_text(encoding="utf-8") if (exp / "proposal.md").exists() else ""
    if not text:
        return "제안 없음"
    tested = (exp / "results.jsonl").exists()
    return "테스트 후 최종 제안" if tested and "테스트 전" not in text.split("\n## ")[0] else "원칙 점검 단계 제안 (테스트 전)"


def build(case_dir: Path) -> str:
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    triage_path = case_dir / "triage.md"
    head = None
    if triage_path.exists():
        try:
            head, _ = parse(triage_path.read_text(encoding="utf-8"))
        except ReportError as e:
            head = {"error": str(e)}
    issues = (head or {}).get("issues") or []
    out = [f"# 케이스 리포트 — {case['case']}", ""]

    # -- 결론
    out += ["## 결론", ""]
    if head is None:
        out += ["- triage 전 — 탐지 신호만 있다", ""]
    elif "error" in head:
        out += [f"- triage 보고서 형식 오류: {head['error']}", ""]
    else:
        out += ["| # | 신호 | 원인 층 | 확인 | 다음 | 진행 |", "|---|---|---|---|---|---|"]
        for n, it in enumerate(issues):
            layer = it["layer"] + (f" (+{it['secondary']})" if it.get("secondary") else "")
            conf = f"재현됨 {it.get('baseline', '')}" if it["confidence"] == "reproduced" else "가설"
            progress = it["proposal"]
            exp = _experiment(case_dir, n) if it["next"] == "apo" else None
            if it["next"] == "apo":
                if exp is None:
                    progress = "APO 실험 전"
                else:
                    cells = _target_cells(exp)
                    base = cells.pop("baseline", "?")
                    rec = _recommendation(exp)
                    progress = _proposal_status(exp) + (f" · **추천: {rec}**" if rec else "") + (
                        f" — 재현 기준 실패: 지금 {base} → " + ", ".join(f"{a} {c}" for a, c in sorted(cells.items()))
                        if cells else "")
            out.append(f"| {n + 1} | {it['signal']} | {layer} | {conf} | {NEXT[it['next']]} | {progress} |")
        out += ["", "반영은 사람이 결정한다 — 아래 각 문제의 제안과 근거를 보고 판단.", ""]

    # -- 1. 탐지
    out += ["## 1. 탐지", "", f"- 시나리오 `{case.get('scenario')}` · 결과 `{case.get('disposition')}` · 종료 `{case.get('end_reason')}`",
            f"- 버전: `{json.dumps(case.get('version'), ensure_ascii=False)}`", ""]
    out += ["| 등급 | 신호 | 좌표 | 내용 |", "|---|---|---|---|"]
    out += [f"| {s['level']} | {s['kind']} | `{s['at'] or '-'}` | {s['detail'][:80]} |" for s in case["signals"]]
    if case.get("guards"):
        out += ["", "가드레일 발동 (막거나 고친 곳): " + ", ".join(f"{k} {v}" for k, v in case["guards"].items())]
    if (case_dir / "trace.md").exists():
        trace = (case_dir / "trace.md").read_text(encoding="utf-8")
        body = trace[trace.find("```mermaid"):] if "```mermaid" in trace else ""
        out += ["", "<details><summary>통화 흐름도 (노드 id = 좌표)</summary>", "", body.strip(), "", "</details>"]
    out.append("")

    # -- 2. 원인 분석
    out += ["## 2. 원인 분석 (triage)", ""]
    if not issues:
        out += ["- 없음", ""]
    for n, it in enumerate(issues):
        out += [f"### 문제 {n + 1} — {it['signal']}", "", f"**{it['summary']}**", "",
                f"- 원인 층: `{it['layer']}`" + (f" · 보조 `{it['secondary']}`" if it.get("secondary") else ""),
                f"- 확인: {'재현됨 ' + str(it.get('baseline', '')) if it['confidence'] == 'reproduced' else '가설'}"
                + (f" · 기준 명세 `{it['spec']}`" if it.get("spec") else ""),
                f"- 제안: {it['proposal']}", f"- 다음: {NEXT[it['next']]}", ""]
    if triage_path.exists():
        out += [f"근거·판단 이유 전문: `{_rel(triage_path)}`", ""]

    # -- 3. APO
    apo_issues = [(n, it) for n, it in enumerate(issues) if it["next"] == "apo"]
    if apo_issues:
        out += ["## 3. 지시문 개선 (APO)", ""]
    for n, it in apo_issues:
        exp = _experiment(case_dir, n)
        out += [f"### 문제 {n + 1} — {it['signal']}", ""]
        if exp is None:
            out += ["- 실험 전", ""]
            continue
        out += [f"- 실험 폴더: `{_rel(exp)}` · 상태: {_proposal_status(exp)}", ""]
        cands = sorted(d for d in exp.iterdir() if d.is_dir() and (d / "diff.patch").exists())
        if cands:
            out += ["| 후보 | 바뀐 줄 |", "|---|---|"]
            for d in cands:
                minus, plus = _changed((d / "diff.patch").read_text(encoding="utf-8"))
                out.append(f"| {d.name} | −{minus} +{plus} |")
            out.append("")
        if (exp / "results.jsonl").exists():
            out += ["**테스트 결과** (실패 수 / 반복 — 재현 술어 기준. 출력 검토 결과는 최종 제안에)", "", table(exp), ""]
            out += _outputs(exp)
        if (exp / "proposal.md").exists():
            prop = re.sub(r"^(#{1,3}) ", lambda m: "#" * (len(m[1]) + 3) + " ",
                          (exp / "proposal.md").read_text(encoding="utf-8").strip(), flags=re.M)   # 리포트 제목 아래로
            out += ["**제안**", "", prop, ""]
        for d in cands:
            out += [f"<details><summary>{d.name} diff</summary>", "", "```diff",
                    (d / "diff.patch").read_text(encoding="utf-8").strip(), "```", "", "</details>", ""]

    # -- 4. 반영 후 확인
    applied = [(n, it, _experiment(case_dir, n)) for n, it in apo_issues]
    applied = [(n, it, e) for n, it, e in applied if e is not None and (e / "applied.json").exists()]
    if applied:
        out += ["## 4. 반영 후 확인", ""]
    for n, it, exp in applied:
        info = json.loads((exp / "applied.json").read_text(encoding="utf-8"))
        same = all(info["files_match"].values())
        out += [f"### 문제 {n + 1} — 후보 `{info['candidate']}` 반영 (커밋 {info['commit']})", "",
                f"- 저장소 지시문 = 후보: {'예' if same else '아니오 — 반영 내용이 후보와 다르다 ' + str(info['files_match'])}", ""]
        rows = [json.loads(l) for l in (exp / "results.jsonl").open(encoding="utf-8")]
        latest = {(r["kind"], r["arm"], r["test"]): r for r in rows}
        tests = list(dict.fromkeys(r["test"] for r in rows if r["arm"] == "applied"))
        val = lambda r: "-" if r is None or r.get("error") else (
            f"{r['fails']}/{r['n']}" if r["kind"] == "repro" else f"통과 {r['passed']}/{r['n']}")
        out += ["| 테스트 | 반영 전 (baseline) | 테스트 때 후보 | 반영 후 | 재현 |", "|---|---|---|---|---|"]
        for t in tests:
            kind = "repro" if ("repro", "applied", t) in latest else "regression"
            b, c, a = (latest.get((kind, arm, t)) for arm in ("baseline", info["candidate"], "applied"))
            ok = "-"
            if c and a and not c.get("error") and not a.get("error"):
                key = "fails" if kind == "repro" else "passed"
                ok = "같음" if c[key] == a[key] else ("나아짐" if (a[key] < c[key]) == (kind == "repro") else "**나빠짐**")
            out.append(f"| {t} | {val(b)} | {val(c)} | {val(a)} | {ok} |")
        out.append("")

    # -- 5. 사람이 할 일
    out += [f"## {5 if applied else 4}. 사람이 할 일", ""]
    def apo_todo(n: int) -> str:
        exp = _experiment(case_dir, n)
        rec = _recommendation(exp) if exp else None
        if exp is None:
            return "APO 실험 (apo → apo-test)"
        if rec and rec.startswith("없음"):
            return "추천 후보 없음 — 최종 제안의 「다음에 시도할 방향」·「사람이 정할 것」을 보고 APO 재실행 / 다른 층 검토 중 결정"
        return ("APO 최종 제안 검토 → 승인하면 저장소에 반영 → `apo_test.py verify <실험> --candidate <후보>` → "
                "이 리포트 다시 생성(4절)")

    todo = [f"- 문제 {n + 1} ({it['signal']}): " + (apo_todo(n) if it["next"] == "apo" else
                                                 f"{NEXT[it['next']]} — {it['proposal']}")
            for n, it in enumerate(issues) if it["next"] != "none"
            and not (it["next"] == "apo" and any(m == n for m, _, _ in applied))]
    out += [f"- 문제 {n + 1}: 반영 확인 완료 — 위 4절 표" for n, _, _ in applied]
    out += todo or ([] if applied else ["- 없음"])
    return "\n".join(out) + "\n"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    case_dir = Path(sys.argv[1])
    case_dir = case_dir if case_dir.is_absolute() else ROOT / case_dir
    if not (case_dir / "case.json").exists():
        sys.exit(f"케이스 폴더가 아니다: {case_dir}")
    (case_dir / "report.md").write_text(build(case_dir), encoding="utf-8")
    print(f"리포트: {_rel(case_dir / 'report.md')}")


if __name__ == "__main__":
    main()
