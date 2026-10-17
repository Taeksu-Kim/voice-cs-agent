"""로그 폴더의 지표를 터미널에 보여 준다. 평가 실행 중이면 --follow로 몇 초마다 갱신.

    python scripts/monitor.py                      # 가장 최근 로그 폴더
    python scripts/monitor.py runs/20261013_014436_C --follow
    python scripts/monitor.py logs/20261013 --follow --interval 5
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.config import ROOT
from csagent.monitoring import MAX_REPLY_CHARS, aggregate, list_sources


def render(m: dict) -> str:
    out = [f"■ {m['source']}   갱신 {datetime.fromtimestamp(m['updated']):%H:%M:%S}"]
    if m["first_ts"]:
        span = (m["last_ts"] - m["first_ts"]) / 60
        out[-1] += f"   경과 {span:.1f}분 (마지막 기록 {datetime.fromtimestamp(m['last_ts']):%H:%M:%S})"

    if ev := m.get("eval"):
        out.append(f"\n[평가] 완료 {ev['done']}회  통과 {ev['passed']}/{ev['done']}"
                   f" ({ev['passed'] / ev['done']:.0%})  시뮬레이터 이탈 {ev['sim_drift']}회")
        for sid, s in ev["scenarios"].items():
            fails = ", ".join(f"{k}×{v}" for k, v in sorted(s["failed"].items(), key=lambda x: -x[1]))
            drift = f"  [이탈 {s['sim_drift']}]" if s["sim_drift"] else ""
            out.append(f"  {sid:<26}{s['passed']:>2}/{s['n']:<2} {fails}{drift}")

    c, t = m["calls"], m["turns"]
    out.append(f"\n[통화] 시작 {c['started']}  종료 {c['ended']}  결과 {_kv(c['dispositions'])}")
    out.append(f"[턴] {t['n']}턴  지연 p50 {t['ms_total']['p50']:,}ms  p90 {t['ms_total']['p90']:,}ms"
               f"  max {t['ms_total']['max']:,}ms  | 첫 글자 p50 {t['ms_first_token']['p50']:,}ms"
               f"  | LLM 스텝 p50 {t['llm_steps']['p50']}  | {MAX_REPLY_CHARS}자 초과 {t['over_chars']}턴")

    llm = m["llm"]
    out.append(f"\n[LLM] 최근 60초: 호출 {llm['last_60s']['calls']}  "
               f"토큰 {llm['last_60s']['prompt_tokens']:,}+{llm['last_60s']['completion_tokens']:,}")
    out.append(f"  {'출처':<9}{'호출':>6}{'입력 토큰':>12}{'출력 토큰':>10}{'p50':>9}{'p90':>9}{'오류':>5}")
    for src, s in llm["by_source"].items():
        out.append(f"  {src:<9}{s['calls']:>6}{s['prompt_tokens']:>12,}{s['completion_tokens']:>10,}"
                   f"{s['ms']['p50']:>7,}ms{s['ms']['p90']:>7,}ms{s['errors']:>5}")

    i, tl = m["incidents"], m["tools"]
    out.append(f"\n[사건] 폴백 {_kv(i['fallback']) or 0}  절차 강제 {i['step_enforced']}  스텝 소진 {i['max_steps']}"
               f"  라우팅 실패 {i['route_failed']}  되묻기 {i['ask_again']}  업무 전환 {_kv(i['task_switch']) or 0}"
               f"  LLM 오류 {i['llm_error']}  턴 오류 {i['turn_error']}")
    out.append(f"[도구] {_kv(tl['status'])}")
    if tl["blocked"]:
        out.append(f"  차단: {_kv(tl['blocked'])}")
    if tl["failed"]:
        out.append(f"  실패: {_kv(tl['failed'])}")

    if m["warnings"]:
        out.append("\n[최근 경고]")
        for w in m["warnings"][-8:]:
            where = " ".join(str(w[k]) for k in ("scenario", "call_id", "turn") if w.get(k) is not None)
            detail = w.get("reason") or w.get("error") or w.get("output", "")
            out.append(f"  {datetime.fromtimestamp(w['ts']):%H:%M:%S} {w['level']:<7} {w['event']:<14}"
                       f" {where}  {str(detail)[:80]}")
    return "\n".join(out)


def _kv(d: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in sorted(d.items(), key=lambda x: -x[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", help="로그 폴더 (runs/… 또는 logs/…). 없으면 가장 최근")
    ap.add_argument("--follow", action="store_true", help="계속 갱신")
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()
    if args.source:
        log_dir = Path(args.source)
        log_dir = log_dir if log_dir.is_absolute() else ROOT / log_dir
    else:
        latest = next((s for s in list_sources(ROOT) if s["has_events"]), None)
        if latest is None:
            sys.exit("이벤트 로그가 있는 폴더가 없습니다")
        log_dir = ROOT / latest["id"]
    while True:
        text = render(aggregate(log_dir))
        if not args.follow:
            print(text)
            return
        print("\033[2J\033[H" + text, flush=True)   # 화면 지우고 다시 그림
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
