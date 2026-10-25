---
name: triage
description: 이상 탐지가 만든 케이스 묶음(runs/cases/<시각>/<케이스>/)을 받아, 로그로 원인을 찾고 재현으로 확인한 뒤 원인 층(지시문·가이드·문서·코드·판정기·환경)과 개선 제안을 보고서로 남긴다. 아무것도 고치지 않는다. scripts/detect.py가 케이스를 만들었을 때, 또는 사람이 특정 통화·케이스의 원인 분석을 요청할 때 쓴다.
allowed-tools: Read, Grep, Glob, Bash(grep:*), Bash(ls:*), Bash(/home/ubuntu/miniconda3/envs/customer-service-agent/bin/python scripts/repro.py:*), Bash(/home/ubuntu/miniconda3/envs/customer-service-agent/bin/python scripts/trace_graph.py:*), Bash(/home/ubuntu/miniconda3/envs/customer-service-agent/bin/python scripts/triage_save.py:*), Bash(bash scripts/serve.sh status), Bash(git log:*), Bash(git show:*), Bash(git diff:*)
---

# triage — 케이스 원인 분석과 개선 제안

입력: 케이스 폴더 하나 (`runs/cases/<시각>/<출처>_<키>/`). 인자로 받는다. 없으면 가장 최근 `runs/cases/*/summary.md`를 읽고 bad 신호가 있는 케이스부터 하나씩.

**이 스킬은 읽고, 재현하고, 보고서를 남긴다. 코드·지시문·가이드·문서·시나리오를 고치지 않는다.**
지시문이 원인이면 보고서에 그렇게 적고 끝낸다 — 지시문 수정·실험은 별도 스킬(APO)이 이 보고서를 입력으로 받아 한다.

## 도구 쓰는 법 (권한이 이 범위로 묶여 있다)

- 파일 보기는 **Read**, 검색은 Grep 도구(없으면 `grep -n <패턴> <파일>` 한 줄). `llm_calls.jsonl`처럼 큰 파일은 Read(`allow_large`)로 읽거나 grep으로 call_id를 찾는다
- 플러그인·MCP 도구(ctx_execute 등)는 쓰지 않는다 — 권한이 없다
- **백그라운드 실행(run_in_background)·Monitor를 쓰지 않는다. 명령은 하나씩 끝까지 기다린다** (Bash timeout 600000) —
  비대화형 실행이라 알림을 기다리면 세션이 끝나고 이어진 세션에서는 권한이 풀린다. 재현은 로컬 벡터 DB를 하나만 열 수 있어 동시에 돌리면 실패한다
- 명령은 아래 형태 **그대로 한 줄에 하나**. 앞에 `cd`를 붙이거나, 변수·`&&`·`;`·파이프로 묶거나, 임시 스크립트를 쓰면 거부된다
  (작업 디렉터리는 이미 프로젝트 루트다)
  - `/home/ubuntu/miniconda3/envs/customer-service-agent/bin/python scripts/repro.py <case> --at t3 --mode turn --repeat 5 --fail repeat=true`
  - `/home/ubuntu/miniconda3/envs/customer-service-agent/bin/python scripts/trace_graph.py <출처 폴더> <키>`
  - `/home/ubuntu/miniconda3/envs/customer-service-agent/bin/python scripts/triage_save.py <케이스 폴더>` (보고서는 heredoc으로)
  - `bash scripts/serve.sh status` · `git log …` · `git diff <commit> -- <경로>`
- 파일은 만들거나 고치지 않는다 (케이스의 `repro.yaml`도). 재현 명세는 명령줄 인자로 주고, **결과 폴더(`runs/repro/<…>/`)의 `spec.json`이 기준 명세**가 된다

## 먼저 읽을 것

- `reference/system_map.md` — 무엇이 어디 있는지 (지시문·프로토콜·가드레일·도구·판정기), 기록 형식, 쓸 수 있는 명령
- `reference/layers.md` — 원인 층 분류 기준과 경계 사례
- `reference/report_template.md` — 보고서 형식 (저장 스크립트가 이 형식만 받는다)

## 순서

1. **케이스 읽기** — `case.json`(신호·좌표·가드레일 발동·버전) → `trace.md`(흐름도) → 신호 좌표 턴의 `llm_calls.jsonl` 원문(입력 메시지·출력).
   버전의 commit이 지금 코드와 다르면 `git log`·`git diff <commit>`으로 그 사이 바뀐 것을 본다 — 이미 고쳐졌을 수 있다
2. **판정기 오판인가** — 신호가 채점 항목(`check:*`)이면, 판정 근거(주장·인용)를 원문·도구 결과와 직접 대조한다.
   오판이면 층 = 판정기, 3번 건너뛰고 5번으로
3. **재현으로 기준치** — `repro.yaml` 초안(가장 심한 신호 하나만 담겨 있다)을 참고해, 신호마다 구간·술어를 맞춘 명세를 명령줄 인자로 `scripts/repro.py`에 준다. 기본 5회.
   먼저 `bash scripts/serve.sh status`로 모델 서버를 확인한다 (꺼져 있으면 재현하지 않고 확신도 = 가설)
   **결함이 앞 턴 대화에 의존하면**(반복·되풀이처럼 상담원 자기 앞 말이 원인) 재현 기준은 그 전 턴부터 continue 모드(`customer: recorded`)로 잡는다.
   turn 모드는 앞 턴을 기록된 말로 고정해서, 지시문을 바꾼 뒤 앞 말부터 달라지는 경우를 보지 못한다 (10/25 s3_b0_defect: turn 0/5, continue 3/3)
   - 재현되면(예: 5/5) 확신도 = 재현됨. 이 명세와 기준치가 수정 전후 비교의 기준이 된다 — **이후 바꾸지 않는다**
   - 재현이 안 되면 고객 쪽 변수부터 의심한다 (continue 모드는 `customer: recorded`로 고객 말 고정). 그래도 안 되면 확신도 = 가설
4. **원인과 층** — 그 지점의 LLM 입력에 무엇이 있었고 무엇이 없었는지로 원인을 찾는다:
   지시문에 해당 규칙이 없다 / 있는데 안 지켰다 / 근거 문서가 검색에 안 들어왔다 / 코드가 말을 이어 붙였다 / 회사가 정하지 않은 영역이다 …
   층은 `reference/layers.md` 기준으로 하나를 주 원인으로, 같은 문제에 원인이 더 걸려 있으면 보조 원인을 하나 더.
   **한 케이스의 신호들이 원인이 서로 다른 문제면 문제를 나눈다** (보고서 issues 항목 하나씩 — 각자 층·재현·제안)
5. **개선 제안** — 층에 맞는 제안만. 지시문 층이면 「무엇을 어떤 방향으로」까지 (문구 최종안은 APO 몫).
   가이드 층이면 사람이 정할 질문 형태로 (「타사 문의에 어디까지 안내해도 되나」)
6. **저장** — 보고서를 `reference/report_template.md` 형식으로 써서 저장 스크립트에 넘긴다:
   ```
   /home/ubuntu/miniconda3/envs/customer-service-agent/bin/python scripts/triage_save.py <케이스 폴더> <<'EOF'
   (보고서 전문)
   EOF
   ```
   케이스 폴더의 `triage.md`와 누적 목록 `runs/triage/index.jsonl`에 남는다. 판정기 오판·조치 없음으로 끝난 것도 저장한다

## 하지 않는 것

- 무엇이든 수정하지 않는다 (재현 실행과 보고서 저장만)
- 실패를 통과시키려고 「이렇게 말해도 된다」는 가이드·예외를 제안하지 않는다 — 허용 범위를 넓히는 것은 회사(사람)가 정할 질문으로만 올린다
- 재현 기준(fail_if)을 결과를 본 뒤 바꾸지 않는다
- 원인을 모르면 모른다고 쓴다 (확신도 = 가설, 무엇을 더 보면 가려지는지 적는다)
- 한 케이스를 근거로 넓은 일반화를 하지 않는다 — 같은 원인이 다른 케이스에도 있는지는 `runs/triage/index.jsonl`을 검색해서 적는다 (파일이 없으면 첫 보고서 — 「없음」)
