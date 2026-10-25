---
name: apo-test
description: apo 스킬이 만든 지시문 후보들의 테스트를 계획하고(plan), 실행기가 돌린 결과를 검토해 후보를 고른다(review). 테스트 실행 자체는 Claude 밖의 고정 실행기(scripts/apo_test.py run-all)가 한다. 저장소 지시문은 고치지 않는다 — 반영은 사람이 결정한다. 인자 — `plan <실험 폴더>` 또는 `review <실험 폴더>`.
allowed-tools: Read, Grep, Glob, Bash(grep:*), Bash(ls:*), Bash(/home/ubuntu/miniconda3/envs/customer-service-agent/bin/python scripts/apo_test.py plan:*), Bash(/home/ubuntu/miniconda3/envs/customer-service-agent/bin/python scripts/apo_test.py table:*), Bash(/home/ubuntu/miniconda3/envs/customer-service-agent/bin/python scripts/apo_candidate.py proposal:*)
---

# apo-test — 지시문 후보 테스트 계획과 결과 검토

입력: `plan <실험 폴더>` 또는 `review <실험 폴더>` (`runs/apo/<시각>_<케이스>_i<번호>/`).
실험 폴더에는 `source.json`(triage 문제·재현 기준), 후보 폴더들, `proposal.md`(apo의 원칙 점검 단계 제안)가 있다.

**흐름** — Claude는 계획과 판단만, 실행은 고정 실행기가 한다 (재현은 몇 분~수십 분이 걸리고 동시에 못 돌린다):

```
/apo-test plan <실험>      → tests.yaml (이 스킬)
apo_test.py run-all <실험>  → results.jsonl (실행기 — Claude 밖, 실행 스크립트가 부른다)
/apo-test review <실험>    → 출력 검토 · 선택 · proposal.md (이 스킬)
```

**저장소 지시문을 고치지 않는다.** 후보는 실행기가 `CSAGENT_PROTOCOL_OVERRIDE`로 끼워 돌릴 뿐이다. 이 스킬은 재현·평가를 직접 실행하지 않는다.

## 도구 쓰는 법

- 파일 보기는 Read, 검색은 Grep(없으면 `grep -n` 한 줄). 플러그인·MCP 도구, 백그라운드 실행, Monitor는 쓰지 않는다
- 명령은 한 줄에 하나, `cd`·변수·`&&`·파이프 없이:
  - `/home/ubuntu/miniconda3/envs/customer-service-agent/bin/python scripts/apo_test.py plan <실험 폴더>` + heredoc으로 tests.yaml
  - `… apo_test.py table <실험 폴더>` — 결과 표
  - `… apo_candidate.py proposal <실험 폴더>` + heredoc — 최종 제안 (apo의 제안을 덮어쓴다)

## plan

1. **읽기** — `source.json`, `proposal.md`(특히 「테스트 단계에서 볼 것」), 후보마다 `diff.patch`·`rationale.md`,
   triage 케이스의 `triage.md`, 재현 기준 명세(`source.json`의 `spec`)와 그 결과 폴더의 `repro.jsonl`(지금 지시문의 실패 출력)
2. **tests.yaml** (`plan`):
   - `target`: `source.json`의 `spec` 그대로, repeat 3 (온도 0이라 거의 결정적). **바꾸지 않는다**
   - `variants` 3~4개: **같은 상황(그 턴 직전까지 같은 대화)에서 고객이 다르게 말하는 경우.** 실패 술어에 걸리면 정말 잘못된 행동인 말만 고른다.
     예: 같은 요구를 다른 표현으로 / 같은 요구에 다른 하위 질문 / 더 짧거나 거친 말 / 같은 요구만 되풀이하는 말(이때 같은 답은 정상 — 술어가 거른다).
     후보 문구나 케이스 답을 겨냥한 말을 만들지 않는다 (후보에 유리한 시험이 된다). 각 변형에 `why`를 단다. repeat 2
   - **부작용 변형** 1~2개 (variants에 같이): 후보 rationale의 「새로 생길 수 있는 실패」·「지운 문장이 막던 과거 실패」를 드러내는 고객 말.
     이런 실패(지어내기 등)는 대개 재현 술어에 안 걸린다 — review의 출력 검토에서 이 변형을 먼저 본다
   - `guards`: 같은 통화의 다른 문제 재현 명세(triage의 다른 issue `spec`) 등 퇴행이 걱정되는 곳. repeat 2.
     **결함이 앞 턴 대화에 의존하면**(반복·되풀이처럼 자기 앞 말이 원인) target이 turn 모드여도 그 전 턴부터 continue 모드 명세를 guard로 꼭 넣는다 —
     turn 모드는 앞 턴 이력을 지금 지시문이 만든 말로 고정해서, 후보가 만든 앞 말에서 같은 결함이 나는지를 보지 못한다 (10/25 s3_b0_defect 2차: target 0/5였지만 continue 3/3)
   - `regression.scenarios`: 후보가 바꾼 단계·분기를 지나는 시나리오만 (`scenarios/<업무>.yaml`의 id와 기대값을 보고 고른다, 보통 2~3개). repeat 1.
     채점은 규칙만(기본) — 말 내용은 review에서 출력으로 본다. 술어가 LLM 판정 항목이면 `judge: true`
   - `suite: <업무>`: 케이스 모음(`runs/suite/<업무>.yaml`)이 있으면 넣는다 — 그 업무의 알려진 결함·지켜야 할 행동이 자동으로 들어간다
3. 계획을 저장하고 끝낸다 (실행하지 않는다)

## review

실행기가 돈 뒤 불린다. 실행기는 **단계별로 거른다**: target에서 baseline보다 나아지지 않은 후보는 이후 테스트를 건너뛰고, 회귀는 baseline과 target 실패 0인 후보만 돈다.

1. **표** — `table`. 빈 칸·오류 칸이 있으면 그 사실과 이유(건너뜀 / 오류)를 적는다. baseline target이 재현 기준(`source.json`의 `baseline`)과 크게 다르면 그 사실을 적는다
2. **출력 검토** — 실패 수만 보지 않는다. 각 칸의 새 출력(`results.jsonl`의 `out` 폴더 → `repro.jsonl`의 texts)을 그 턴의 근거 — triage 케이스 `llm_calls.jsonl`의 [참고 문서]·도구 결과 — 와 대조한다:
   문서에 없는 절차·화면·조건을 지어냈나, 확답·약속을 했나, 고객 말에 실제로 답했나, 너무 길어졌나, 필수 안내를 빠뜨렸나.
   술어를 통과했어도 이런 문제가 있으면 그 칸은 **실패로 본다** (표에 따로 적는다). 부작용 변형부터 본다
3. **회귀** — 통과 수와 새로 실패한 채점 항목(특히 grounded·no_repeat·disposition·milestones). 회귀 결과 폴더의 results.jsonl에서 실패 내용을 본다
4. **선택과 제안** (`proposal`) — 고르는 기준, 순서대로:
   ① target 실패가 baseline보다 줄었다 ② 변형에서 나빠지지 않았다 ③ guard·회귀에서 퇴행이 없다 ④ 출력 검토 문제 없음 ⑤ 바뀐 줄이 적다.
   모두 만족하는 후보가 없으면 「추천 없음」과 이유, 다음에 시도할 방향(또는 triage 층 재검토)을 쓴다.
   제안 맨 위: `## 추천: <후보>` 또는 `## 추천 없음` · 결과 표 · 확신도(테스트 횟수가 작으면 그렇게 쓴다).
   아래: 출력 검토 · 회귀 상세 · 남은 위험 · 사람이 정할 것(apo 제안에 있던 결정 포함) · 다음 라운드에 넘길 것(추천 없음일 때)

## 하지 않는 것

- 저장소 지시문·코드를 고치지 않는다. 후보 내용도 고치지 않는다 — 고쳐야 하면 「apo 재실행: 이런 방향」을 제안에 적는다
- 재현·평가를 직접 실행하지 않는다 (실행기 몫). 결과가 비어 있으면 「실행 전」으로 적고 끝낸다
- target 명세·실패 기준을 바꾸지 않는다. 결과를 본 뒤 계획을 바꾸지 않는다 (다시 짜야 하면 이유를 남기고 계획 전체를 새로)
- 실패 수 차이가 반복 횟수에 비해 작으면(예: 3회 중 1회 차이) 효과가 있다고 쓰지 않는다
