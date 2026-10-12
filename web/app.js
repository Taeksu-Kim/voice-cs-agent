// 상담 에이전트 콘솔 — 세션을 열고, 발화마다 SSE 스트림(POST)을 읽어 화면에 그린다.
const $ = (id) => document.getElementById(id);
const state = { sessionId: null, scenarios: [], controller: null, turnNo: 0 };

async function api(path, body) {
  const res = await fetch(path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : undefined);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res;
}

// ---------------------------------------------------------------- 시작

async function loadScenarios() {
  state.scenarios = await (await api("/api/scenarios")).json();
  $("scenario").innerHTML = state.scenarios
    .map((s) => `<option value="${s.id}">${s.task}·${s.behavior} — ${esc(s.goal).slice(0, 40)}</option>`)
    .join("");
}

async function startCall() {
  const scenario = state.scenarios.find((s) => s.id === $("scenario").value);
  const data = await (await api("/api/sessions", { scenario_id: scenario.id })).json();
  state.sessionId = data.session_id;
  state.turnNo = 0;
  $("chat").innerHTML = "";
  $("turns").innerHTML = "";
  $("callstate").innerHTML = "";
  const orders = (data.customer.orders || [])
    .map((o) => `<li>${esc(o.item)} — ${esc(o.status)}${o.eta ? ` (예정 ${o.eta})` : ""}</li>`).join("");
  $("customer").innerHTML =
    `<b>${esc(data.customer.name || "고객")}</b> · 오늘 ${data.today} · ${esc(scenario.goal)}` +
    (orders ? `<ul class="orders">${orders}</ul>` : `<div class="muted">주문 없음</div>`);
  $("customer").classList.remove("muted");
  setComposer(true);
  $("utterance").value = scenario.opening || "";
  $("utterance").focus();
  addMessage("system", "통화가 연결됐습니다.");
  setStatus("통화 중");
}

// ---------------------------------------------------------------- 한 턴

async function sendTurn(text) {
  addMessage("customer", text);
  const bubble = addMessage("agent", "");
  bubble.classList.add("streaming");
  const card = addTurnCard(++state.turnNo, text);
  setComposer(false, true);
  setStatus("응답 생성 중…");

  state.controller = new AbortController();
  const t0 = performance.now();
  let firstDeltaMs = null;
  try {
    const res = await fetch(`/api/sessions/${state.sessionId}/turn`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ utterance: text }), signal: state.controller.signal,
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    for await (const [event, data] of readSSE(res)) {
      if (event === "delta") {
        firstDeltaMs ??= performance.now() - t0;
        bubble.firstChild.textContent += data.text;
        scrollChat();
      } else if (event === "task") renderTask(card, data);
      else if (event === "retrieval") renderRetrieval(card, data);
      else if (event === "step") setStatus(`${data.worker} · LLM 생성 중 (스텝 ${data.step})`);
      else if (event === "trace") renderTrace(card, data);
      else if (event === "done") { finishTurn(bubble, card, data, firstDeltaMs); renderState(data.state); }
      else if (event === "error") throw new Error(data.message);
    }
  } catch (e) {
    const aborted = e.name === "AbortError";
    bubble.classList.remove("streaming");
    addMeta(bubble, aborted ? "중단됨 (이 턴은 기록되지 않음)" : `오류: ${e.message}`);
    setStatus(aborted ? "중단됨" : "오류", aborted ? "warn" : "err");
    setComposer(true);
  } finally {
    state.controller = null;
    $("stop").disabled = true;
  }
}

function finishTurn(bubble, card, reply, firstDeltaMs) {
  bubble.classList.remove("streaming");
  bubble.firstChild.textContent = reply.text;  // 스트림 조각 대신 확정 문장으로 맞춘다
  const t = reply.timings_ms;
  card.insertAdjacentHTML("beforeend", `<div class="timings">
      ${chip("검색", t.retrieve)} ${chip("첫 글자", t.first_token)} ${chip("LLM", t.llm)} ${chip("전체", t.total)}
      <span class="chip">토큰 ${reply.tokens.prompt}+${reply.tokens.completion}</span>
      ${reply.disposition ? `<span class="chip ok">${reply.disposition}</span>` : ""}
    </div>`);
  if (reply.session_ended) {
    addMessage("system", `통화 종료 (${reply.disposition || "-"})`);
    setStatus("통화 종료", "ok");
    setComposer(false);
  } else {
    setStatus("통화 중");
    setComposer(true);
  }
}

// ---------------------------------------------------------------- SSE (POST 응답 본문을 직접 읽는다)

async function* readSSE(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) yield [event, JSON.parse(data)];
    }
  }
}

// ---------------------------------------------------------------- 그리기

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.appendChild(document.createTextNode(text));
  $("chat").appendChild(el);
  scrollChat();
  return el;
}

function addMeta(bubble, text) {
  const m = document.createElement("span");
  m.className = "meta";
  m.textContent = text;
  bubble.appendChild(m);
}

function addTurnCard(no, text) {
  const card = document.createElement("div");
  card.className = "turn";
  card.innerHTML = `<h3>턴 ${no} · <span class="muted">${esc(text)}</span></h3>`;
  $("turns").prepend(card);
  return card;
}

function renderTask(card, data) {
  const label = data.attempt ? `업무 전환 → <b>${esc(data.task)}</b>` : `업무: <b>${esc(data.task)}</b>`;
  card.insertAdjacentHTML("beforeend", `<div class="label">${label}</div>`);
}

function renderRetrieval(card, data) {
  const items = data.docs.map((d) =>
    `<li>${d.score.toFixed(3)} · ${esc(d.title)}${d.section ? ` <span class="label">› ${esc(d.section)}</span>` : ""}</li>`).join("");
  card.insertAdjacentHTML("beforeend", `<div><span class="label">검색 ${data.ms}ms</span><ul>${items}</ul></div>`);
}

// 추적 기록 한 줄 — 에이전트가 이번 턴에 무엇을 했는지 시간순으로
const TRACE = {
  route: (d) => ["🧭", `라우팅 → <b>${esc(d.task)}</b>`, d.reason],
  task_start: (d) => ["▶", `업무 시작 <b>${esc(d.task)}</b>`],
  task_done: (d) => ["✔", `업무 완료 <b>${esc(d.task)}</b>`],
  retrieval: (d) => ["🔎", `검색 (${esc(d.worker)}) ${d.chunk_ids.length}건`],
  llm: (d) => ["🤖", `LLM ${esc(d.role)} ${d.ms}ms`, `토큰 ${d.prompt_tokens}+${d.completion_tokens}`],
  intent: (d) => ["↪", `업무 전환 요청 (${esc(d.worker)})`],
  step_enforced: (d) => ["⛔", `절차 강제: 먼저 <b>${esc(d.tool)}</b>`, "", "warn"],
  pending: (d) => ["❓", `질문 걸어 둠: ${d.kind === "confirm" ? "실행 재확인" : "담당자 확인"} ${esc(d.action)}`, json(d.target)],
  pending_answer: (d) => [d.verified ? "✅" : "⚠", `고객 답 판정 <b>${esc(d.answer)}</b> ${d.verified ? "(확정)" : "(미확정)"}`,
    `인용: "${esc(d.quote)}"`, d.verified ? "ok" : "warn"],
  tool_request: (d) => ["🛠", `도구 요청 <b>${esc(d.name)}</b>`, json(d.args)],
  gate: (d) => d.allowed ? null : ["🚫", `게이트 차단 <b>${esc(d.name)}</b>`, esc(d.reason), "err"],
  tool_result: (d) => d.ok ? ["✓", `결과 ${esc(d.name)}`, json(omit(d, ["name", "risk", "ok"]))]
                           : ["✗", `실패 ${esc(d.name)}`, esc(d.error || ""), "err"],
};

function renderTrace(card, ev) {
  const fmt = TRACE[ev.kind];
  const row = fmt && fmt(ev.data);
  if (!row) return;
  const [icon, title, detail = "", cls = ""] = row;
  let list = card.querySelector("ol.trace");
  if (!list) { list = document.createElement("ol"); list.className = "trace"; card.appendChild(list); }
  list.insertAdjacentHTML("beforeend", `<li class="${cls}"><span class="t">+${ev.ms}ms</span>
    <span class="i">${icon}</span><span>${title}${detail ? `<div class="d">${detail}</div>` : ""}</span></li>`);
}

function renderState(st) {
  if (!st) return;
  const yes = (b) => b ? '<span class="ok">✓</span>' : '<span class="muted">-</span>';
  const rows = [
    ["업무", st.active ? `<b>${esc(st.active.task)}</b> ${json(st.active.slots)}` : "-"],
    ["대기열", st.queue.length ? esc(st.queue.join(", ")) : "-"],
    ["본인 인증", `${yes(st.authenticated)}${st.auth_failures ? ` 실패 ${st.auth_failures}회` : ""}`],
    ["담당자 확인", yes(st.contact_verified)],
    ["걸린 질문", st.pending ? `${esc(st.pending.action || st.pending.kind)} ${json(st.pending.target)} (턴 ${st.pending.turn})` : "-"],
    ["동의 받음", st.confirmations.map((c) => `${esc(c.action)} ${json(c.target)}`).join("<br>") || "-"],
    ["실행됨", st.executed.map(esc).join("<br>") || "-"],
  ];
  $("callstate").innerHTML = `<h3>통화 상태 · 턴 ${st.turn}</h3><table>${
    rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("")}</table>`;
}

function json(o) { return o && Object.keys(o).length ? `<code>${esc(JSON.stringify(o))}</code>` : ""; }
function omit(o, keys) { return Object.fromEntries(Object.entries(o).filter(([k]) => !keys.includes(k))); }

function renderTool(card, data, kind) {
  const body = kind === "호출" ? data.args : data.result;
  card.insertAdjacentHTML("beforeend",
    `<details ${kind === "호출" ? "open" : ""}><summary>도구 ${kind}: <b>${esc(data.name)}</b></summary>
     <pre>${esc(JSON.stringify(body, null, 2))}</pre></details>`);
}

function chip(label, ms) {
  const cls = ms > 3000 ? "err" : ms > 1500 ? "warn" : "";
  return `<span class="chip ${cls}">${label} ${Math.round(ms)}ms</span>`;
}

function setComposer(enabled, streaming = false) {
  $("utterance").disabled = !enabled;
  $("send").disabled = !enabled;
  $("stop").disabled = !streaming;
  if (enabled) $("utterance").focus();
}

function setStatus(text, kind = "") {
  $("status").textContent = text;
  $("status").className = `status ${kind ? `chip ${kind}` : "muted"}`;
}

function scrollChat() { $("chat").scrollTop = $("chat").scrollHeight; }

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------------------------------------------------------------- 이벤트

$("start").addEventListener("click", () => startCall().catch((e) => setStatus(e.message, "err")));
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("utterance").value.trim();
  if (!text || !state.sessionId) return;
  $("utterance").value = "";
  sendTurn(text);
});
$("stop").addEventListener("click", () => state.controller?.abort());
loadScenarios().catch((e) => setStatus(`시나리오 로드 실패: ${e.message}`, "err"));
