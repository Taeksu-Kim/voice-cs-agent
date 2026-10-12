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
      } else if (event === "retrieval") renderRetrieval(card, data);
      else if (event === "tool_call") renderTool(card, data, "호출");
      else if (event === "tool_result") renderTool(card, data, "결과");
      else if (event === "done") finishTurn(bubble, card, data, firstDeltaMs);
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

function renderRetrieval(card, data) {
  const items = data.docs.map((d) =>
    `<li>${d.score.toFixed(3)} · ${esc(d.title)}${d.section ? ` <span class="label">› ${esc(d.section)}</span>` : ""}</li>`).join("");
  card.insertAdjacentHTML("beforeend", `<div><span class="label">검색 ${data.ms}ms</span><ul>${items}</ul></div>`);
}

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
