// 상담 에이전트 콘솔 — 세션을 열고, 발화마다 SSE 스트림(POST)을 읽어 화면에 그린다.
const $ = (id) => document.getElementById(id);
const state = { sessionId: null, scenarios: [], callers: [], caller: null, situation: null, controller: null, turnNo: 0 };

async function api(path, body) {
  const res = await fetch(path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : undefined);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res;
}

// ---------------------------------------------------------------- 시작: 발신자 → 상황 카드 → 통화

async function loadCallers() {
  state.scenarios = await (await api("/api/scenarios")).json();   // 녹화 재생에서 고객 정보 찾기용
  state.callers = await (await api("/api/callers")).json();
  pickCaller(state.callers[0]?.id);
}

function pickCaller(id) {
  if (voice.ws || state.controller) return;   // 통화·응답 진행 중엔 바꾸지 않는다
  state.caller = state.callers.find((c) => c.id === id) || null;
  state.situation = null;
  endTextCall();
  $("callers").innerHTML = state.callers.map((c) =>
    `<button type="button" class="caller${c === state.caller ? " on" : ""}" data-caller="${esc(c.id)}">
      <b>${esc(c.name)}</b>${c.direction === "outbound" ? ' <span class="pill warn">아웃바운드 — 우리가 거는 전화</span>' : ""}
      <span class="muted">${esc(c.profile)}</span></button>`).join("");
  renderSituations();
  renderInfo();
}

function pickSituation(id) {
  if (voice.ws || state.controller) return;
  state.situation = state.caller?.situations.find((x) => x.id === id) || null;
  endTextCall();
  renderSituations();
  renderInfo();
}

function renderSituations() {
  const c = state.caller;
  if (!c) { $("situations").innerHTML = ""; return; }
  const card = (x) => `<button type="button" class="sit${x === state.situation ? " on" : ""}" data-sit="${esc(x.id)}"
      data-desc="${esc(x.desc)}"><b>${esc(x.title)}</b><span class="muted">${x.related.map((r) => esc(refLabel(c.customer, r))).join(" · ")}</span></button>`;
  $("situations").innerHTML = c.situations.map(card).join("") +
    `<button type="button" class="sit free${state.situation ? "" : " on"}" data-sit="" data-desc="상황 없이 이 고객의 데이터로 무엇이든 물어본다">
      <b>자유 통화</b><span class="muted">상황 없이</span></button>`;
}

// 상황의 related 참조 (order:A3002 · card:C1 · transfer:T1 · seller) → CRM 항목
function refItem(c, ref) {
  const [kind, key] = ref.split(":");
  if (kind === "seller") return c.seller;
  const [list, idKey] = { order: ["orders", "order_id"], card: ["cards", "card_id"], transfer: ["transfers", "transfer_id"] }[kind] || [];
  return (c[list] || []).find((x) => String(x[idKey]) === key);
}

function refLabel(c, ref) {
  const x = refItem(c, ref) || {}, kind = ref.split(":")[0];
  return { order: () => `${x.order_id} ${x.item}`, card: () => `${x.name} ${x.last4}`,
           transfer: () => `${x.to_name} ${won(x.amount)}`, seller: () => x.store_name }[kind]?.() || ref;
}

// 테스터용 정보 패널 — 말할 때 필요한 값을 고정 위치에. 전부 이 고객 CRM에 등록된 값 (에이전트가 조회할 수 있는 값)
function renderInfo() {
  const c = state.caller, sit = state.situation;
  if (!c) { $("info").innerHTML = `<div class="muted">발신자를 고르세요.</div>`; return; }
  const cu = c.customer;
  const head = `<div class="who"><b>${esc(c.name)}</b>${c.direction === "outbound"
    ? ' <span class="pill warn">아웃바운드 — 전화를 받는 쪽, 먼저 「여보세요」</span>' : ""} <span class="muted">오늘 ${esc(c.today)}</span>
    ${voice.ws || $("main").dataset.view === "pick" ? "" : '<button type="button" class="link" id="repick">← 다른 발신자·상황</button>'}</div>`;
  const refs = sit ? sit.related : [
    ...(cu.orders || []).map((o) => `order:${o.order_id}`), ...(cu.cards || []).map((k) => `card:${k.card_id}`),
    ...(cu.transfers || []).map((t) => `transfer:${t.transfer_id}`), ...(cu.seller ? ["seller"] : [])];
  const rows = refs.map((r) => infoRow(cu, r)).join("");
  $("info").innerHTML = head +
    (sit ? `<div class="sit-now"><b>${esc(sit.title)}</b> — ${esc(sit.desc)}</div>` : `<div class="sit-now muted">자유 통화 — 이 고객의 등록 정보 전체</div>`) +
    (rows ? `<table class="grid info">${rows}</table>` : `<div class="muted">등록 정보 없음</div>`);
  $("repick")?.addEventListener("click", () => setView("pick"));
}

function infoRow(cu, ref) {
  const x = refItem(cu, ref) || {}, kind = ref.split(":")[0];
  const row = (label, cells) => `<tr><th>${label}</th><td>${cells.filter(Boolean).join(" · ")}</td></tr>`;
  if (kind === "order") return row("주문", [`<b>${esc(x.order_id)}</b>`, esc(x.item), esc(x.status),
    x.shipped_at && `발송 ${esc(x.shipped_at)}`, `최근 기록 ${esc(x.last_tracking || "없음")}`, x.eta && `도착 예정 ${esc(x.eta)}`]);
  if (kind === "card") return row("카드", [`<b>${esc(x.name)}</b>`, `끝 4자리 <b>${esc(x.last4)}</b>`]);
  if (kind === "transfer") return row("송금", [esc(x.date), `받는 분 ${esc(x.to_name)}`, esc(x.bank), `<b>${won(x.amount)}</b>`]);
  if (kind === "seller") return row("상점", [`<b>${esc(x.store_name)}</b>`, `담당자 ${esc(x.manager_name)}`,
    x.penalty_points_30d != null && `최근 30일 페널티 ${x.penalty_points_30d}점`,
    ...(x.items || []).map((i) => `${esc(i.type)} ${i.points}점`)]);
  return "";
}

function setView(view) {   // pick: 발신자·상황 고르기 / chat: 통화 화면
  $("main").dataset.view = view;
  if (document.querySelector(".tabs button.on")?.dataset.show === "live") renderInfo();
}

function endTextCall() {   // 고르기를 바꾸면 진행 중이던 텍스트 통화는 끝난 것으로
  state.sessionId = null;
  setComposer(false);
}

async function startCall() {
  // 소리 재생기는 클릭 직후 await 전에 만든다 — 크롬 자동재생 정책상 await 뒤에 만든 AudioContext는 정지 상태로 시작한다
  if ($("input-mode").value === "voice") prepareVoicePlayer();
  const c = state.caller, sit = state.situation;
  if (!c) throw new Error("발신자를 고르세요");
  const mode = $("input-mode").value;   // text | voice
  await endVoice("새 통화", true);
  const data = await (await api("/api/sessions", { scenario_id: c.id, mode, situation: sit?.id || "" })).json();
  state.sessionId = data.session_id;
  state.turnNo = 0;
  $("chat").innerHTML = "";
  $("turns").innerHTML = "";
  $("callstate").innerHTML = "";
  backofficeReset();
  backofficeCustomer(data.customer, data.today);
  setView("chat");
  setStatus("통화 중");
  const outbound = c.direction === "outbound";
  if (outbound) addMessage("system", "아웃바운드 — 에이전트가 건 전화를 받는 쪽입니다. 먼저 「여보세요」라고 말하세요.");
  if (data.greeting) addMeta(addMessage("agent", data.greeting), "인바운드 고정 인사 (LLM 아님)");   // 음성이면 서버가 접속하자마자 읽어 줌
  if (mode === "voice") {
    $("composer").hidden = true;
    await startVoice(data.session_id);
    renderInfo();
    return;
  }
  $("composer").hidden = false;
  setComposer(true);
  $("utterance").value = outbound ? "여보세요" : "";
  $("utterance").focus();
  addMessage("system", "통화가 연결됐습니다.");
  renderInfo();
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
  addSources(bubble, card._sources);
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

// 출처 — 이번 턴에 검색돼 LLM 입력에 들어간 문서 (읽어 주지 않음, 화면에만).
// 답이 실제로 어느 문서를 썼는지는 모델이 표시하지 않으므로 「참고 문서」로 적는다
function addSources(bubble, docs) {
  if (!docs || !docs.length || bubble.querySelector(".sources")) return;
  const seen = new Set();
  const items = docs.filter((d) => d.url && !seen.has(d.url) && seen.add(d.url)).map((d) =>
    `<li><a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(d.title)}</a>${
      d.score != null ? ` <span class="muted">${Number(d.score).toFixed(3)}</span>` : ""}</li>`).join("");
  if (items) bubble.insertAdjacentHTML("beforeend",
    `<details class="sources"${seen.size <= 2 ? " open" : ""}><summary>LLM에 넣은 검색 문서 ${seen.size} (답이 쓴 근거인지는 미표시)</summary><ol>${items}</ol></details>`);
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
  card._sources = (card._sources || []).concat(data.docs);   // 턴이 끝나면 채팅창에 「출처」로
  const items = data.docs.map((d) =>
    `<li>${d.score.toFixed(3)} · ${esc(d.title)}${d.section ? ` <span class="label">› ${esc(d.section)}</span>` : ""}</li>`).join("");
  card.insertAdjacentHTML("beforeend", `<div><span class="label">검색 ${data.ms}ms</span><ul>${items}</ul></div>`);
}

// 추적 기록 한 줄 — 에이전트가 이번 턴에 무엇을 했는지 시간순으로
const TRACE = {
  route: (d) => ["🧭", `라우팅 → <b>${esc((d.tasks || [d.task]).join(" → "))}</b>${rawButton(d.call_id)}`, esc(d.reason)],
  task_start: (d) => ["▶", `업무 시작 <b>${esc(d.task)}</b>`],
  task_done: (d) => ["✔", `업무 완료 <b>${esc(d.task)}</b>`],
  precheck: (d) => ["↩", `인자 오류 <b>${esc(d.name)}</b>`, esc(d.error), "warn"],
  task_queued: (d) => ["⏳", `대기열 추가 <b>${esc(d.task)}</b>`],
  task_switch: (d) => ["⇄", `대기 용건으로 전환 <b>${esc(d.from)} → ${esc(d.to)}</b>`],
  slot_rule: (d) => ["🚫", `슬롯 규칙 차단 <b>${esc(d.name)}</b>`, `${esc(d.slot)} = ${esc(String(d.value))}`, "err"],
  retrieval: (d) => ["🔎", `검색 (${esc(d.worker)}) ${d.chunk_ids.length}건`],
  llm: (d) => ["🤖", `LLM ${esc(d.role)} ${d.ms}ms${rawButton(d.call_id)}`, `토큰 ${d.prompt_tokens}+${d.completion_tokens}`],
  intent: (d) => ["↪", `업무 전환 요청 (${esc(d.worker)})`],
  step_enforced: (d) => ["⛔", `절차 강제: 먼저 <b>${esc(d.tool)}</b>`, "", "warn"],
  pending: (d) => ["❓", `질문 걸어 둠: ${d.kind === "confirm" ? `실행 동의 ${esc(d.action)}` : "담당자 확인"}`, json(d.target)],
  pending_answer: (d) => [d.verified ? "✅" : "⚠", `고객 답 판정 <b>${esc(d.answer)}</b> ${d.verified ? "(확정)" : "(미확정)"}`,
    `인용: "${esc(d.quote)}"`, d.verified ? "ok" : "warn"],
  tool_request: (d) => ["🛠", `도구 요청 <b>${esc(d.name)}</b>`, json(d.args)],
  gate: (d) => d.allowed ? null : d.reason === "confirmation_required"
    ? ["🔒", `동의 전이라 실행 보류 <b>${esc(d.name)}</b>`, "확인 질문을 건다", "warn"]
    : ["🚫", `게이트 차단 <b>${esc(d.name)}</b>`, esc(d.reason), "err"],
  tool_result: (d) => d.ok ? ["✓", `결과 ${esc(d.name)}`, json(omit(d, ["name", "risk", "ok"]))]
                           : ["✗", `실패 ${esc(d.name)}`, esc(d.error || ""), "err"],
};

// LLM 호출 원문 보기 — trace의 call_id로 서버 로그(llm_calls.jsonl)에서 꺼낸다
const rawButton = (id) => id ? ` <button type="button" class="raw" data-llm-call="${esc(id)}">원문</button>` : "";

document.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-llm-call]");
  if (!btn) return;
  const dialog = $("llm-dialog");
  $("llm-dialog-body").textContent = "불러오는 중…";
  dialog.showModal();
  try {
    const rec = await (await api(`/api/llm_calls/${encodeURIComponent(btn.dataset.llmCall)}`)).json();
    $("llm-dialog-title").textContent =
      `${rec.source} · ${rec.call_id} · ${rec.ms ?? "-"}ms · 토큰 ${rec.prompt_tokens ?? "-"}+${rec.completion_tokens ?? "-"}`;
    $("llm-dialog-body").innerHTML = rec.messages.map((m) =>
      `<section><h4>${esc(m.role)}</h4><pre>${esc(m.content)}</pre></section>`).join("") +
      `<section class="output"><h4>출력${rec.error ? ` (${esc(rec.error)})` : ""}</h4><pre>${esc(rec.output)}</pre></section>`;
  } catch (err) {
    $("llm-dialog-body").textContent = `불러오지 못했습니다: ${err.message}`;
  }
});

function renderTrace(card, ev) {
  backofficeTrace(ev);   // 업무 화면 (backoffice.js)
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
  backofficeState(st);
  const yes = (b) => b ? '<span class="ok">✓</span>' : '<span class="muted">-</span>';
  const rows = [
    ["업무", st.active ? `<b>${esc(st.active.task)}</b> ${json(st.active.slots)}` : "-"],
    ["대기열", st.queue.length ? esc(st.queue.join(", ")) : "-"],
    ["본인 인증", `${yes(st.authenticated)}${st.auth_failures ? ` 실패 ${st.auth_failures}회` : ""}`],
    ["담당자 확인", yes(st.contact_verified)],
    ["걸린 질문", st.pending ? (st.pending.items.length
        ? st.pending.items.map((x) => `${esc(x.action)} ${json(x.target)}`).join("<br>") : "담당자 확인") + ` (턴 ${st.pending.turn})` : "-"],
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

$("callers").addEventListener("click", (e) => {
  const b = e.target.closest("[data-caller]");
  if (b) pickCaller(b.dataset.caller);
});
$("situations").addEventListener("click", (e) => {
  const b = e.target.closest("[data-sit]");
  if (b) pickSituation(b.dataset.sit);
});
$("start").addEventListener("click", () => startCall().catch((e) => setStatus(e.message, "err")));
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("utterance").value.trim();
  if (!text || !state.sessionId) return;
  $("utterance").value = "";
  sendTurn(text);
});
$("stop").addEventListener("click", () => state.controller?.abort());
loadCallers().catch((e) => setStatus(`발신자 목록 로드 실패: ${e.message}`, "err"));
