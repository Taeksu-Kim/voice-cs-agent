// 에이전트 모니터 — 로그 폴더(평가 실행·서버)의 지표를 몇 초마다 다시 읽어 그린다 (/api/monitor).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const num = (n) => (n ?? 0).toLocaleString("ko-KR");
const ms = (n) => `${num(n)}ms`;
const time = (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString("ko-KR", { hour12: false }) : "-");
const pills = (obj) => Object.entries(obj || {}).sort((a, b) => b[1] - a[1])
  .map(([k, v]) => `<span class="pill">${esc(k)} ${v}</span>`).join("") || `<span class="muted">없음</span>`;
const sum = (obj) => Object.values(obj || {}).reduce((a, b) => a + b, 0);

let timer = null;

async function getJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

async function loadSources() {
  const { server, sources } = await getJson("/api/monitor/sources");
  const want = new URLSearchParams(location.search).get("source");
  $("source").innerHTML = sources.map((s) =>
    `<option value="${esc(s.id)}">${s.kind === "eval" ? "평가" : "서버"} · ${esc(s.id)}${s.id === server ? " (이 서버)" : ""}${s.has_events ? "" : " — 이벤트 없음"}</option>`).join("");
  if (want && sources.some((s) => s.id === want)) $("source").value = want;
}

async function refresh() {
  const source = $("source").value;
  if (!source) return;
  try {
    render(await getJson(`/api/monitor?source=${encodeURIComponent(source)}`));
  } catch (err) {
    $("updated").textContent = `불러오지 못했습니다: ${err.message}`;
  }
}

function tile(label, value, sub = "", cls = "") {
  return `<div class="tile ${cls}"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
}

function render(m) {
  const span = m.first_ts ? ((m.last_ts - m.first_ts) / 60).toFixed(1) : "0";
  $("updated").textContent = `갱신 ${time(m.updated)} · 경과 ${span}분 · 마지막 기록 ${time(m.last_ts)}`;

  const t = m.turns, i = m.incidents, last = m.llm.last_60s, ev = m.eval;
  const llmTotal = Object.values(m.llm.by_source).reduce(
    (a, s) => ({ p: a.p + s.prompt_tokens, c: a.c + s.completion_tokens, n: a.n + s.calls, e: a.e + s.errors }),
    { p: 0, c: 0, n: 0, e: 0 });
  const fallbacks = sum(i.fallback);
  $("tiles").innerHTML = [
    ev ? tile("평가 통과", `${ev.passed}/${ev.done}`, `${Math.round((ev.passed / ev.done) * 100)}% · 시뮬레이터 이탈 ${ev.sim_drift}`) : "",
    tile("통화", `${m.calls.ended}/${m.calls.started}`, "종료/시작"),
    tile("턴 지연 p50", ms(t.ms_total.p50), `p90 ${ms(t.ms_total.p90)} · max ${ms(t.ms_total.max)}`),
    tile("첫 글자 p50", ms(t.ms_first_token.p50), `p90 ${ms(t.ms_first_token.p90)}`),
    tile("최근 60초 토큰", num(last.prompt_tokens + last.completion_tokens),
      `입력 ${num(last.prompt_tokens)} · 출력 ${num(last.completion_tokens)} · ${last.calls}회`),
    tile("누적 LLM 토큰", num(llmTotal.p + llmTotal.c), `${num(llmTotal.n)}회 호출`),
    tile("폴백", fallbacks, pills(i.fallback), fallbacks ? "warn" : ""),
    tile(`150자 초과`, t.over_chars, `${t.n}턴 중`, t.over_chars ? "warn" : ""),
    tile("오류", llmTotal.e + i.turn_error, `LLM ${llmTotal.e} · 턴 ${i.turn_error}`, llmTotal.e + i.turn_error ? "err" : ""),
  ].join("");

  $("eval").hidden = !ev;
  if (ev) {
    $("eval").querySelector("div").innerHTML = `<table><tr><th>시나리오</th><th class="num">통과</th><th>실패 항목</th><th class="num">이탈</th></tr>` +
      Object.entries(ev.scenarios).map(([sid, s]) => `<tr><td>${esc(sid)}</td>
        <td class="num ${s.passed === s.n ? "pass" : "fail"}">${s.passed}/${s.n}</td>
        <td class="fails">${pills(s.failed)}</td><td class="num">${s.sim_drift || ""}</td></tr>`).join("") + "</table>";
  }

  $("llm").innerHTML = `<table><tr><th>출처</th><th class="num">호출</th><th class="num">입력 토큰</th><th class="num">출력 토큰</th>
    <th class="num">p50</th><th class="num">p90</th><th class="num">오류</th></tr>` +
    Object.entries(m.llm.by_source).map(([src, s]) => `<tr><td>${esc(src)}</td><td class="num">${num(s.calls)}</td>
      <td class="num">${num(s.prompt_tokens)}</td><td class="num">${num(s.completion_tokens)}</td>
      <td class="num">${ms(s.ms.p50)}</td><td class="num">${ms(s.ms.p90)}</td>
      <td class="num ${s.errors ? "fail" : ""}">${s.errors}</td></tr>`).join("") + "</table>";

  $("incidents").innerHTML = `<table>
    <tr><td>통화 결과</td><td>${pills(m.calls.dispositions)}</td></tr>
    <tr><td>폴백 (사유별)</td><td>${pills(i.fallback)}</td></tr>
    <tr><td>절차 강제 · 스텝 소진 · 라우팅 실패 · 되묻기</td><td>${i.step_enforced} · ${i.max_steps} · ${i.route_failed} · ${i.ask_again}</td></tr>
    <tr><td>업무 전환</td><td>${pills(i.task_switch)}</td></tr>
    <tr><td>도구 결말</td><td>${pills(m.tools.status)}</td></tr>
    <tr><td>도구 차단 (도구:사유)</td><td>${pills(m.tools.blocked)}</td></tr>
    <tr><td>도구 실패 (도구:사유)</td><td>${pills(m.tools.failed)}</td></tr>
    <tr><td>도구별 호출</td><td>${pills(m.tools.by_tool)}</td></tr>
    <tr><td>턴당 LLM 스텝</td><td>p50 ${t.llm_steps.p50} · p90 ${t.llm_steps.p90} · max ${t.llm_steps.max}</td></tr></table>`;

  $("warnings").innerHTML = m.warnings.length ? `<table><tr><th>시각</th><th>수준</th><th>이벤트</th><th>위치</th><th>내용</th></tr>` +
    m.warnings.slice().reverse().map((w) => `<tr><td>${time(w.ts)}</td><td class="lvl-${esc(w.level)}">${esc(w.level)}</td>
      <td>${esc(w.event)}</td><td>${esc([w.scenario, w.call_id, w.turn && `${w.turn}턴`].filter(Boolean).join(" · "))}</td>
      <td class="fails">${esc(String(w.reason || w.error || w.output || "").slice(0, 120))}
      ${w.llm_call_id ? `<button type="button" class="raw" data-llm-call="${esc(w.llm_call_id)}">원문</button>` : ""}</td></tr>`).join("") + "</table>"
    : `<span class="muted">없음</span>`;
}

// 경고에 딸린 LLM 호출 원문 보기
document.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-llm-call]");
  if (!btn) return;
  $("llm-dialog-body").textContent = "불러오는 중…";
  $("llm-dialog").showModal();
  try {
    const rec = await getJson(`/api/llm_calls/${encodeURIComponent(btn.dataset.llmCall)}?source=${encodeURIComponent($("source").value)}`);
    $("llm-dialog-title").textContent = `${rec.source} · ${rec.call_id} · ${rec.ms ?? "-"}ms`;
    $("llm-dialog-body").innerHTML = rec.messages.map((m) =>
      `<section><h4>${esc(m.role)}</h4><pre>${esc(m.content)}</pre></section>`).join("") +
      `<section class="output"><h4>출력</h4><pre>${esc(rec.output)}</pre></section>`;
  } catch (err) {
    $("llm-dialog-body").textContent = `불러오지 못했습니다: ${err.message}`;
  }
});

function schedule() {
  clearInterval(timer);
  if ($("live").checked) timer = setInterval(refresh, 3000);
}

$("source").addEventListener("change", () => {
  history.replaceState(null, "", `?source=${encodeURIComponent($("source").value)}`);
  refresh();
});
$("live").addEventListener("change", schedule);

loadSources().then(refresh).then(schedule);
