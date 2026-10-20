// 업무 화면 — 에이전트의 도구 호출을 상담원 백오피스 화면처럼 그린다.
// 입력은 추적 기록(trace) 이벤트뿐: tool_request(입력값) → precheck·slot_rule·gate(보류·차단) → tool_result(결과).
// 텍스트(SSE)·음성(WebSocket)·데모(기록 재생)가 모두 renderTrace를 거치므로 세 경로에서 같게 보인다.
// 화면에 보이는 값은 전부 도구 인자·결과에 실제로 있던 값이다 (문자 본문 같은 건 만들지 않는다).

const bo = { open: {}, waiting: {}, cards: {} };

// card_id 옆에 직전 카드 조회 결과의 이름을 붙인다 (에이전트가 넣은 값은 card_id 그대로 보여 준다)
const cardLabel = (id) => bo.cards[id] ? `${id}  (${bo.cards[id].name} •••• ${bo.cards[id].last4})` : id;

const STATUS_CLS = { 정상: "ok", 배송중: "warn", 배송완료: "ok", 일시정지: "warn", 분실신고: "err" };
const won = (n) => (typeof n === "number" ? n.toLocaleString("ko-KR") + "원" : esc(n ?? ""));
const pill = (v) => `<span class="pill ${STATUS_CLS[v] || ""}">${esc(v ?? "-")}</span>`;
const table = (cols, rows) => rows.length
  ? `<table class="grid"><tr>${cols.map(([, h]) => `<th>${h}</th>`).join("")}</tr>${rows.map((r) =>
      `<tr>${cols.map(([k, , f]) => `<td>${f ? f(r[k], r) : esc(r[k] ?? "")}</td>`).join("")}</tr>`).join("")}</table>`
  : `<div class="muted">조회 결과 없음</div>`;
const fields = (pairs) => `<dl class="form">${pairs.map(([k, v]) =>
  `<dt>${k}</dt><dd><span class="field">${v === "" || v == null ? "<span class='muted'>(비어 있음)</span>" : esc(v)}</span></dd>`).join("")}</dl>`;

// 도구별 화면: 메뉴 경로 · 입력 칸 · 결과 그리기
const SCREENS = {
  lookup_orders: { menu: "토스쇼핑 › 주문·배송 조회",
    input: (a) => [["조회 대상", "통화 중인 고객 본인"], ["주문번호", a.order_id || "(전체)"]],
    result: (r) => (r.not_found ? `<div class="big err">등록되지 않은 주문번호 · ${esc(r.not_found)}</div>` : "") + table([["order_id", "주문번호"], ["item", "상품"], ["status", "상태", pill],
      ["shipped_at", "발송일"], ["last_tracking", "최근 배송 기록"], ["eta", "도착 예정"]], r.orders || []) +
      (r.today ? `<div class="foot">기준일 ${esc(r.today)}</div>` : "") },
  lookup_transfers: { menu: "송금 › 최근 송금 내역", input: () => [["조회 대상", "통화 중인 고객 본인"]],
    result: (r) => table([["transfer_id", "번호"], ["date", "일시"], ["to_name", "받는 분"], ["bank", "은행"],
      ["amount", "금액", won]], r.transfers || []) },
  lookup_cards: { menu: "카드 › 보유 카드", input: () => [["조회 대상", "통화 중인 고객 본인"]],
    result: (r) => table([["card_id", "ID"], ["name", "카드"], ["last4", "끝 4자리", (v) => `•••• ${esc(v)}`],
      ["status", "상태", pill]], r.cards || []) },
  lookup_seller: { menu: "셀러 › 상점 정보", input: () => [["조회 대상", "통화 상대 상점"]],
    result: (r) => {
      const s = r.seller || {}, pts = s.penalty_points_30d;
      const bar = typeof pts === "number" ? `<div class="gauge"><div style="width:${Math.min(100, pts * 10)}%"></div>
        <span>${pts} / 10점 (10점이면 이용 정지)</span></div>` : "";
      return fields([["상점", s.store_name], ["담당자", s.manager_name]]) +
        (bar ? `<div class="sub">최근 30일 페널티</div>${bar}` : "") +
        (s.items ? table([["type", "위반 항목"], ["points", "점수"]], s.items) : "");
    } },
  request_app_auth: { menu: "본인 확인 › 앱 인증 요청", input: () => [["요청 방식", "고객 휴대폰 토스 앱 푸시"]],
    result: (r) => r.status === "success" ? `<div class="big ok">✓ 인증 완료</div>`
      : `<div class="big err">✗ 인증 실패${r.failures ? ` (누적 ${r.failures}회)` : ""}</div>` },
  ask_contact: { menu: "아웃바운드 › 담당자 확인", input: () => [["확인 내용", "통화 상대가 담당자 본인인지"]],
    result: (r) => r.status === "pending" ? `<div class="big warn">고객 답변 대기 — 다음 발화로 판정</div>`
      : `<div class="big ok">이미 확인됨</div>` },
  suspend_card: { menu: "카드 › 일시정지", input: (a) => [["대상 카드", cardLabel(a.card_id)]],
    result: (r) => `<div class="big warn">일시정지 처리됨 · ${esc(r.card_id)}</div>` },
  report_card_lost: { menu: "카드 › 분실신고", input: (a) => [["대상 카드", cardLabel(a.card_id)]],
    result: (r) => `<div class="big err">분실신고 접수됨 · ${esc(r.card_id)}</div>` },
  // target은 안내 항목 id — 이름·링크는 도메인 API가 지식 베이스에서 찾은 값 (domain/guides.yaml)
  send_app_link: { menu: "안내 › 안내 링크 문자", input: (a) => [["안내 항목", a.target], ["받는 번호", "고객 등록 휴대폰"]],
    result: (r) => `<div class="big ok">✉ 발송 완료 · ${esc(r.title || r.target)}</div>` +
      (r.url ? `<div class="foot">보낸 링크 <a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.url)}</a></div>` : "") },
  schedule_callback: { menu: "상담 › 재통화 예약", input: (a) => [["희망 시각", a.when]],
    result: (r) => `<div class="big ok">📅 예약됨 · ${esc(r.when)}</div>` },
  record_lead: { menu: "아웃바운드 › 리드 기록",
    input: (a) => [["니즈", a.need], ["예산", a.budget], ["관심도", a.interest], ["등급", a.grade]],
    result: () => `<div class="big ok">기록 저장됨</div>` },
  transfer_to_agent: { menu: "상담 › 상담원 이관", input: (a) => [["이관 사유", a.reason], ["통화 요약", a.summary]],
    result: () => `<div class="big ok">상담원 대기열에 이관됨</div>` },
  end_call: { menu: "상담 › 통화 종료", input: (a) => [["통화 결과", a.disposition]],
    result: (r) => `<div class="big">${esc(r.disposition)}</div>` },
};

const GATE_TEXT = {
  confirmation_required: ["동의 대기", "warn", "실행 전 고객 동의를 묻는 중 — 아직 실행하지 않음"],
  auth_required: ["차단", "err", "본인 인증 전이라 실행할 수 없음"],
  contact_required: ["차단", "err", "담당자 확인 전이라 실행할 수 없음"],
  not_allowed: ["차단", "err", "지금 업무 단계에서 쓸 수 없는 기능"],
  duplicate: ["중복", "warn", "이미 같은 대상으로 실행됨 — 다시 실행하지 않음"],
  tasks_queued: ["보류", "warn", "남은 용건이 있어 종료를 미룸"],
  confirmed_not_executed: ["차단", "err", "동의받은 처리를 아직 실행하지 않음 — 먼저 실행해야 함"],
};

function boWindow(name, args, ms) {
  const sc = SCREENS[name] || { menu: name, input: (a) => Object.entries(a || {}), result: (r) => `<pre>${esc(JSON.stringify(r, null, 1))}</pre>` };
  const el = document.createElement("section");
  el.className = "bo-win";
  el.innerHTML = `<header><span class="menu">${esc(sc.menu)}</span>
      <span class="stamp" data-s="run">요청</span></header>
    <div class="meta">턴 ${state.turnNo} · +${ms}ms · <code>${esc(name)}</code></div>
    <div class="sub">입력</div>${fields(sc.input(args || {}))}
    <div class="note"></div><div class="res"></div>`;
  el._sc = sc;
  $("bo-list").prepend(el);
  $("bo-empty").hidden = true;
  return el;
}

function stamp(el, text, cls, note) {
  const s = el.querySelector(".stamp");
  s.textContent = text;
  s.dataset.s = cls;
  if (note) el.querySelector(".note").innerHTML = `<div class="callout ${cls}">${esc(note)}</div>`;
}

// renderTrace(app.js)가 이벤트마다 부른다
function backofficeTrace(ev) {
  const d = ev.data || {};
  if (ev.kind === "tool_request") {
    bo.open[d.name] = boWindow(d.name, d.args, ev.ms);
  } else if (ev.kind === "precheck" && bo.open[d.name]) {
    stamp(bo.open[d.name], "입력 오류", "err", d.error);
  } else if (ev.kind === "slot_rule" && bo.open[d.name]) {
    stamp(bo.open[d.name], "차단", "err", `업무 규칙: ${d.slot} = ${d.value}`);
  } else if (ev.kind === "gate" && !d.allowed && bo.open[d.name]) {
    const [t, c, note] = GATE_TEXT[d.reason] || ["차단", "err", `게이트: ${d.reason}`];
    stamp(bo.open[d.name], t, c, note);
    if (d.reason === "confirmation_required") bo.waiting[d.name] = bo.open[d.name];
  } else if (ev.kind === "tool_result" && bo.open[d.name]) {
    const el = bo.open[d.name];
    delete bo.open[d.name];
    const { name, risk, ok, error, ...result } = d;
    if (ok === false || error) {
      stamp(el, "실패", "err", error || "처리 실패");
    } else {
      stamp(el, "완료", "ok");
      if (name === "lookup_cards") for (const c of result.cards || []) bo.cards[c.card_id] = c;
      if (name === "request_app_auth" && result.status === "success") backofficeState({ authenticated: true });
      if (risk === "execute" || risk === "notify") refreshCrm();   // 정보 패널에 실행 결과 반영 (app.js)
      const waited = bo.waiting[name];
      if (waited) {   // 앞 턴에서 동의를 기다리던 요청 — 동의를 받아 이번에 실행됐다
        stamp(waited, "동의 후 실행", "done");
        waited.querySelector(".note").innerHTML = `<div class="callout done">고객 동의 → 턴 ${state.turnNo}에서 실행</div>`;
        delete bo.waiting[name];
      }
      el.querySelector(".res").innerHTML = `<div class="sub">결과</div>${el._sc.result(result)}`;
    }
  }
}

// 고객 정보 줄 — 통화 시작 시 CRM, 턴마다 인증 상태
function backofficeCustomer(customer, today) {
  $("bo-customer").innerHTML = `<b>${esc(customer?.name || "고객")}</b>
    ${customer?.seller ? `· 상점 ${esc(customer.seller.store_name || "")}` : ""} · 기준일 ${esc(today || "-")}
    <span id="bo-auth" class="pill">인증 전</span>`;
}

function backofficeState(st) {
  const el = document.getElementById("bo-auth");
  if (!el || !st) return;
  const [t, c] = st.authenticated ? ["본인 인증됨", "ok"] : st.contact_verified ? ["담당자 확인됨", "ok"]
    : st.auth_failures ? [`인증 실패 ${st.auth_failures}회`, "err"] : ["인증 전", ""];
  el.textContent = t;
  el.className = `pill ${c}`;
}

function backofficeReset() {
  bo.open = {}; bo.waiting = {}; bo.cards = {};
  $("bo-list").innerHTML = "";
  $("bo-empty").hidden = false;
  $("bo-customer").innerHTML = "";
}
