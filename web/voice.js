// 음성 — 실제 테스트 탭(마이크 → WebSocket)과 데모 시나리오·녹화 재생 탭(실제 통화 기록 재생). app.js의 그리기 함수를 같이 쓴다.
// 마이크는 보안 컨텍스트에서만 열린다: Windows 브라우저에서 http://localhost:8080 으로 접속할 것 (WSL IP 주소면 차단).

const voice = { ws: null, mic: null, player: null, bubble: null, card: null, ending: false };
const SR = 16000;

// ---------------------------------------------------------------- 재생기 (에이전트·데모 음성)

class Player {
  constructor(onPlaying) {
    this.ctx = new AudioContext();
    this.sources = new Set();
    this.next = 0;
    this.onPlaying = onPlaying || (() => {});
  }
  async decode(bytes) { return this.ctx.decodeAudioData(bytes); }
  // 앞 소리 뒤에 이어 붙인다. 반환: 재생 시작까지 남은 초
  play(buffer, at = 0) {
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.ctx.destination);
    const start = Math.max(this.ctx.currentTime + at, this.next);
    src.start(start);
    this.next = start + buffer.duration;
    this.sources.add(src);
    if (this.sources.size === 1) this.onPlaying(true);
    src.onended = () => {
      this.sources.delete(src);
      if (!this.sources.size) this.onPlaying(false);
    };
    return { startIn: start - this.ctx.currentTime, endIn: this.next - this.ctx.currentTime };
  }
  stop() {
    for (const s of this.sources) { s.onended = null; try { s.stop(); } catch {} }
    this.sources.clear();
    this.next = 0;
    this.onPlaying(false);
  }
  async close() { this.stop(); await this.ctx.close(); }
}

// ---------------------------------------------------------------- 마이크 → 16kHz PCM16 20ms 프레임

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(320); this.n = 0; }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === 320) { this.port.postMessage(this.buf.slice()); this.n = 0; }
    }
    return true;
  }
}
registerProcessor("tap", Tap);`;

async function openMic(onFrame) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const ctx = new AudioContext({ sampleRate: SR });   // 브라우저가 마이크 샘플레이트에서 16kHz로 바꿔 준다
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" })));
  const node = new AudioWorkletNode(ctx, "tap");
  node.port.onmessage = (e) => {
    const f = e.data, pcm = new Int16Array(f.length);
    let peak = 0;
    for (let i = 0; i < f.length; i++) {
      const v = Math.max(-1, Math.min(1, f[i]));
      pcm[i] = v * 32767;
      peak = Math.max(peak, Math.abs(v));
    }
    onFrame(pcm.buffer, peak);
  };
  ctx.createMediaStreamSource(stream).connect(node);
  return { close: async () => { stream.getTracks().forEach((t) => t.stop()); await ctx.close(); } };
}

// ---------------------------------------------------------------- 실제 테스트: 음성 통화

function prepareVoicePlayer() {   // 클릭 처리 안에서 동기로 불러야 한다 (자동재생 정책)
  voice.player?.close().catch(() => {});
  voice.player = new Player((on) => voice.ws?.readyState === 1 &&
    voice.ws.send(JSON.stringify({ type: "playing", on })));
  voice.player.ctx.resume();
}

async function startVoice(sessionId) {
  if (!window.isSecureContext) throw new Error("마이크는 http://localhost:8080 으로 열어야 합니다 (WSL IP 주소는 차단됨)");
  if (!voice.player) prepareVoicePlayer();
  await voice.player.ctx.resume();
  const q = $("degrade").checked ? "?degrade=tel" : "";
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/sessions/${sessionId}/voice${q}`);
  ws.binaryType = "arraybuffer";
  voice.ws = ws;
  voice.ending = false;
  let pendingAudio = null;
  ws.onmessage = async (e) => {
    if (e.data instanceof ArrayBuffer) {
      const meta = pendingAudio;
      pendingAudio = null;
      const buf = await voice.player.decode(e.data);
      voice.player.play(buf);
      if (meta && voice.card) voice.card.querySelector(".sents")?.insertAdjacentHTML("beforeend",
        `<li>#${meta.seq} 합성 ${meta.gen_ms}ms · ${buf.duration.toFixed(1)}s — ${esc(meta.text)}</li>`);
      return;
    }
    const m = JSON.parse(e.data);
    if (m.type === "audio") pendingAudio = m;
    else onVoiceMessage(m);
  };
  ws.onclose = () => { if (voice.ws === ws) endVoice("연결 종료"); };   // 새 통화로 닫은 옛 연결은 무시
  await new Promise((ok, fail) => { ws.onopen = ok; ws.onerror = () => fail(new Error("WebSocket 연결 실패")); });
  voice.mic = await openMic((pcm, peak) => {
    if (ws.readyState === 1) ws.send(pcm);
    $("level").style.width = `${Math.min(100, peak * 140)}%`;
  });
  $("voicebar").hidden = false;
  $("composer").hidden = true;
  $("hangup").disabled = false;
  setVad("듣는 중");
  addMessage("system", "음성 통화가 연결됐습니다. 말씀하세요 (이어폰 권장 — 스피커 소리가 마이크로 들어가면 끼어들기로 잡힘).");
}

function onVoiceMessage(m) {
  if (m.type === "vad") {
    setVad(m.state === "start" ? "고객 말하는 중" : "발화 끝 → 전사 중", m.state === "start" ? "warn" : "");
  } else if (m.type === "stt") {
    if (voice.bubble && voice.bubble.classList.contains("streaming")) {
      voice.bubble.classList.remove("streaming");
      addMeta(voice.bubble, "끼어들기로 중단 (이 턴은 기록되지 않음 — 고객 말은 다음 발화에 이어 붙임)");
    }
    if (!m.text) { setVad("듣는 중 (전사 결과 없음)"); return; }
    addMessage("customer", m.text);
    voice.bubble = addMessage("agent", "");
    voice.bubble.classList.add("streaming");
    voice.card = addTurnCard(++state.turnNo, m.text);
    voice.card.insertAdjacentHTML("beforeend",
      `<div class="label">STT ${m.ms}ms · 음성 ${m.audio_s}s</div><ol class="sents label"></ol>`);
    setStatus("응답 생성 중…");
  } else if (m.type === "agent") {
    const { event, data } = m;
    if (!voice.card) return;
    if (event === "delta") { voice.bubble.firstChild.textContent += data.text; scrollChat(); }
    else if (event === "task") renderTask(voice.card, data);
    else if (event === "retrieval") renderRetrieval(voice.card, data);
    else if (event === "step") setStatus(`${data.worker} · LLM 생성 중 (스텝 ${data.step})`);
    else if (event === "trace") renderTrace(voice.card, data);
    else if (event === "done") {
      if (data.utterance) voice.card.querySelector("h3 .muted").textContent = data.utterance;   // 끼어들기로 앞 발화가 붙었을 수 있다
      finishTurn(voice.bubble, voice.card, data, null);
      renderState(data.state);
      setComposer(false);
      voice.ending = data.session_ended;
    } else if (event === "error") addMeta(voice.bubble, `오류: ${data.message}`);
  } else if (m.type === "latency") {
    if (voice.card) voice.card.insertAdjacentHTML("beforeend", latencyChips(m));
    if (!voice.card?.querySelector(".chip.ok") && voice.bubble?.classList.contains("streaming")) {
      voice.bubble.classList.remove("streaming");
    }
    setVad("듣는 중");
    if (voice.ending && voice.player) {   // 에이전트가 통화를 끝냈다 — 마지막 말을 다 들려준 뒤 끊는다
      const left = Math.max(0, voice.player.next - voice.player.ctx.currentTime);
      setTimeout(() => endVoice("에이전트가 종료"), left * 1000 + 500);
    }
  } else if (m.type === "stop") {
    voice.player?.stop();
    setVad("끼어들기 — 재생 중단", "warn");
  } else if (m.type === "error") {
    setStatus(m.message, "err");
  }
}

// 발화 끝(실제 말소리가 끝난 시점)부터 첫 소리까지의 지연 예산
function latencyChips(l) {
  const base = l.vad_wait || 0;
  const step = (label, v) => v == null ? "" : chip(label, base + v);
  return `<div class="timings"><span class="label">말 끝난 뒤 →</span>
    ${chip("VAD 대기", base)} ${step("STT", l.stt)} ${step("첫 글자", l.first_delta)}
    ${step("첫 문장", l.first_sentence)} ${step("첫 소리", l.first_audio)} ${step("생성 완료", l.done)}</div>`;
}

async function endVoice(reason, keepPlayer = false) {
  const ws = voice.ws;
  voice.ws = null;
  if (ws && ws.readyState === 1) { ws.send(JSON.stringify({ type: "hangup" })); ws.close(); }
  await voice.mic?.close().catch(() => {});
  if (!keepPlayer) { await voice.player?.close().catch(() => {}); voice.player = null; }
  voice.mic = null;
  $("voicebar").hidden = true;
  $("hangup").disabled = true;
  if (ws) {
    addMessage("system", `통화 종료 (${reason}) — 기록은 녹화 재생 탭에서 재생할 수 있습니다.`);
    setStatus("통화 종료", "ok");
    renderInfo();   // 「다른 발신자·상황」 버튼을 다시 보인다
  }
}

function setVad(text, kind = "") {
  $("vadstate").textContent = text;
  $("vadstate").className = `chip ${kind}`;
}

// ---------------------------------------------------------------- 데모 시나리오: 골라 둔 녹화 목록 (demos/demos.json)

async function loadDemoCatalog() {
  const items = await (await api("/api/demos")).json();
  $("demo-list").innerHTML = items.length ? `<h3>데모 시나리오 <span class="muted">— 실제로 돌린 통화 기록을 그대로 재생 (응답을 새로 만들지 않음)</span></h3>
    <div class="situations">${items.map((d) =>
      `<button type="button" class="sit" data-src="${esc(d.source)}" data-rec="${esc(d.record_id)}" data-desc="${esc(d.desc || d.title)}">
        <b>${esc(d.title)}</b><span class="muted">${esc(d.desc)}</span><span class="muted">${esc(d.source)} · ${esc(d.record_id)}</span></button>`).join("")}</div>`
    : `<div class="muted">골라 둔 녹화가 없습니다. 실제 테스트 탭에서 통화한 뒤, 녹화 재생 탭에서 그 기록을 고르고 「데모 목록에 추가」를 누르세요.</div>`;
}

$("demo-list").addEventListener("click", (e) => {
  const b = e.target.closest("[data-rec]");
  if (!b) return;
  const player = new Player();   // 클릭 안에서 동기로 (자동재생 정책)
  player.ctx.resume();
  setView("chat");
  $("demo-back").hidden = false;
  playDemo(player, b.dataset.src, b.dataset.rec).catch((err) => setStatus(`재생 실패: ${err.message}`, "err"));
});

// ---------------------------------------------------------------- 녹화 재생: 실제 통화 기록 재생

const demo = { rec: null, player: null, timers: [], stopped: false };

async function loadDemoSources() {
  const { sources } = await (await api("/api/monitor/sources")).json();
  $("demo-source").innerHTML = sources.map((s) =>
    `<option value="${s.id}">${s.kind === "eval" ? "평가" : "서버"} · ${s.id}</option>`).join("");
  await loadDemoList();
}

async function loadDemoList() {
  const src = $("demo-source").value;
  const list = await (await api(`/api/recordings?source=${encodeURIComponent(src)}`)).json();
  $("demo-rec").innerHTML = list.slice().reverse().map((r) =>
    `<option value="${esc(r.id)}">${r.kind === "voice" ? "🎙" : "⌨"} ${esc(r.scenario)}${r.situation ? `/${esc(r.situation)}` : ""} · ${r.turns}턴 · ${
      esc(r.disposition || r.end_reason || "-")}${r.passed == null ? "" : r.passed ? " · 통과" : " · 실패"} · ${esc(r.id)}</option>`)
    .join("") || `<option value="">기록 없음</option>`;
}

const sleep = (ms) => new Promise((ok) => demo.timers.push(setTimeout(ok, ms)));

async function tts(text, voiceName) {
  const res = await api("/api/tts", { text, voice: voiceName });
  return demo.player.decode(await res.arrayBuffer());
}

function splitSentences(text) {   // 텍스트 기록용 — 서버의 SentenceSplitter와 같은 기준(문장부호 + 공백)
  return text.split(/(?<=[.?!])\s+/).map((s) => s.trim()).filter(Boolean);
}

// 재생할 소리를 먼저 다 합성한다 — 재생 중 합성 대기가 원본 간격에 섞이지 않게
async function prepareDemo(rec) {
  const turns = [];
  let done = 0;
  const total = (rec.greeting ? 1 : 0)
    + rec.turns.reduce((n, t) => n + 1 + (t.sentences.length || splitSentences(t.agent).length), 0);
  const tick = () => setStatus(`재생 준비 — 음성 합성 ${++done}/${total}`);
  demo.greeting = rec.greeting ? await tts(rec.greeting, "F1") : null;
  if (rec.greeting) tick();
  for (const t of rec.turns) {
    if (demo.stopped) return null;
    const customer = await tts(t.customer || t.utterance, "M1"); tick();
    const sents = t.sentences.length ? t.sentences : splitSentences(t.agent).map((text) => ({ text }));
    const agent = [];
    for (const s of sents) { agent.push({ ...s, buf: await tts(s.text, "F1") }); tick(); }
    turns.push({ t, customer, agent });
  }
  return turns;
}

async function playDemo(player, src, id) {
  stopDemo();
  demo.stopped = false;
  demo.player = player;
  if (!id) return;
  const rec = await (await api(`/api/recordings/${encodeURIComponent(id)}?source=${encodeURIComponent(src)}`)).json();
  demo.rec = rec;
  $("chat").innerHTML = ""; $("turns").innerHTML = ""; $("callstate").innerHTML = "";
  backofficeReset();
  const sc = state.scenarios.find((x) => x.id === rec.scenario);   // 기록엔 CRM 사본이 없어 시나리오·발신자 정의에서
  backofficeCustomer(sc?.customer || { name: rec.scenario }, sc?.today || "");
  const sit = state.callers.find((c) => c.id === rec.scenario)?.situations.find((x) => x.id === rec.situation);
  state.turnNo = 0;
  $("demo-badge").innerHTML = rec.kind === "voice"
    ? `<b>녹화 재생</b> · 실제 음성 통화 기록 ${esc(rec.id)} (${esc(rec.created_at)}) · 응답·도구·지연은 기록 그대로, 고객 목소리는 STT 결과를 다시 합성한 것`
    : `<b>녹화 재생</b> · 텍스트 기록 ${esc(rec.id)} (${rec.mode === "eval" ? `평가 ${esc(src)} — 고객은 LLM 시뮬레이터` : esc(rec.created_at)}) · 응답·도구는 기록 그대로, 음성 지연은 기록 없음(재생 간격 임의)`;
  if (sit) $("demo-badge").insertAdjacentHTML("beforeend", `<div>상황: <b>${esc(sit.title)}</b> — ${esc(sit.desc)}</div>`);
  $("demo-badge").hidden = false;
  $("demo-stop").disabled = false;
  const turns = await prepareDemo(rec);
  if (!turns) return;
  addMessage("system", `통화 재생 시작 — ${rec.scenario}${rec.situation ? ` / ${rec.situation}` : ""}`);
  setStatus("재생 중");
  if (demo.greeting) {
    addMeta(addMessage("agent", rec.greeting), "인바운드 고정 인사 (LLM 아님)");
    await sleep(demo.player.play(demo.greeting).endIn * 1000 + 300);
  }
  for (const { t, customer, agent } of turns) {
    if (demo.stopped) return;
    // 고객 발화
    const bubbleC = addMessage("customer", t.customer || t.utterance);
    const { endIn } = demo.player.play(customer);
    await sleep(endIn * 1000);
    if (demo.stopped) return;
    const card = addTurnCard(++state.turnNo, t.utterance || t.customer);
    if (t.utterance && t.customer && t.utterance !== t.customer) addMeta(bubbleC, "앞 턴이 끼어들기로 끊겨 이어 붙은 발화");
    const bubble = addMessage("agent", "");
    bubble.classList.add("streaming");
    const lat = t.latency || {};
    const t0 = performance.now();
    // 추적 기록을 원래 시각에 맞춰 (음성 기록: 말 끝 → VAD 대기 + STT 뒤에 에이전트 시작)
    const traceBase = rec.kind === "voice" ? (lat.vad_wait || 0) + (lat.stt || 0) : 0;
    for (const ev of t.trace) {
      demo.timers.push(setTimeout(() => !demo.stopped && renderTrace(card, ev), traceBase + ev.ms));
    }
    // 에이전트 문장: 음성 기록은 원래 보낸 시각(at_ms) 이후에, 텍스트 기록은 0.5초 간격으로 이어서
    let lastEnd = 0;
    for (const [i, s] of agent.entries()) {
      const at = rec.kind === "voice" ? (lat.vad_wait || 0) + (s.at_ms ?? 0) : (i === 0 ? 500 : 0);
      await sleep(Math.max(0, t0 + at - performance.now()));
      if (demo.stopped) return;
      const r = demo.player.play(s.buf);
      bubble.firstChild.textContent += (bubble.firstChild.textContent ? " " : "") + s.text;
      scrollChat();
      lastEnd = performance.now() + r.endIn * 1000;
    }
    await sleep(Math.max(0, lastEnd - performance.now()));
    if (demo.stopped) return;
    bubble.classList.remove("streaming");
    if (t.reply) {
      bubble.firstChild.textContent = t.agent;
      card._sources = t.sources;
      finishTurn(bubble, card, { ...t.reply, session_ended: false }, null);
    }
    if (rec.kind === "voice") card.insertAdjacentHTML("beforeend", latencyChips(lat));
    if (t.barge_in || !t.completed) addMeta(bubble, t.completed ? "재생 중 고객이 끼어듦" : "끼어들기로 중단 (기록되지 않은 턴)");
    await sleep(400);
  }
  addMessage("system", `통화 종료 (${rec.disposition || rec.end_reason || "-"}) — 재생 끝`);
  setStatus("재생 끝", "ok");
  $("demo-stop").disabled = true;
}

function stopDemo() {
  demo.stopped = true;
  demo.timers.forEach(clearTimeout);
  demo.timers = [];
  demo.player?.close().catch(() => {});
  demo.player = null;
  $("demo-stop").disabled = true;
}

// ---------------------------------------------------------------- 탭

function showTab(name) {
  for (const el of document.querySelectorAll("[data-tab]")) el.hidden = !el.dataset.tab.split(" ").includes(name);
  for (const b of document.querySelectorAll(".tabs button")) b.classList.toggle("on", b.dataset.show === name);
  stopDemo();
  $("demo-badge").hidden = true;
  $("demo-back").hidden = true;
  if (name !== "live") endVoice("탭 전환");
  // 실제 테스트·데모 시나리오는 고르는 화면부터, 녹화 재생은 바로 통화 화면
  setView(name === "records" ? "chat" : "pick");
  if (name === "demos") loadDemoCatalog().catch((e) => setStatus(e.message, "err"));
  if (name === "records") loadDemoSources().catch((e) => setStatus(e.message, "err"));
}

for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => showTab(b.dataset.show));
$("hangup").addEventListener("click", () => endVoice("고객이 끊음"));
$("demo-source").addEventListener("change", () => loadDemoList().catch((e) => setStatus(e.message, "err")));
$("demo-play").addEventListener("click", () => {
  const player = new Player();   // 클릭 안에서 동기로 (자동재생 정책)
  player.ctx.resume();
  playDemo(player, $("demo-source").value, $("demo-rec").value).catch((e) => setStatus(`재생 실패: ${e.message}`, "err"));
});
$("demo-stop").addEventListener("click", () => { stopDemo(); setStatus("재생 정지", "warn"); });
$("demo-back").addEventListener("click", () => { stopDemo(); $("demo-badge").hidden = true; $("demo-back").hidden = true; setView("pick"); });
$("demo-add").addEventListener("click", async () => {
  const src = $("demo-source").value, id = $("demo-rec").value;
  if (!id) return;
  const title = prompt("데모 제목 (예: 카드 분실 — 동의 받고 분실신고)");
  if (!title) return;
  const desc = prompt("설명 (목록 카드에 보임, 비워도 됨)") || "";
  try {
    const items = await (await api("/api/demos", { title, desc, source: src, record_id: id })).json();
    setStatus(`데모 목록에 추가됨 (${items.length}건)`, "ok");
  } catch (e) { setStatus(`추가 실패: ${e.message}`, "err"); }
});
