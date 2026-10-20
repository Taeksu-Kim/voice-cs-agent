"""실시간 음성 통화 한 건 — WebSocket 하나 위에서 VAD → STT → 에이전트 → 문장 단위 TTS를 돌린다.

브라우저 → 서버
  binary                  PCM16 LE 16kHz mono (크기 자유, 보통 20~40ms)
  {"type":"playing","on"} 브라우저가 에이전트 음성을 재생 중인지 (끼어들기 판정용)
  {"type":"played"|"cut","tid","seq"}  그 문장을 끝까지 재생함 | 재생 중 끊김 (고객이 실제로 들은 말, docs/09 B)
  {"type":"hangup"}
서버 → 브라우저
  {"type":"vad","state":"start"|"end"}
  {"type":"stt","text","ms","audio_s"}
  {"type":"agent","event":<SSE와 같은 이름>,"data":...}   delta · tool_call · tool_result · trace · done · error
  {"type":"audio","tid","seq","text","gen_ms"} 다음에 binary(wav) 한 개 (tid: 음성 턴 번호)
  {"type":"stop","reason"}                 재생 중단 (끼어들기)
  {"type":"hangup","reason"}               서버가 통화를 끝냄 (무응답) — 남은 소리를 들려준 뒤 끊는다
  {"type":"latency", ...}                  턴별 단계 시간 (ms, 발화 끝 판정 시점 기준)

끼어들기: 에이전트가 생성 중이거나 브라우저가 재생 중일 때 VAD가 발화 시작을 내면
- 재생을 멈추게 하고 남은 TTS를 버린다
- 생성 중이면 에이전트 생성기를 닫는다 → 텍스트 콘솔의 「중단」과 같이 그 턴은 기록되지 않는다.
  그 턴의 고객 발화는 버리지 않고 다음 발화 앞에 붙인다 (고객 입장에선 말을 이어 한 것)
- 이미 done이 난 턴(재생만 남은 경우)은 기록돼 있다 — 고객이 일부만 들었다는 점은 voice_turns에 played로 남긴다
무응답 (docs/09 C): 에이전트가 말을 마치고 silence_s 동안 아무도 말하지 않으면 「들리세요?」를 한 번 묻고,
그 뒤로도 silence_s 동안 말이 없으면 종료 안내 후 끊는다 (disposition no_response). 고객이 말하면 다시 센다.

- 다음 턴을 시작하기 전에 끊긴 턴의 기록을 고객이 실제로 들은 말로 맞춘다 (docs/09 B, 텍스트의 G3와 같은 원칙):
  재생 중 끊김 → 이력의 에이전트 말을 들린 문장까지로 (끊긴 문장은 「…」). 생성 중 끊김이라도 들린 문장이 있으면
  그 주고받음을 이력에 남긴다 (이어 붙이지 않음). 들린 말이 하나도 없으면 지금처럼 다음 발화에 이어 붙인다
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Optional

import httpx
import numpy as np

from csagent.agent.contracts import TurnInput
from csagent.obs import event
from csagent.voice.sentences import SentenceSplitter
from csagent.voice.vad import Endpointer

log = logging.getLogger("csagent.voice")


class VoiceClient:
    """scripts/voice_server.py 클라이언트."""

    def __init__(self, base_url: str, *, voice: str = "F1", steps: int = 8) -> None:
        self.base = base_url.rstrip("/")
        self.voice, self.steps = voice, steps
        self._http = httpx.AsyncClient(timeout=30.0, trust_env=False)

    async def stt(self, pcm: np.ndarray, degrade: str = "") -> dict:
        body = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
        r = await self._http.post(f"{self.base}/stt", params={"degrade": degrade}, content=body)
        r.raise_for_status()
        return r.json()

    async def tts(self, text: str, voice: Optional[str] = None) -> tuple[bytes, int]:
        r = await self._http.post(f"{self.base}/tts", json={"text": text, "voice": voice or self.voice,
                                                            "steps": self.steps})
        r.raise_for_status()
        return r.content, int(r.headers.get("X-Gen-Ms", 0))

    async def aclose(self) -> None:
        await self._http.aclose()


SILENCE_NUDGE = "여보세요, 제 말 들리세요?"
SILENCE_END = "응답이 없으셔서 통화를 종료할게요. 필요하시면 다시 연락 주세요."


class VoiceCall:
    """WebSocket 하나 = 통화 하나. run_turn은 기존 텍스트 턴 처리(app.py)와 같은 이벤트를 낸다."""

    def __init__(self, ws, session, agent, voice: VoiceClient, *,
                 turn_events: Callable, on_ended: Callable, endpointer: Optional[Endpointer] = None,
                 degrade: str = "", silence_s: float = 8.0) -> None:
        self.ws, self.s, self.agent, self.voice = ws, session, agent, voice
        self.turn_events = turn_events      # (session, utterance, cancel) → (이벤트이름, 데이터) 동기 생성기
        self.on_ended = on_ended
        self.ep = endpointer or Endpointer()
        self.degrade = degrade
        self.loop = asyncio.get_running_loop()
        self.send_lock = asyncio.Lock()
        self.playing = False                # 브라우저 보고
        self.gen_task: Optional[asyncio.Task] = None
        self.cancel = threading.Event()
        self.carry = ""                     # 끼어들기로 버려진 턴의 고객 발화
        self.tid = 0                        # 음성 턴 번호 (브라우저의 played·cut 보고와 맞춘다)
        self.heard: dict[tuple[int, int], str] = {}   # (tid, seq) → "full" | "cut"
        self.silence_s = silence_s
        self.last_activity = time.monotonic()   # 마지막으로 누군가 말한(재생·발화) 시각
        self.in_speech = False              # 고객이 말하는 중 (VAD start ~ end)
        self.nudged = False                 # 무응답으로 「들리세요?」를 이미 물었나
        self.watch: Optional[asyncio.Task] = None

    # ------------------------------------------------------------ 송신
    async def send(self, data: dict) -> None:
        async with self.send_lock:
            await self.ws.send_json(data)

    async def send_audio(self, meta: dict, wav: bytes) -> None:
        async with self.send_lock:       # 메타와 wav가 다른 메시지 사이에 끼지 않게
            await self.ws.send_json(meta)
            await self.ws.send_bytes(wav)

    async def greet(self, text: str) -> None:
        """인바운드 첫 인사 (고정 문구). 고객이 그 위에 말하면 브라우저가 재생을 끊는다 (playing 보고 → barge_in)."""
        wav, gen_ms = await self.voice.tts(text)
        await self.send_audio({"type": "audio", "seq": 0, "text": text, "gen_ms": gen_ms, "greeting": True}, wav)

    # ------------------------------------------------------------ 수신
    async def on_audio(self, data: bytes) -> None:
        t_recv = time.monotonic()
        pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        for ev in await asyncio.to_thread(self.ep.feed, pcm):
            self.last_activity = time.monotonic()
            self.in_speech = ev.kind == "start"
            if ev.kind == "start":
                self.nudged = False         # 고객이 말했다 — 무응답 카운트를 처음부터
                await self.send({"type": "vad", "state": "start"})
                if self.busy:
                    await self.barge_in()
            else:
                await self.send({"type": "vad", "state": "end"})
                # 프레임이 실시간으로 이어 온다고 보고, 이 프레임을 받은 시각에서 침묵 대기만큼 거슬러 올라간다
                # (오디오 스트림 시각과 벽시계를 처음에 한 번 맞추면, 프레임이 끊겼다 이어지는 클라이언트에서 어긋난다)
                speech_end_wall = t_recv - (ev.t - ev.speech_end_t)
                self.start_turn(ev.audio, time.monotonic(), speech_end_wall)

    def on_playback(self, ctl: dict) -> None:
        """브라우저 보고: 문장 하나를 끝까지 재생함(played) / 재생 중 끊김(cut)."""
        key = (int(ctl.get("tid", 0)), int(ctl.get("seq", 0)))
        if ctl.get("type") == "played" or key not in self.heard:
            self.heard[key] = "full" if ctl.get("type") == "played" else "cut"

    def _settle_heard(self) -> None:
        """직전 음성 턴이 끼어들기로 끊겼으면, 기록을 고객이 실제로 들은 말로 맞춘다 (docs/09 B)."""
        rec = self.s.voice_turns[-1] if self.s.voice_turns else None
        if rec is None or rec.get("settled") or not rec.get("barge_in"):
            return
        rec["settled"] = True
        parts = []
        for sent in rec["sentences"]:
            status = self.heard.get((rec["tid"], sent["seq"]))
            if status == "full":
                parts.append(sent["text"])
            else:
                if status == "cut":   # 듣다가 끊었다 — 끝까지 못 들었다는 표시
                    parts.append(sent["text"].rstrip(".?! ") + "…")
                break
        heard = " ".join(parts)
        rec["heard"] = heard
        history, state = self.s.state.history, self.s.state
        if rec["completed"]:
            if not (history and history[-1][0] == "agent"):
                return
            if heard:
                if history[-1][1] != heard:
                    history[-1] = ("agent", heard)
                    state.trace.add(state.turn, "heard_fix", {"tid": rec["tid"], "heard": heard, "completed": True})
            else:   # 한마디도 못 들었다 — 이 턴은 없었던 것처럼, 고객 말은 다음 발화에 이어 붙인다 (생성 중 끊김과 같게)
                del history[-2:]
                self.carry = rec["utterance"]
                state.trace.add(state.turn, "heard_fix", {"tid": rec["tid"], "heard": "", "completed": True})
        elif heard:   # 생성 중 끊겼지만 들린 문장이 있다 — 주고받은 것으로 남긴다
            history += [("customer", rec["utterance"]), ("agent", heard)]
            self.carry = ""
            state.trace.add(state.turn, "heard_fix", {"tid": rec["tid"], "heard": heard, "completed": False})
        event(log, "heard_fix", call_id=state.call_id, turn=state.turn, tid=rec["tid"],
              completed=rec["completed"], heard_chars=len(heard))

    def set_playing(self, on: bool) -> None:
        self.playing = on
        self.last_activity = time.monotonic()

    def start_silence_watch(self) -> None:
        if self.silence_s > 0 and self.watch is None:
            self.watch = asyncio.create_task(self._watch_silence())

    async def _watch_silence(self) -> None:
        """무응답 감시 — 에이전트가 말을 마친 뒤 아무도 말하지 않으면 한 번 묻고, 그래도 없으면 끊는다 (docs/09 C).
        고객이 말하지 않는 한 턴이 돌지 않아 통화가 무한정 열려 있던 문제."""
        while not self.s.ended:
            await asyncio.sleep(min(0.25, self.silence_s / 4))
            if self.busy or self.in_speech or time.monotonic() - self.last_activity < self.silence_s:
                continue
            state = self.s.state
            text = SILENCE_END if self.nudged else SILENCE_NUDGE
            state.history.append(("agent", text))   # 고객 대답 없이 한 말 — 다음 턴 모델이 알아야 한다
            state.trace.add(state.turn, "silence", {"text": text, "end": self.nudged})
            event(log, "silence", call_id=state.call_id, turn=state.turn, end=self.nudged)
            wav, gen_ms = await self.voice.tts(text)
            await self.send_audio({"type": "audio", "seq": 0, "text": text, "gen_ms": gen_ms, "silence": True}, wav)
            self.last_activity = time.monotonic()
            if self.nudged:
                self.s.ended, self.s.disposition = True, "no_response"
                await self.send({"type": "hangup", "reason": "no_response"})
                self.on_ended(self.s, "no_response")
                return
            self.nudged = True

    @property
    def busy(self) -> bool:
        return self.playing or (self.gen_task is not None and not self.gen_task.done())

    async def barge_in(self) -> None:
        """기다리지 않고 표시만 한다 — 수신 루프가 막히면 그동안 들어온 음성의 VAD가 밀린다."""
        active = self.gen_task is not None and not self.gen_task.done()
        await self.send({"type": "stop", "reason": "barge_in"})
        event(log, "barge_in", call_id=self.s.state.call_id, turn=self.s.state.turn, turn_active=active)
        if not active and self.s.voice_turns:   # 생성은 끝났고 재생 중에 끊었다 — 그 턴 기록에 끊김을 남긴다 (docs/09 B)
            self.s.voice_turns[-1]["barge_in"] = True
        self.cancel.set()

    def start_turn(self, audio: np.ndarray, t_end: float, speech_end_wall: float) -> None:
        prev, self.cancel = self.gen_task, threading.Event()
        self.gen_task = asyncio.create_task(self._turn(audio, t_end, speech_end_wall, self.cancel, prev))

    # ------------------------------------------------------------ 한 턴
    async def _turn(self, audio: np.ndarray, t_end: float, speech_end_wall: float,
                    cancel: threading.Event, prev: Optional[asyncio.Task]) -> None:
        lat: dict[str, Any] = {"vad_wait": round((t_end - speech_end_wall) * 1000)}
        ms = lambda: round((time.monotonic() - t_end) * 1000)   # noqa: E731
        try:
            stt = await self.voice.stt(audio, self.degrade)   # 전사는 앞 턴 정리와 겹쳐 돌린다
        except Exception as e:
            await self.send({"type": "error", "message": f"STT 실패: {type(e).__name__}: {e}"})
            return
        lat["stt"] = ms()
        if prev is not None:       # 끼어들기로 닫히는 앞 턴이 정리될 때까지 (carry가 정해진다)
            await asyncio.gather(prev, return_exceptions=True)
            lat["prev_wait"] = ms() - lat["stt"]
        self._settle_heard()       # 끊긴 앞 턴의 기록을 들은 말로 (재생 보고는 전사하는 동안 들어온다)
        text = stt["text"].strip()
        await self.send({"type": "stt", **stt})
        if cancel.is_set():
            self.carry = " ".join(x for x in (self.carry, text) if x)
            return
        # 전사가 비어도(기침 등으로 끼어들기가 걸린 경우) 버려진 앞 턴 발화가 있으면 그걸로 다시 응답한다
        utterance = " ".join(x for x in (self.carry, text) if x)
        self.carry = ""
        if not utterance:
            return

        splitter = SentenceSplitter()
        tts_q: asyncio.Queue = asyncio.Queue()
        self.tid += 1
        tid = self.tid
        record = {"turn": None, "tid": tid, "stt": stt, "utterance": utterance, "sentences": [], "latency": lat,
                  "completed": False, "barge_in": False}

        async def speak() -> None:
            seq = 0
            while True:
                sentence = await tts_q.get()
                if sentence is None or cancel.is_set():
                    return
                wav, gen_ms = await self.voice.tts(sentence)
                if cancel.is_set():
                    return
                seq += 1
                if seq == 1:
                    lat["first_audio"] = ms()
                # at_ms: 이 문장 음성을 브라우저로 보낸 시각 (발화 끝 판정 기준) — 데모 재생 간격의 원본
                record["sentences"].append({"seq": seq, "text": sentence, "gen_ms": gen_ms, "at_ms": ms()})
                await self.send_audio({"type": "audio", "tid": tid, "seq": seq, "text": sentence, "gen_ms": gen_ms}, wav)

        speaker = asyncio.create_task(speak())

        def put(sentence: str) -> None:
            if "first_sentence" not in lat:
                lat["first_sentence"] = ms()
            self.loop.call_soon_threadsafe(tts_q.put_nowait, sentence)

        def produce() -> None:
            """스레드에서 에이전트를 돌린다 (LLM·검색 호출이 동기라서)."""
            for name, data in self.turn_events(self.s, utterance, cancel):
                if name == "delta":
                    if "first_delta" not in lat:
                        lat["first_delta"] = ms()
                    for sentence in splitter.feed(data.get("text", "")):
                        put(sentence)
                elif name in ("say_end", "tool_call"):
                    # say 문자열이 닫히면 남은 조각도 문장이다 — 뒤의 JSON(도구 인자)이 다 나올 때까지 기다리지 않는다.
                    # 도구 앞뒤 say는 LLM 호출이 따로라 공백 없이 이어지므로 여기서 끊어야 붙지 않는다
                    for sentence in splitter.flush():
                        put(sentence)
                elif name == "done":
                    for sentence in splitter.flush():
                        put(sentence)
                    lat["done"] = ms()
                    record["turn"], record["completed"] = self.s.state.turn, True
                    data = {**data, "utterance": utterance}
                asyncio.run_coroutine_threadsafe(self.send({"type": "agent", "event": name, "data": data}),
                                                 self.loop).result()

        try:
            await asyncio.to_thread(produce)
        finally:
            tts_q.put_nowait(None)
            record["turn"] = self.s.state.turn    # 끼어들기로 닫힌 턴도 번호를 받았다 (trace와 맞춤)
        if cancel.is_set() and not record["completed"]:
            self.carry = utterance      # 기록 안 된 턴 — 고객 말은 다음 발화에 이어 붙인다
            speaker.cancel()
            record["barge_in"] = True
        else:
            await speaker
            record["barge_in"] = cancel.is_set()
        await self.send({"type": "latency", **lat})
        event(log, "voice_turn", call_id=self.s.state.call_id, turn=self.s.state.turn, **lat,
              completed=record["completed"], barge_in=record["barge_in"])
        self.s.voice_turns.append(record)
        self.last_activity = time.monotonic()
        if self.s.ended:
            self.on_ended(self.s)

    async def close(self) -> None:
        if self.watch is not None:
            self.watch.cancel()
        self.cancel.set()
        if self.gen_task is not None:
            await asyncio.gather(self.gen_task, return_exceptions=True)
