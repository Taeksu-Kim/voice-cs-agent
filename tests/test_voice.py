import time

import numpy as np
from fastapi.testclient import TestClient

from csagent.llm import ScriptedLLM
from csagent.server.app import create_app
from csagent.voice.sentences import SentenceSplitter
from csagent.voice.vad import CHUNK, SR, Endpointer
from tests.test_agent_evaluation import _say, make_agent
from tests.test_server import SC


def _loud(chunk):   # 가짜 VAD: 진폭으로 말소리 판정
    return 1.0 if np.abs(chunk).mean() > 0.05 else 0.0


def _pcm(seconds, amp):
    return np.full(int(seconds * SR), amp, dtype=np.float32)


def _bytes(seconds, amp):
    return (_pcm(seconds, amp) * 32767).astype("<i2").tobytes()


def test_splitter_cuts_on_sentence_end_not_decimal():
    sp = SentenceSplitter()
    out = []
    for piece in ["주문하신 상품은 1.", "5kg이에요. 내일 도", "착해요? 네", "!"]:
        out += sp.feed(piece)
    out += sp.flush()
    assert out == ["주문하신 상품은 1.5kg이에요.", "내일 도착해요?", "네!"]


def test_splitter_cuts_long_text_at_comma():
    sp = SentenceSplitter(max_chars=20)
    out = sp.feed("가나다라마바사, 아자차카타파하 가나다라마바사아자차")
    assert out[0] == "가나다라마바사,"


def test_endpointer_start_end_and_preroll():
    ep = Endpointer(_loud, min_speech_ms=250, min_silence_ms=600, pre_roll_ms=300)
    evs = ep.feed(np.concatenate([_pcm(1.0, 0), _pcm(0.1, 0.3)]))
    assert evs == []                                   # 250ms 미만 말소리는 시작으로 안 본다
    evs = ep.feed(np.concatenate([_pcm(0.5, 0.3), _pcm(0.7, 0)]))
    assert [e.kind for e in evs] == ["start", "end"]
    start, end = evs
    assert abs(start.t - 1.0) < 0.05                  # 시작 시각은 말소리가 처음 이어진 시점
    assert abs(end.speech_end_t - 1.6) < 0.05
    # 발화 구간 = 앞 여유(300ms) + 말소리 0.6s + 침묵 대기
    assert len(end.audio) >= int(0.9 * SR) // CHUNK * CHUNK


class FakeVoice:
    def __init__(self, texts):
        self.texts = list(texts)
        self.tts_calls = []

    async def stt(self, pcm, degrade=""):
        return {"text": self.texts.pop(0), "ms": 1, "audio_s": round(len(pcm) / SR, 2)}

    async def tts(self, text, voice=None):
        self.tts_calls.append(text)
        return b"RIFF-fake", 1

    async def aclose(self):
        pass


class SlowLLM(ScriptedLLM):
    def stream_chat(self, messages, **kw):
        for item in super().stream_chat(messages, **kw):
            time.sleep(0.03)
            yield item


def _voice_client(llm, voice):
    app = create_app(make_agent(llm), [SC], web_dir=None,
                     make_voice=lambda: (voice, Endpointer(_loud)))
    c = TestClient(app)
    sid = c.post("/api/sessions", json={"scenario_id": "s1_demo", "mode": "voice"}).json()["session_id"]
    return c, sid


def _until(ws, pred, limit=500):
    got = []
    for _ in range(limit):
        m = ws.receive()
        if m.get("bytes") is not None:
            got.append({"type": "_wav"})
            continue
        msg = __import__("json").loads(m["text"])
        got.append(msg)
        if pred(msg):
            return got
    raise AssertionError(got)


def _is_done(m):
    return m["type"] == "agent" and m["event"] == "done"


def test_voice_turn_stt_agent_tts_latency():
    voice = FakeVoice(["택배 어디쯤이에요"])
    c, sid = _voice_client(ScriptedLLM([_say("배송중이에요. 내일 도착해요.")]), voice)
    with c.websocket_connect(f"/api/sessions/{sid}/voice") as ws:
        ws.send_bytes(_bytes(0.5, 0.3) + _bytes(0.7, 0))
        got = _until(ws, lambda m: m["type"] == "latency")
    types = [m["type"] for m in got]
    assert types[:4] == ["audio", "_wav", "vad", "vad"] and "stt" in types
    audio = [m for m in got if m["type"] == "audio"]
    assert audio[0]["greeting"] and "AI 상담원" in audio[0]["text"]   # 접속하자마자 인사
    assert [m["text"] for m in audio[1:]] == ["배송중이에요.", "내일 도착해요."]
    assert types.count("_wav") == 3
    lat = got[-1]
    assert {"stt", "first_delta", "first_sentence", "first_audio", "done"} <= lat.keys()
    s = c.get(f"/api/sessions/{sid}").json()
    assert s["mode"] == "voice" and s["history"][1] == ["customer", "택배 어디쯤이에요"]


def test_barge_in_cancels_turn_and_carries_utterance():
    voice = FakeVoice(["택배가", "어디쯤 왔어요"])
    long = "확인해 보니 " + "주문하신 상품은 배송 중이에요 " * 6 + "."
    c, sid = _voice_client(SlowLLM([_say(long), _say("배송중이에요.")]), voice)
    with c.websocket_connect(f"/api/sessions/{sid}/voice") as ws:
        ws.send_bytes(_bytes(0.5, 0.3) + _bytes(0.7, 0))
        _until(ws, lambda m: m["type"] == "agent" and m["event"] == "delta")   # 생성 중에
        ws.send_bytes(_bytes(0.5, 0.3) + _bytes(0.7, 0))                       # 고객이 끼어든다
        got = _until(ws, _is_done)
    assert any(m["type"] == "stop" for m in got)
    assert got[-1]["data"]["utterance"] == "택배가 어디쯤 왔어요"
    s = c.get(f"/api/sessions/{sid}").json()
    customer = [t for r, t in s["history"] if r == "customer"]
    assert customer == ["택배가 어디쯤 왔어요"]                 # 끊긴 턴은 기록되지 않는다
