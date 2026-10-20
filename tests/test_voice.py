import json
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


def test_barge_in_during_playback_keeps_only_what_customer_heard():
    # 재생 중에 끊으면 이력의 에이전트 말은 들린 문장까지 (끊긴 문장은 「…」) — docs/09 B
    voice = FakeVoice(["택배 어디쯤이에요", "잠깐만요"])
    c, sid = _voice_client(ScriptedLLM([_say("배송중이에요. 내일 도착 예정이에요. 늦으면 고객센터로 문의해 주세요."),
                                        _say("네, 말씀하세요.")]), voice)
    with c.websocket_connect(f"/api/sessions/{sid}/voice") as ws:
        ws.send_bytes(_bytes(0.5, 0.3) + _bytes(0.7, 0))
        got = _until(ws, lambda m: m["type"] == "latency")
        tid = [m for m in got if m["type"] == "audio" and not m.get("greeting")][0]["tid"]
        ws.send_text(json.dumps({"type": "playing", "on": True}))
        ws.send_text(json.dumps({"type": "played", "tid": tid, "seq": 1}))
        ws.send_text(json.dumps({"type": "cut", "tid": tid, "seq": 2}))
        ws.send_bytes(_bytes(0.5, 0.3) + _bytes(0.7, 0))           # 재생 중에 끼어든다
        _until(ws, _is_done)
    s = c.get(f"/api/sessions/{sid}").json()
    agent = [t for r, t in s["history"] if r == "agent"]
    assert agent[1] == "배송중이에요. 내일 도착 예정이에요…"            # 세 번째 문장은 못 들었다
    assert [t for r, t in s["history"] if r == "customer"] == ["택배 어디쯤이에요", "잠깐만요"]


def test_barge_in_during_generation_records_heard_sentences():
    # 생성 중에 끊었어도 이미 들린 문장이 있으면 그 주고받음을 남긴다 (이어 붙이지 않음)
    voice = FakeVoice(["택배가 어디쯤", "아 됐어요"])
    long = "배송중이에요. " + "주문하신 상품은 배송 중이에요 " * 6 + "."
    c, sid = _voice_client(SlowLLM([_say(long), _say("네, 알겠습니다.")]), voice)
    with c.websocket_connect(f"/api/sessions/{sid}/voice") as ws:
        ws.send_bytes(_bytes(0.5, 0.3) + _bytes(0.7, 0))
        first = _until(ws, lambda m: m["type"] == "audio" and not m.get("greeting"))[-1]
        ws.send_text(json.dumps({"type": "played", "tid": first["tid"], "seq": 1}))
        ws.send_bytes(_bytes(0.5, 0.3) + _bytes(0.7, 0))           # 생성 중에 끼어든다
        got = _until(ws, _is_done)
    assert got[-1]["data"]["utterance"] == "아 됐어요"               # 이어 붙이지 않았다
    s = c.get(f"/api/sessions/{sid}").json()
    assert s["history"][1:5] == [["customer", "택배가 어디쯤"], ["agent", "배송중이에요."],
                                 ["customer", "아 됐어요"], ["agent", "네, 알겠습니다."]]


def test_silence_nudges_once_then_ends_call():
    # 고객이 말하지 않으면 「들리세요?」 한 번, 그래도 없으면 종료 안내 후 끊는다 (docs/09 C)
    voice = FakeVoice([])
    app = create_app(make_agent(ScriptedLLM([])), [SC], web_dir=None,
                     make_voice=lambda: (voice, Endpointer(_loud)), silence_s=0.2)
    c = TestClient(app)
    sid = c.post("/api/sessions", json={"scenario_id": "s1_demo", "mode": "voice"}).json()["session_id"]
    with c.websocket_connect(f"/api/sessions/{sid}/voice") as ws:
        got = _until(ws, lambda m: m["type"] == "hangup")
    said = [m["text"] for m in got if m["type"] == "audio" and m.get("silence")]
    assert said == ["여보세요, 제 말 들리세요?", "응답이 없으셔서 통화를 종료할게요. 필요하시면 다시 연락 주세요."]
    s = c.get(f"/api/sessions/{sid}").json()
    assert s["history"][-2:] == [["agent", said[0]], ["agent", said[1]]]


def test_customer_speech_resets_silence_count():
    voice = FakeVoice(["네 들려요"])
    app = create_app(make_agent(ScriptedLLM([_say("네, 말씀하세요.")])), [SC], web_dir=None,
                     make_voice=lambda: (voice, Endpointer(_loud)), silence_s=0.3)
    c = TestClient(app)
    sid = c.post("/api/sessions", json={"scenario_id": "s1_demo", "mode": "voice"}).json()["session_id"]
    with c.websocket_connect(f"/api/sessions/{sid}/voice") as ws:
        _until(ws, lambda m: m["type"] == "audio" and m.get("silence"))      # 「들리세요?」
        ws.send_bytes(_bytes(0.5, 0.3) + _bytes(0.7, 0))                    # 고객이 대답한다
        _until(ws, _is_done)
        got = _until(ws, lambda m: m["type"] == "audio" and m.get("silence"))   # 다시 조용해지면 처음부터
    assert got[-1]["text"] == "여보세요, 제 말 들리세요?"                     # 끊지 않고 다시 묻는다
