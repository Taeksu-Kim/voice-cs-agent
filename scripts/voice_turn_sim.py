"""통화 한 턴(STT → 리랭크 → LLM → TTS)을 반복하며 단계별 지연을 잰다 — 모든 모델을 GPU에 같이 올린 상태에서
VRAM 초과 시 Windows(WDDM)가 안 쓰는 메모리를 시스템 RAM으로 밀어내고 되가져오는 비용이 지연에 얼마나 섞이는지 보는 용도.

  python scripts/voice_turn_sim.py --wav runs/voice/tts06_sohee_r2/07.wav --turns 4 [--no-tts]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from voice_bench import GpuPeak, load_asr, to_telephone  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
env = dict(l.strip().split("=", 1) for l in open(ROOT / ".env") if "=" in l and not l.startswith("#"))


def post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


p = argparse.ArgumentParser()
p.add_argument("--wav", required=True)
p.add_argument("--stt", default="whisper-turbo")
p.add_argument("--turns", type=int, default=4)
p.add_argument("--no-tts", action="store_true")
args = p.parse_args()

asr = load_asr(args.stt)
tts = None
if not args.no_tts:
    from qwen_tts import Qwen3TTSModel
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device_map="cuda:0", dtype=torch.bfloat16)
wav, sr = to_telephone(*sf.read(args.wav))
docs = [f"문서 {i}: 결제 취소 시 환불은 영업일 기준 3~5일이 걸리며 카드사에 따라 다를 수 있습니다." * 4 for i in range(20)]
llm = env["LLM_URL"].rstrip("/")
print("GPU", GpuPeak.used_mib(), "MiB", flush=True)

for turn in range(args.turns):
    t = {}
    s = time.time(); text = asr(wav, sr); t["stt"] = time.time() - s
    s = time.time()
    post(env["RERANKER_URL"].rstrip("/") + "/rerank", {"model": "qwen3-reranker", "query": text, "documents": docs})
    t["rerank"] = time.time() - s
    s = time.time()
    r = post(llm + "/chat/completions", {
        "model": env["LLM_MODEL"], "max_tokens": 60, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": f"상담원으로서 한 문장으로 답해: {text}"}]})
    t["llm"] = time.time() - s
    answer = r["choices"][0]["message"]["content"]
    if tts:
        s = time.time(); tts.generate_custom_voice(text=answer[:60], speaker="Sohee", language="Korean"); t["tts"] = time.time() - s
    print(f"turn {turn}: " + "  ".join(f"{k} {v:.2f}s" for k, v in t.items()) + f"  | GPU {GpuPeak.used_mib()} MiB", flush=True)
