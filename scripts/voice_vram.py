"""STT+TTS를 한 프로세스에 올렸을 때 VRAM 단계별 점유 — voice_bench.py의 보조.

  python scripts/voice_vram.py --stt whisper-turbo
  python scripts/voice_vram.py --stt qwen-0.6b --skip-tts

단계: CUDA 컨텍스트 → STT 로드 → TTS 로드 → STT 추론 → TTS 추론. 각 단계의 GPU 전체 사용량(nvidia-smi)을 찍는다.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from voice_bench import GpuPeak, SENTENCES, load_asr, to_telephone  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--stt", default="whisper-turbo", choices=["whisper-turbo", "qwen-0.6b", "none"])
p.add_argument("--skip-tts", action="store_true")
p.add_argument("--hold", type=int, default=0, help="측정 후 모델을 올린 채 N초 대기 (다른 서버 속도 측정용)")
args = p.parse_args()

used = GpuPeak.used_mib
base = used()
marks = [("시작", base)]
with GpuPeak() as g:
    torch.zeros(1, device="cuda")
    marks.append(("CUDA 컨텍스트", used()))
    asr = None
    if args.stt != "none":
        asr = load_asr(args.stt)
        marks.append((f"STT {args.stt} 로드", used()))
    tts = None
    if not args.skip_tts:
        from qwen_tts import Qwen3TTSModel
        tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device_map="cuda:0", dtype=torch.bfloat16)
        n = sum(x.numel() for x in tts.model.parameters())
        dt = {str(x.dtype) for x in tts.model.parameters()}
        print(f"TTS params {n/1e6:.0f}M dtypes {dt}")
        marks.append(("TTS 로드", used()))
    wav = sr = None
    if tts:
        t = time.time()
        wavs, sr = tts.generate_custom_voice(text=SENTENCES[7], speaker="Sohee", language="Korean")
        wav = wavs[0]
        marks.append((f"TTS 추론 (RTF {(time.time()-t)/(len(wav)/sr):.2f})", used()))
    if asr and wav is not None:
        w, s = to_telephone(wav, sr)
        print("STT:", asr(w, s))
        marks.append(("STT 추론", used()))
    time.sleep(0.3)

prev = base
for name, v in marks:
    print(f"{name:28s} {v:6d} MiB  (+{v - prev:5d}, 누적 +{v - base:5d})")
    prev = v
print(f"최고치 {g.peak} MiB (누적 +{g.peak - base}),  torch max_alloc {torch.cuda.max_memory_allocated() // 2**20} MiB", flush=True)
if args.hold:
    print("HOLDING", flush=True)
    time.sleep(args.hold)
