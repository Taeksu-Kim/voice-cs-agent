"""STT·TTS 후보 실측 — 27B 서버가 떠 있는 상태에서 모델 하나씩 올려 VRAM 최고치·지연·RTF·CER을 잰다.

  python scripts/voice_bench.py tts --out runs/voice/<이름>
  python scripts/voice_bench.py asr --model qwen-0.6b --wav-dir runs/voice/<이름>
  python scripts/voice_bench.py asr --model whisper-turbo --wav-dir runs/voice/<이름>

env: ~/miniconda3/envs/voice-eval (qwen-tts·qwen-asr·faster-whisper).
ASR 입력은 같은 합성음을 원음(24kHz)과 전화 대역(8kHz·300~3400Hz·μ-law)으로 열화한 두 조건.
합성음이라 실제 전화 음성보다 쉬운 조건이다.
"""
import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# 상담 응답에서 나올 법한 문장 — 숫자·영문·고유명사 섞음
SENTENCES = [
    "네, 확인해 드릴게요.",
    "주문하신 상품은 오늘 발송되어 내일 도착할 가능성이 커요.",
    "카드 분실 신고가 접수되었습니다. 새 카드는 영업일 기준 3일에서 5일 안에 받으실 수 있어요.",
    "토스페이로 결제하신 금액 2만 9천 원은 취소 후 영업일 기준 3일 이내에 환불됩니다.",
    "본인 확인을 위해 휴대폰으로 보내 드린 인증번호 여섯 자리를 말씀해 주세요.",
    "반품을 원하시면 상품 수령 후 7일 이내에 판매자에게 요청해 주셔야 해요.",
    "그 부분은 제가 바로 처리해 드리기 어려워서, 상담원에게 연결해 드릴게요. 잠시만 기다려 주세요.",
    "송장번호를 조회해 보니 아직 택배사에 등록되지 않았어요. 오늘 저녁 이후에 다시 확인해 주시겠어요?",
]


class GpuPeak:
    """nvidia-smi로 GPU 전체 사용량을 폴링 — CUDA 컨텍스트까지 포함한 실제 점유."""

    def __init__(self, interval=0.05):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()

    @staticmethod
    def used_mib():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.strip().splitlines()[0])

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used_mib())
            time.sleep(self.interval)

    def __enter__(self):
        self.base = self.used_mib()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def to_telephone(wav, sr):
    """8kHz 리샘플 + 300~3400Hz 대역 + μ-law 8bit 왕복."""
    import torch
    import torchaudio.functional as AF
    x = torch.from_numpy(wav.astype(np.float32))
    x = AF.resample(x, sr, 8000)
    x = AF.highpass_biquad(x, 8000, 300.0)
    x = AF.lowpass_biquad(x, 8000, 3400.0)
    x = x / max(x.abs().max().item(), 1e-6) * 0.9
    x = AF.mu_law_decoding(AF.mu_law_encoding(x, 256), 256)
    return x.numpy(), 8000


SINO = dict(zip("0123456789", "영일이삼사오육칠팔구"))


def normalize(s):
    """공백·문장부호 제거 + 숫자를 한 자리씩 한자어 수사로 (「3일」=「삼 일」, 「2만 9천」=「이만 구천」).
    SENTENCES의 숫자가 전부 한 자리라 이걸로 충분하다 — 여러 자리 숫자를 넣으면 고칠 것."""
    s = "".join(SINO.get(c, c) for c in s)
    return re.sub(r"[\s\.,\?!~·'\"]", "", s)


def cmd_tts_supertonic(args):
    """Supertonic (ONNX, CPU 전용) — VRAM을 쓰지 않는다."""
    from supertonic import TTS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tts = TTS(model=args.model, intra_op_num_threads=args.threads)
    load_s = time.time() - t0
    style = tts.get_voice_style(args.speaker)
    tts.synthesize("안녕하세요.", voice_style=style, lang="ko")  # warmup
    rows = []
    for i, text in enumerate(SENTENCES):
        t = time.time()
        wav, _ = tts.synthesize(text, voice_style=style, lang="ko", total_steps=args.steps)
        el = time.time() - t
        wav = np.asarray(wav).squeeze()
        dur = len(wav) / tts.sample_rate
        sf.write(out / f"{i:02d}.wav", wav, tts.sample_rate)
        rows.append({"i": i, "text": text, "gen_s": round(el, 3), "audio_s": round(dur, 2), "rtf": round(el / dur, 3)})
        print(rows[-1])
    summary = {"model": args.model, "speaker": args.speaker, "steps": args.steps, "threads": args.threads,
               "load_s": round(load_s, 1), "device": "cpu",
               "rtf_mean": round(float(np.mean([r["rtf"] for r in rows])), 3), "rows": rows}
    (out / "tts.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print({k: v for k, v in summary.items() if k != "rows"})


def cmd_tts(args):
    if args.model.startswith("supertonic"):
        return cmd_tts_supertonic(args)
    import torch
    from qwen_tts import Qwen3TTSModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with GpuPeak() as g:
        t0 = time.time()
        m = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=torch.bfloat16)
        load_s = time.time() - t0
        print("speakers:", m.get_supported_speakers())
        load_peak = g.peak
        m.generate_custom_voice(text="안녕하세요.", speaker=args.speaker, language="Korean")  # warmup
        rows = []
        for i, text in enumerate(SENTENCES):
            torch.cuda.synchronize()
            t = time.time()
            wavs, sr = m.generate_custom_voice(text=text, speaker=args.speaker, language="Korean")
            torch.cuda.synchronize()
            el = time.time() - t
            dur = len(wavs[0]) / sr
            sf.write(out / f"{i:02d}.wav", wavs[0], sr)
            rows.append({"i": i, "text": text, "gen_s": round(el, 2), "audio_s": round(dur, 2), "rtf": round(el / dur, 2)})
            print(rows[-1])
    summary = {
        "model": args.model, "speaker": args.speaker, "load_s": round(load_s, 1),
        "gpu_base_mib": g.base, "gpu_after_load_mib": load_peak, "gpu_peak_mib": g.peak,
        "delta_peak_mib": g.peak - g.base, "torch_max_alloc_mib": torch.cuda.max_memory_allocated() // 2**20,
        "rtf_mean": round(float(np.mean([r["rtf"] for r in rows])), 2), "rows": rows,
    }
    (out / "tts.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print({k: v for k, v in summary.items() if k != "rows"})


def load_asr(name):
    if name.startswith("qwen"):
        import torch
        from qwen_asr import Qwen3ASRModel
        repo = {"qwen-0.6b": "Qwen/Qwen3-ASR-0.6B", "qwen-1.7b": "Qwen/Qwen3-ASR-1.7B"}[name]
        m = Qwen3ASRModel.from_pretrained(repo, device_map="cuda:0", dtype=torch.bfloat16, max_inference_batch_size=1)
        return lambda wav, sr: m.transcribe((wav, sr), language="Korean")[0].text
    from faster_whisper import WhisperModel
    m = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
    return lambda wav, sr: "".join(s.text for s in m.transcribe(_to16k(wav, sr), language="ko", beam_size=1)[0])


def _to16k(wav, sr):
    import torch
    import torchaudio.functional as AF
    return AF.resample(torch.from_numpy(wav.astype(np.float32)), sr, 16000).numpy()


def cmd_asr(args):
    import jiwer
    import torch

    wdir = Path(args.wav_dir)
    refs = {r["i"]: r["text"] for r in json.loads((wdir / "tts.json").read_text())["rows"]}
    with GpuPeak() as g:
        t0 = time.time()
        asr = load_asr(args.model)
        load_s = time.time() - t0
        w0, s0 = sf.read(wdir / "00.wav")
        asr(w0, s0)  # warmup
        rows = []
        for i, ref in refs.items():
            wav, sr = sf.read(wdir / f"{i:02d}.wav")
            for cond in ("clean", "tel8k"):
                w, s = (wav, sr) if cond == "clean" else to_telephone(wav, sr)
                torch.cuda.synchronize()
                t = time.time()
                hyp = asr(w, s)
                el = time.time() - t
                cer = jiwer.cer(normalize(ref), normalize(hyp))
                rows.append({"i": i, "cond": cond, "cer": round(cer, 3), "s": round(el, 2), "rtf": round(el / (len(w) / s), 3), "hyp": hyp})
                print(rows[-1])
    summary = {"model": args.model, "load_s": round(load_s, 1), "gpu_base_mib": g.base, "gpu_peak_mib": g.peak,
               "delta_peak_mib": g.peak - g.base}
    for cond in ("clean", "tel8k"):
        rs = [r for r in rows if r["cond"] == cond]
        summary[cond] = {"cer_mean": round(float(np.mean([r["cer"] for r in rs])), 3),
                         "rtf_mean": round(float(np.mean([r["rtf"] for r in rs])), 3)}
    summary["rows"] = rows
    (wdir / f"asr_{args.model}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("tts")
    t.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    t.add_argument("--speaker", default="Sohee", help="Qwen: Sohee 등 / Supertonic: M1~M5, F1~F5")
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=8, help="Supertonic 합성 스텝 (기본 8)")
    t.add_argument("--threads", type=int, default=None, help="Supertonic ONNX intra-op 스레드")
    a = sp.add_parser("asr")
    a.add_argument("--model", required=True, choices=["qwen-0.6b", "qwen-1.7b", "whisper-turbo"])
    a.add_argument("--wav-dir", required=True)
    args = p.parse_args()
    {"tts": cmd_tts, "asr": cmd_asr}[args.cmd](args)
