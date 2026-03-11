"""Decode audio from saved latents and mux into existing MP4."""
import sys, os, time
from pathlib import Path

app_dir = str(Path(__file__).parent)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

ltx_packages = Path(os.environ.get("LTX_PACKAGES", "/home/alex/LTX-2/packages"))
for sub in ("ltx-core/src", "ltx-pipelines/src"):
    p = str(ltx_packages / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from audio_utils import normalize_output_audio
from config import AppConfig
from pipeline import LTX2Pipeline
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio

latent_path = sys.argv[1]
data = torch.load(latent_path, weights_only=True, map_location="cpu")
audio_latent = data.get("audio")

if audio_latent is None:
    print("No audio latent found")
    sys.exit(1)

print(f"Audio latent: {tuple(audio_latent.shape)}")

cfg = AppConfig.load()
pipe = LTX2Pipeline(cfg)
ledger = pipe._build_ledger()

t0 = time.perf_counter()
print("Decoding audio on CPU...")

# Build audio decoder and vocoder on CPU (avoid GPU entirely)
with torch.inference_mode():
    audio_dec = ledger.audio_decoder().cpu().float()
    voc = ledger.vocoder().cpu().float()
    audio = vae_decode_audio(audio_latent.float(), audio_dec, voc)
    audio = normalize_output_audio(audio)

print(f"Audio decoded in {time.perf_counter()-t0:.1f}s")
sr = audio.sampling_rate
waveform = audio.waveform
duration = waveform.shape[-1] / sr
print(f"Waveform: {tuple(waveform.shape)}, {duration:.1f}s at {sr}Hz")

# Save as WAV
import torchaudio
wav_path = Path(latent_path).with_suffix(".wav")
wav_data = waveform.squeeze(0) if waveform.dim() == 3 else waveform
if wav_data.dim() == 1:
    wav_data = wav_data.unsqueeze(0)
torchaudio.save(str(wav_path), wav_data.cpu(), sr)
print(f"Saved: {wav_path}")

# Mux into MP4
mp4_path = Path(latent_path).with_suffix(".mp4")
out_path = mp4_path.with_stem(mp4_path.stem + "_with_audio")
if mp4_path.exists():
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp4_path), "-i", str(wav_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(out_path),
    ], check=True, capture_output=True)
    print(f"Muxed: {out_path}")
