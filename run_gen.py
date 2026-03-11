"""One-shot long video generation — 30s at full res with NAG + audio."""
import sys, os, gc, logging, datetime, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app_dir = str(Path(__file__).parent)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

ltx_packages = Path(os.environ.get("LTX_PACKAGES", "/home/alex/LTX-2/packages"))
for sub in ("ltx-core/src", "ltx-pipelines/src"):
    p = str(ltx_packages / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from config import AppConfig
from pipeline import LTX2Pipeline
from long_video import LTXVLongVideoService, UpscaleMode
from long_video_presets import seconds_to_frames, calculate_chunks

cfg = AppConfig.load()
cfg.nag_enabled = True
cfg.nag_scale = 11.0
cfg.ffn_chunks = 2

pipe = LTX2Pipeline(cfg)
service = LTXVLongVideoService()

W, H = 768, 512
TOTAL_SECONDS = 30.0
FPS = 25.0
SEED = 7777

total_frames = seconds_to_frames(TOTAL_SECONDS, FPS)
tile_size = 97
overlap = 32
num_chunks, actual_frames = calculate_chunks(total_frames, tile_size, overlap)

print(f"Target: {TOTAL_SECONDS}s = {total_frames} frames at {W}x{H}")
print(f"Chunks: {num_chunks} (tile={tile_size}, overlap={overlap})")
print(f"Actual frames: {actual_frames} = {actual_frames/FPS:.1f}s")

prompt = (
    "A cinematic scene. Amateur handheld candid video in natural daylight on a bustling "
    "urban sidewalk, a young woman with short dark hair, silver lip piercing, and four "
    "small pointed ears nervously tugs her lace choker with her left hand while wearing "
    "a casual tank top and jeans, wide eyes darting right toward an unseen interviewer "
    "as his arm extends a handheld microphone from the right frame corner toward her face. "
    "She hesitantly smiles and stammers in response to his off-screen voice, pedestrians "
    "blurring past brick storefronts and parked cars in the background as overhead power "
    "lines sway gently in a light breeze, casting shifting long shadows and intensifying "
    "lens flares from the high sun. She reacts with a slight frown and averted gaze "
    "pulling back from the mic. Camera does a slow handheld zoom-in on her face with "
    "subtle shake, warm sunlight filtering through leaves causing golden flares to dance "
    "across her skin as urban haze drifts faintly."
)

t0 = time.perf_counter()

def progress(chunk_idx, total, frames_gen, total_f, elapsed):
    print(f"  Chunk {chunk_idx+1}/{total} — {frames_gen}/{total_f}f — {elapsed:.0f}s elapsed")

accumulated, accumulated_audio = service.generate(
    pipeline=pipe,
    prompt=prompt,
    width=W,
    height=H,
    total_frames=total_frames,
    fps=FPS,
    temporal_tile_size=tile_size,
    temporal_overlap=overlap,
    adain_factor=0.3,
    seed=SEED,
    upscale_mode=UpscaleMode.SPATIAL_FINAL,
    long_memory_strength=0.0,
    progress_callback=progress,
)

print(f"\nGeneration done in {time.perf_counter()-t0:.0f}s")
print(f"Video latent: {tuple(accumulated['samples'].shape)}")
if accumulated_audio is not None:
    print(f"Audio latent: {tuple(accumulated_audio.shape)}")

# Save latents — decode in separate process to free transformer VRAM
out_dir = cfg.ensure_output_dir()
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
latent_path = out_dir / f"ltx2_30s_{ts}_s{SEED}.pt"

torch.save({
    "samples": accumulated["samples"].cpu(),
    "audio": accumulated_audio.cpu() if accumulated_audio is not None else None,
    "fps": FPS,
    "seed": SEED,
}, str(latent_path))

print(f"Latents saved: {latent_path}")
print(f"Decoding video + audio...")

# Decode in subprocess (frees all transformer VRAM)
import subprocess
result = subprocess.run(
    [sys.executable, "decode_latents.py", str(latent_path)],
    cwd=app_dir, capture_output=False,
)

# Decode audio and mux
mp4_path = latent_path.with_suffix(".mp4")
if mp4_path.exists() and accumulated_audio is not None:
    print("Decoding audio...")
    result2 = subprocess.run(
        [sys.executable, "decode_audio.py", str(latent_path)],
        cwd=app_dir, capture_output=False,
    )

print(f"\nTotal wall time: {time.perf_counter()-t0:.0f}s")
