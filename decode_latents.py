"""Decode saved latents to MP4 — runs without transformer in VRAM."""
import sys, os, gc, time
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
from config import AppConfig
from pipeline import LTX2Pipeline
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.model.video_vae.tiling import TemporalTilingConfig, SpatialTilingConfig
from ltx_pipelines.utils.media_io import encode_video

if len(sys.argv) < 2:
    print("Usage: python3 decode_latents.py <latents.pt>")
    sys.exit(1)

latent_path = sys.argv[1]
data = torch.load(latent_path, weights_only=True, map_location="cpu")
samples = data["samples"]
fps = data.get("fps", 25.0)
seed = data.get("seed", 42)

print(f"Video latent: {tuple(samples.shape)}")

cfg = AppConfig.load()
device = torch.device("cuda")
pipe = LTX2Pipeline(cfg)
ledger = pipe._build_ledger()

t0 = time.perf_counter()

# Everything under inference_mode to prevent autograd VRAM waste
with torch.inference_mode():
    print("Loading video decoder...")
    video_decoder = ledger.video_decoder()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    generator = torch.Generator(device=device).manual_seed(seed)
    # No transformer during decode — full VRAM available, use default tiling
    tiling = TilingConfig.default()

    print("Decoding + encoding video...")
    decoded_video = vae_decode_video(samples.to(device), video_decoder, tiling, generator)

    out_path = Path(latent_path).with_suffix(".mp4")
    nf = samples.shape[2] * 8 + 1
    video_chunks_number = get_video_chunks_number(nf, tiling)

    encode_video(
        video=decoded_video,
        fps=fps,
        audio=None,
        output_path=str(out_path),
        video_chunks_number=video_chunks_number,
    )

print(f"\nDone! Output: {out_path}")
print(f"Decode time: {time.perf_counter()-t0:.0f}s")
