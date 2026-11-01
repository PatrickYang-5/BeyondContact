"""Prepare the project video, cropped HPB recording, and optimized demo GIFs.

Dependencies: Pillow and imageio-ffmpeg (or provide --ffmpeg).
Original GIFs are retained in Git-ignored figs/gif-originals/ before export.
"""

import argparse
from pathlib import Path
import shutil
import subprocess

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
GIF_ORIGINALS = ROOT / "figs" / "gif-originals"
GIF_NAMES = ("airflow.gif", "airflow60.gif", "airflow90.gif", "airflow135.gif")
HPB_SOURCE = ROOT / "figs" / "hybrid_push_blow.mp4"
PROJECT_SOURCE = ROOT / "figs" / "Video_11.1.mp4"
PROJECT_POSTER_SOURCE = ROOT / "figs" / "Video_page.png"
# Keep the top 8/9 of the recording (960 of 1080 source pixels).
# At the 1280-pixel web width, the identifying strip begins below y=640.
HPB_CROP = "crop=iw:ih*8/9:0:0,scale=1280:-2:flags=lanczos"


def run(ffmpeg, *args):
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def optimize_gifs(ffmpeg):
    GIF_ORIGINALS.mkdir(parents=True, exist_ok=True)
    for name in GIF_NAMES:
        original = GIF_ORIGINALS / name
        asset = ASSETS / name
        if not original.exists():
            if not asset.exists():
                raise FileNotFoundError(f"Missing GIF source: {asset}")
            shutil.move(str(asset), str(original))
        temporary = ASSETS / f"{asset.stem}.tmp.gif"
        try:
            run(
                ffmpeg, "-ignore_loop", "1", "-i", str(original),
                "-filter_complex",
                "[0:v]fps=10,scale=480:-1:flags=lanczos,split[frames][colors];"
                "[colors]palettegen=max_colors=64:stats_mode=diff[palette];"
                "[frames][palette]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                "-map_metadata", "-1", "-fps_mode", "passthrough", "-loop", "0", str(temporary),
            )
            temporary.replace(asset)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"{name}: {asset.stat().st_size / 1024 / 1024:.2f} MiB", flush=True)


def crop_hpb(ffmpeg):
    if not HPB_SOURCE.exists():
        raise FileNotFoundError(f"Missing HPB recording: {HPB_SOURCE}")
    video = ASSETS / "hybrid-push-blow.mp4"
    temporary = ASSETS / "hybrid-push-blow.tmp.mp4"
    try:
        run(
            ffmpeg, "-i", str(HPB_SOURCE), "-vf", HPB_CROP, "-r", "30",
            "-c:v", "libx264", "-preset", "fast", "-crf", "26",
            "-pix_fmt", "yuv420p", "-an", "-map_metadata", "-1",
            "-movflags", "+faststart", str(temporary),
        )
        temporary.replace(video)
    finally:
        temporary.unlink(missing_ok=True)

    # Generate a matching poster from the same crop, without accumulating crops.
    frame = ASSETS / "hpb-poster.tmp.png"
    try:
        run(ffmpeg, "-ss", "20", "-i", str(HPB_SOURCE), "-vf", HPB_CROP,
            "-frames:v", "1", str(frame))
        with Image.open(frame) as image:
            image.convert("RGB").save(ASSETS / "hpb-poster.webp", "WEBP", quality=85, method=6)
    finally:
        frame.unlink(missing_ok=True)
    print(f"hybrid-push-blow.mp4: {video.stat().st_size / 1024 / 1024:.2f} MiB, 1280 x 640", flush=True)


def prepare_project_poster():
    with Image.open(PROJECT_POSTER_SOURCE) as image:
        image.convert("RGB").save(ASSETS / "project-overview-poster.webp", "WEBP", quality=90, method=6)


def prepare_project_video(ffmpeg):
    if not PROJECT_SOURCE.exists():
        raise FileNotFoundError(f"Missing project video: {PROJECT_SOURCE}")
    video = ASSETS / "project-overview.mp4"
    temporary = ASSETS / "project-overview.tmp.mp4"
    scale = "scale=1920:-2:flags=lanczos"
    try:
        run(
            ffmpeg, "-i", str(PROJECT_SOURCE), "-map", "0:v:0", "-map", "0:a?",
            "-vf", scale, "-r", "30", "-c:v", "libx264", "-preset", "fast",
            "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-map_metadata", "-1", "-movflags", "+faststart", str(temporary),
        )
        temporary.replace(video)
    finally:
        temporary.unlink(missing_ok=True)

    prepare_project_poster()
    print(f"project-overview.mp4: {video.stat().st_size / 1024 / 1024:.2f} MiB, 1920 x 1080", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", help="Path to an FFmpeg executable")
    parser.add_argument("--only", choices=("gifs", "hpb", "project", "all"), default="all")
    args = parser.parse_args()
    ffmpeg = args.ffmpeg
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    ASSETS.mkdir(parents=True, exist_ok=True)
    if args.only in ("gifs", "all"):
        optimize_gifs(ffmpeg)
    if args.only in ("hpb", "all"):
        crop_hpb(ffmpeg)
    if args.only in ("project", "all"):
        prepare_project_video(ffmpeg)


if __name__ == "__main__":
    main()
