# Presentation media

These optimized assets are shared by the project website and root README. The website serves them using paths relative to `docs/`; it does not depend on the original `figs/` directory or the supplied manuscript PDF.

| Asset | Supplied source |
| --- | --- |
| `framework.webp` | `figs/overview.png` |
| `airflow-pipeline.webp` | `figs/extension.png` |
| `platforms.webp` | `figs/Showcase_newnew.png` |
| `hardware-system.webp` | `figs/new_system.png` |
| `hpb-results.webp` | `figs/HPB.png` |
| `rr-results.webp` | `figs/RR.png` |
| `hybrid-push-blow.mp4` | `figs/hybrid_push_blow.mp4` |
| `rr-60.mp4` | `figs/60度.mp4` |
| `rr-90.mp4` | `figs/90度.mp4` |
| `rr-135.mp4` | `figs/135度.mp4` |
| `airflow.gif` | `figs/gif-originals/airflow.gif` |
| `airflow60.gif` | `figs/gif-originals/airflow60.gif` |
| `airflow90.gif` | `figs/gif-originals/airflow90.gif` |
| `airflow135.gif` | `figs/gif-originals/airflow135.gif` |

Figures retain their original aspect ratios, use WebP at quality 90, and are limited to 1,400–2,400 pixels in width. Source transparency is composited onto an opaque white background before export, so figures remain white-backed in both the website and README, including dark-mode viewers. Opaque artwork and photographs retain their original colors. Embedded figures link to these larger versions. Image metadata is omitted.

To regenerate the figures from the local originals, run from the repository root with Python and Pillow installed:

```bash
python scripts/prepare_web_figures.py
```

Videos are H.264, 30 fps, YUV 4:2:0, encoded with CRF 26 and the fast preset. RR recordings are 1280 × 720. The HPB recording and poster are cropped to 1280 × 640, removing the bottom strip containing the institutional logo while preserving the arm, object, and target region. MP4 `faststart` places the playback metadata before the video payload. Full durations and playback speed are preserved; the web copies omit audio. RR clips are approximately 16, 9, and 9 seconds; the HPB compilation is approximately 5 minutes.

The supplied GIF originals are retained under the Git-ignored `figs/gif-originals/` directory. Their web copies are 480 × 300 at 10 fps, with a 64-color palette and infinite looping. They preserve the timing of the supplied excerpts with frame sampling; they are not the full recordings. The README displays all four near the top. The website uses the HPB animation in the hero, followed by HPB and selectable RR previews before the framework section. Full videos remain available in expandable recording panels. A pause control and static fallbacks for `prefers-reduced-motion` are included on the website.

To regenerate the GIFs, cropped HPB video, and matching poster with Python, Pillow, and imageio-ffmpeg installed:

```bash
python scripts/prepare_demo_media.py
```

Use `--only gifs` or `--only hpb` to update one group, or `--ffmpeg /path/to/ffmpeg` to use an installed FFmpeg executable.

Posters are WebP frames extracted at 20 seconds for HPB and at the beginning for RR. All public-facing filenames use ASCII. Keep these derivatives in the published `docs/` tree, and retain the source media separately when making future edits.
