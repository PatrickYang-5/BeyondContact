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

Figures retain their original aspect ratios, use WebP at quality 90, and are limited to 1,400–2,400 pixels in width. Source transparency is composited onto an opaque white background before export, so figures remain white-backed in both the website and README, including dark-mode viewers. Opaque artwork and photographs retain their original colors. Embedded figures link to these larger versions. Image metadata is omitted.

To regenerate the figures from the local originals, run from the repository root with Python and Pillow installed:

```bash
python scripts/prepare_web_figures.py
```

Videos are H.264, 1280 × 720, 30 fps, YUV 4:2:0, encoded with CRF 26 and the fast preset. MP4 `faststart` places the playback metadata before the video payload. Full durations and playback speed are preserved; the web copies omit audio. RR clips are approximately 16, 9, and 9 seconds; the HPB compilation is approximately 5 minutes.

Posters are WebP frames extracted at 20 seconds for HPB and at the beginning for RR. All public-facing filenames use ASCII. Keep these derivatives in the published `docs/` tree, and retain the source media separately when making future edits.
