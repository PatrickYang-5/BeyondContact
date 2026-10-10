"""Export the shared README/site figures with an opaque white background.

Run from any directory with Python and Pillow installed. Source figures stay
in the Git-ignored figs/ directory; only docs/assets/ derivatives are replaced.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "figs"
DESTINATION = ROOT / "docs" / "assets"

# The supplied RR figure contains about 94 million pixels.
Image.MAX_IMAGE_PIXELS = 120_000_000

FIGURES = (
    ("overview.png", "framework.webp", 2400),
    ("extension.png", "airflow-pipeline.webp", 2000),
    ("Showcase_newnew.png", "platforms.webp", 1400),
    ("new_system.png", "hardware-system.webp", 1800),
    ("HPB.png", "hpb-results.webp", 2400),
    ("RR.png", "rr-results.webp", 2400),
)


def main():
    missing = [name for name, _, _ in FIGURES if not (SOURCE / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local source figures: {', '.join(missing)}")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for source_name, asset_name, width in FIGURES:
        with Image.open(SOURCE / source_name) as source:
            source.thumbnail((width, width * 2), Image.Resampling.LANCZOS)
            rgba = source.convert("RGBA")
            # Preserve opaque artwork and blend translucent edges onto white.
            # convert('RGB') alone would discard alpha and expose hidden black.
            flattened = Image.new("RGB", rgba.size, (255, 255, 255))
            flattened.paste(rgba, mask=rgba.getchannel("A"))
            flattened.save(
                DESTINATION / asset_name, "WEBP", quality=90, method=6
            )
        print(f"{asset_name}: {flattened.width} x {flattened.height}, white background")


if __name__ == "__main__":
    main()
