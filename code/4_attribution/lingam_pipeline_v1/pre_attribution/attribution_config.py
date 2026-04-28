from __future__ import annotations

from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PIPELINE_ROOT / "outputs"
COMMON_OUTPUT_DIR = OUTPUT_ROOT / "common"

SSPS = ("ssp1", "ssp2", "ssp3", "ssp4", "ssp5")
YEARS = ("2040", "2060", "2080", "2100")
N_REPEATS = 100
SAMPLE_RATIO = 0.8
FREQ_THRESHOLD = 0.8
SIGN_THRESHOLD = 0.8

FONT_FAMILY = "Times New Roman"
FONT_SERIF_FALLBACKS = ("Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif")
BASE_FONT_SIZE = 9.0

FIGURE_FACE = "#FFFFFF"
GRID_COLOR = "#E6E2DD"
AXIS_COLOR = "#8A857F"
TEXT_SECONDARY = "#6F6A64"

GROUP_COLORS = {
    "Hydrogeology": "#7E8FA3",
    "Climate": "#9AAF9A",
    "Anthropogenic": "#C69074",
    "Target": "#8D6E63",
}

SIGN_COLORS = {
    "positive": "#C69074",
    "negative": "#8CA0B3",
}

SEQUENTIAL_CMAP_HEX = ("#FFFFFF", "#D8E0E5", "#AEBCCA", "#7E8FA3")


def ensure_common_output_dir(output_dir: str | Path | None = None) -> Path:
    path = Path(output_dir) if output_dir is not None else COMMON_OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path

