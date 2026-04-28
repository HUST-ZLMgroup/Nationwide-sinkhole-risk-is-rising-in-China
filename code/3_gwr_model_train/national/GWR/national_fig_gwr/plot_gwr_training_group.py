from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import jenkspy
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
GWR_ROOT = SCRIPT_DIR.parent
CODE_V2_ROOT = SCRIPT_DIR.parents[3]
PROJECT_ROOT = CODE_V2_ROOT.parent
if str(CODE_V2_ROOT) not in sys.path:
    sys.path.append(str(CODE_V2_ROOT))

MODEL_PATH = GWR_ROOT / "gwr_model_national_GWR.pkl"
ARTIFACTS_PATH = GWR_ROOT / "gwr_classification_artifacts.pkl"
TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "Extracted_HAVE_future"
    / "Positive_Negative_balanced"
    / "AllFeatures_Positive_Negative_balanced_25366_ssp1_cleaned.csv"
)
METRICS_PATH = SCRIPT_DIR / "gwr_classification_metrics_rskf5x10_boot1000.csv"
CV_CACHE_PATH = SCRIPT_DIR / "_cache" / "gwr_cls_rskf5x10_boot1000_seed2026_debug1.pkl.gz"
DEFAULT_OUTPUT = SCRIPT_DIR / "final" / "gwr_training.svg"
DEFAULT_DECISION_THRESHOLD = 0.5

FIGURE_FACE = "#FFFFFF"
AXIS_COLOR = "#8A857F"
GRID_COLOR = "#E6E2DD"
TEXT_SECONDARY = "#6F6A64"
TEXT_COLOR = "#000000"

COLORS = {
    "hydro": "#7E8FA3",
    "climate": "#9AAF9A",
    "anthro": "#C69074",
    "prob": "#8D6E63",
    "low": "#8CA0B3",
    "neutral": "#C7BEB4",
    "dark": "#5F5A55",
    "test": "#7E8FA3",
    "train": "#C69074",
}

CLASS_COLORS = ["#E9ECEC", "#D6E0DA", "#BFCBBB", "#A58F80", "#8D6E63"]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": FIGURE_FACE,
            "axes.facecolor": FIGURE_FACE,
            "savefig.facecolor": FIGURE_FACE,
            "savefig.edgecolor": FIGURE_FACE,
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.5,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_gzip_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def sigmoid_from_metadata(raw_scores: np.ndarray, metadata: dict) -> np.ndarray:
    raw = np.asarray(raw_scores, dtype=float).reshape(-1)
    center = float(metadata["center"])
    scale = float(metadata["scale"])
    clip_z = float(metadata.get("clip_z", 6.0))
    z = (raw - center) / scale
    z = np.clip(z, -clip_z, clip_z)
    return 1.0 / (1.0 + np.exp(-z))


def binary_labels_from_cutoff(labels: np.ndarray, cutoff: float) -> np.ndarray:
    return (np.asarray(labels, dtype=float).reshape(-1) > float(cutoff)).astype(int)


def validation_split_indices(probabilities: np.ndarray, y_binary: np.ndarray, artifacts: dict) -> tuple[np.ndarray, np.ndarray]:
    valid_idx = np.flatnonzero(np.isfinite(np.asarray(probabilities, dtype=float).reshape(-1)))
    y_valid_source = y_binary[valid_idx]
    train_idx, validation_idx = train_test_split(
        valid_idx,
        test_size=float(artifacts["validation_size"]),
        random_state=int(artifacts["random_state"]),
        stratify=y_valid_source,
    )
    return np.asarray(train_idx, dtype=int), np.asarray(validation_idx, dtype=int)


def threshold_curves(y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    rows = []
    for threshold in thresholds:
        pred = probabilities >= float(threshold)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "accuracy": float(accuracy),
                "f1": float(f1),
                "youden": float(sensitivity + specificity - 1.0),
            }
        )
    return pd.DataFrame(rows)


def clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_COLOR, length=3, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.5)


def add_panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.018, dy: float = 0.010) -> None:
    bbox = ax.get_position()
    fig.text(bbox.x0 + dx, bbox.y1 + dy, label, ha="left", va="bottom", fontweight="bold", fontsize=14)


def plot_threshold_selection(
    ax: plt.Axes,
    probabilities: np.ndarray,
    y_binary: np.ndarray,
    artifacts: dict,
) -> None:
    threshold = DEFAULT_DECISION_THRESHOLD
    _, validation_idx = validation_split_indices(probabilities, y_binary, artifacts)
    validation_prob = probabilities[validation_idx]
    validation_y = y_binary[validation_idx]
    grid = np.linspace(0.02, 0.98, 241)
    curve_df = threshold_curves(validation_y, validation_prob, grid)

    line_specs = [
        ("Sensitivity", "sensitivity", COLORS["hydro"], 1.45, "-"),
        ("Specificity", "specificity", COLORS["anthro"], 1.45, "-"),
        ("F1", "f1", COLORS["climate"], 1.20, "-"),
    ]
    for label, column, color, linewidth, linestyle in line_specs:
        ax.plot(
            curve_df["threshold"],
            curve_df[column],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            alpha=0.96,
        )

    threshold_row = curve_df.iloc[(curve_df["threshold"] - threshold).abs().argmin()]
    threshold_y = float(threshold_row["f1"])
    ax.axvline(threshold, color=COLORS["prob"], linestyle=(0, (3, 3)), linewidth=0.85)
    ax.scatter(
        [threshold],
        [threshold_y],
        s=26,
        color=COLORS["prob"],
        edgecolors=FIGURE_FACE,
        linewidths=0.7,
        zorder=5,
    )
    ax.text(
        threshold + 0.018,
        threshold_y - 0.04,
        f"p = {threshold:.3f}",
        ha="left",
        va="top",
        fontsize=8.0,
        color=TEXT_SECONDARY,
    )
    ax.text(
        0.04,
        0.08,
        "validation subset",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=TEXT_SECONDARY,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Validation metric")
    clean_axes(ax, grid_axis="both")
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.15), frameon=False, handlelength=1.35, handletextpad=0.45)


def plot_training_class_distribution(ax: plt.Axes, training_probabilities: np.ndarray, n_samples: int) -> None:
    probs = np.asarray(training_probabilities, dtype=float).reshape(-1)
    probs = probs[np.isfinite(probs)]
    labels = ["Lowest", "Low", "Middle", "High", "Highest"]
    class_breaks = np.asarray(jenkspy.jenks_breaks(probs, n_classes=len(labels)), dtype=float)
    class_ids = np.digitize(probs, class_breaks[1:-1], right=True)
    class_ids = np.clip(class_ids, 0, len(labels) - 1)
    counts = np.bincount(class_ids, minlength=len(labels))
    perc = counts / counts.sum() * 100.0

    bins = np.linspace(float(class_breaks[0]), float(class_breaks[-1]), 95)
    hist, edges = np.histogram(probs, bins=bins)
    hist = hist / max(hist.max(), 1)
    centers = (edges[:-1] + edges[1:]) / 2

    for c, h, left, right in zip(centers, hist, edges[:-1], edges[1:]):
        cls = int(np.clip(np.digitize([c], class_breaks[1:-1], right=True)[0], 0, len(labels) - 1))
        ax.fill_between([left, right], [0, 0], [h, h], color=CLASS_COLORS[cls], edgecolor="none", alpha=0.95)

    for boundary in class_breaks:
        ax.axvline(boundary, color=FIGURE_FACE, linewidth=0.9, zorder=3)
        ax.axvline(boundary, color=AXIS_COLOR, linewidth=0.35, zorder=4, alpha=0.45)

    y_label = 1.08
    for i, label in enumerate(labels):
        xmid = (class_breaks[i] + class_breaks[i + 1]) / 2
        ax.text(xmid, y_label, label, ha="center", va="bottom", fontsize=7.4)
        y_pct = -0.13 if i % 2 == 0 else -0.25
        ax.text(xmid, y_pct, f"{perc[i]:.1f}%", ha="center", va="top", fontsize=7.2, color=TEXT_SECONDARY)

    ax.text(0.01, 0.92, f"n = {n_samples:,}", transform=ax.transAxes, ha="left", va="top", fontsize=8.4, color=TEXT_SECONDARY)
    ax.set_xlim(float(class_breaks[0]), float(class_breaks[-1]))
    ax.set_ylim(-0.32, 1.18)
    ax.set_xlabel("Calibrated probability")
    ax.set_ylabel("Density")
    ax.set_yticks([0, 0.5, 1.0])
    clean_axes(ax, grid_axis="y")


def plot_metric_constellation(ax: plt.Axes, metrics_df: pd.DataFrame) -> None:
    metric_map = [
        ("Acc.", "acc"),
        ("Bacc.", "bacc"),
        ("F1", "f1"),
        ("AUC", "auc_roc"),
        ("AP", "ap"),
    ]
    y_positions = np.arange(len(metric_map))[::-1]
    rng = np.random.default_rng(2026)

    for y, (label, col) in zip(y_positions, metric_map):
        train_vals = metrics_df.loc[metrics_df["split"] == "train", col].astype(float).to_numpy()
        test_vals = metrics_df.loc[metrics_df["split"] == "test", col].astype(float).to_numpy()

        ax.scatter(
            train_vals,
            y + rng.normal(0.08, 0.018, size=len(train_vals)),
            s=7,
            color=COLORS["train"],
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        ax.scatter(
            test_vals,
            y + rng.normal(-0.08, 0.018, size=len(test_vals)),
            s=9,
            color=COLORS["test"],
            alpha=0.34,
            linewidth=0,
            zorder=2,
        )

        q_low, q_mid, q_high = np.quantile(test_vals, [0.025, 0.5, 0.975])
        ax.plot([q_low, q_high], [y, y], color=COLORS["test"], linewidth=1.8, solid_capstyle="round", zorder=3)
        ax.scatter([q_mid], [y], s=24, color=COLORS["test"], edgecolors=FIGURE_FACE, linewidth=0.7, zorder=4)
        ax.text(q_high + 0.004, y, f"{q_mid:.3f}", ha="left", va="center", fontsize=8.0, color=TEXT_SECONDARY)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([m[0] for m in metric_map])
    ax.set_xlim(0.86, 0.995)
    ax.set_xlabel("Validation metric value")
    ax.tick_params(axis="y", length=0)
    clean_axes(ax, grid_axis="x")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["test"], markersize=5, label="Test"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["train"], alpha=0.55, markersize=5, label="Train"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, borderaxespad=0.1, handletextpad=0.4)


def mean_curve(curves: list[tuple[np.ndarray, np.ndarray]], x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interpolated = []
    for x, y in curves:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]
        unique_x, unique_idx = np.unique(xs, return_index=True)
        unique_y = ys[unique_idx]
        interpolated.append(np.interp(x_grid, unique_x, unique_y))
    arr = np.vstack(interpolated)
    return np.nanmean(arr, axis=0), np.nanpercentile(arr, 5, axis=0), np.nanpercentile(arr, 95, axis=0)


def plot_roc_fan(ax_roc: plt.Axes, cv_cache: dict) -> None:
    runs = cv_cache["runs"]
    roc_curves = []
    for run in runs:
        fpr, tpr = run["roc"]
        roc_curves.append((fpr, tpr))
        ax_roc.plot(fpr, tpr, color=COLORS["test"], linewidth=0.55, alpha=0.10)

    xg = np.linspace(0, 1, 250)
    roc_mean, roc_low, roc_high = mean_curve(roc_curves, xg)

    ax_roc.fill_between(xg, roc_low, roc_high, color=COLORS["test"], alpha=0.10, linewidth=0)
    ax_roc.plot(xg, roc_mean, color=COLORS["test"], linewidth=1.8)
    ax_roc.plot([0, 1], [0, 1], color=AXIS_COLOR, linestyle=(0, (2, 2)), linewidth=0.65)
    ax_roc.text(0.97, 0.12, "ROC", ha="right", va="bottom", transform=ax_roc.transAxes)
    ax_roc.text(0.97, 0.04, "AUC 0.968", ha="right", va="bottom", transform=ax_roc.transAxes, fontsize=8.0, color=TEXT_SECONDARY)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    clean_axes(ax_roc, grid_axis="both")


def plot_reliability_curve(ax_cal: plt.Axes, cv_cache: dict, metrics_df: pd.DataFrame, threshold: float) -> None:
    runs = cv_cache["runs"]
    cal_x = []
    cal_y = []
    weights = []
    for run in runs:
        bins, frac_pos, mean_pred, counts = run["cal"]
        cal_x.append(np.asarray(mean_pred, dtype=float))
        cal_y.append(np.asarray(frac_pos, dtype=float))
        weights.append(np.asarray(counts, dtype=float))

    cal_x = np.vstack(cal_x)
    cal_y = np.vstack(cal_y)
    weights = np.vstack(weights)
    valid_weight = np.where(weights > 0, weights, np.nan)
    mean_pred = np.nanmean(np.where(np.isfinite(valid_weight), cal_x, np.nan), axis=0)
    frac_pos = np.nansum(cal_y * weights, axis=0) / np.maximum(np.nansum(weights, axis=0), 1)
    count_mean = np.nanmean(weights, axis=0)
    size = 12 + 70 * np.sqrt(count_mean / max(count_mean.max(), 1))

    ax_cal.plot([0, 1], [0, 1], color=AXIS_COLOR, linestyle=(0, (2, 2)), linewidth=0.75)
    ax_cal.plot(mean_pred, frac_pos, color=COLORS["prob"], linewidth=1.4)
    ax_cal.scatter(mean_pred, frac_pos, s=size, color=COLORS["prob"], alpha=0.62, edgecolors=FIGURE_FACE, linewidths=0.6)
    ax_cal.axvline(threshold, color=COLORS["prob"], linestyle=(0, (3, 3)), linewidth=0.75)
    brier = float(metrics_df.loc[metrics_df["split"] == "test", "brier"].mean())
    ax_cal.text(0.04, 0.94, "Reliability", transform=ax_cal.transAxes, ha="left", va="top")
    ax_cal.text(0.98, 0.05, f"p = {threshold:.3f}", transform=ax_cal.transAxes, ha="right", va="bottom", fontsize=8.0, color=TEXT_SECONDARY)
    ax_cal.text(0.98, 0.13, f"Brier = {brier:.3f}", transform=ax_cal.transAxes, ha="right", va="bottom", fontsize=8.0, color=TEXT_SECONDARY)
    ax_cal.set_xlabel("Predicted probability")
    ax_cal.set_ylabel("Observed frequency")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    clean_axes(ax_cal, grid_axis="both")


def build_figure(output_path: Path) -> Path:
    apply_style()
    model_bundle = load_pickle(MODEL_PATH)
    artifacts = load_pickle(ARTIFACTS_PATH)
    training_df = pd.read_csv(TRAINING_DATA_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)
    cv_cache = load_gzip_pickle(CV_CACHE_PATH)

    transform_metadata = artifacts["transform_metadata"]

    train_raw_scores = np.asarray(model_bundle["gwr_results"].predy, dtype=float).reshape(-1)
    train_probabilities = sigmoid_from_metadata(train_raw_scores, transform_metadata)
    train_labels = np.asarray(model_bundle["gwr_results"].y, dtype=float).reshape(-1)
    y_binary = binary_labels_from_cutoff(train_labels, float(artifacts["label_cutoff"]))
    if len(training_df) != len(train_probabilities):
        raise ValueError(
            f"Training data length ({len(training_df):,}) does not match GWR training predictions "
            f"({len(train_probabilities):,})."
        )

    fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4), facecolor=FIGURE_FACE)
    outer = fig.add_gridspec(
        2,
        height_ratios=[1.0, 1.0],
        left=0.068,
        right=0.985,
        bottom=0.090,
        top=0.955,
        hspace=0.44,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[1.30, 1.70], wspace=0.20)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.38)

    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])
    ax_e = fig.add_subplot(bottom[0, 2])

    plot_threshold_selection(ax_a, train_probabilities, y_binary, artifacts)
    plot_training_class_distribution(ax_b, train_probabilities, n_samples=len(training_df))
    plot_metric_constellation(ax_c, metrics_df)
    plot_roc_fan(ax_d, cv_cache)
    plot_reliability_curve(ax_e, cv_cache, metrics_df, DEFAULT_DECISION_THRESHOLD)

    add_panel_label(fig, ax_a, "a")
    add_panel_label(fig, ax_b, "b")
    add_panel_label(fig, ax_c, "c")
    add_panel_label(fig, ax_d, "d")
    add_panel_label(fig, ax_e, "e")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the national GWR training group figure.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output SVG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_figure(args.output)
    print(f"[OK] GWR training group figure -> {path}")


if __name__ == "__main__":
    main()
