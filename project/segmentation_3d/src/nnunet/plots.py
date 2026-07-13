import json
import logging
import matplotlib
matplotlib.use("Agg")  # headless / no display on the training host
import matplotlib.pyplot as plt
import numpy as np
import os

logger = logging.getLogger(__name__)

# colours
POINT_COLOUR = "#2c5f8a"
BAR_COLOUR = "#a9c6e0"
MEAN_COLOUR = "#0b2a45"
SUMMARY_BAR_COLOUR = "#f0c27b"
SUMMARY_MEAN_COLOUR = "#8a5a12"


def load_json(json_file_path: str) -> dict:
    with open(json_file_path) as f:
        return json.load(f)


def get_label_names(summary, dataset_json_path):
    """Map each integer label key to a display name, from dataset_json_path file."""

    labels = sorted(
        {k for case in summary["metric_per_case"] for k in case["metrics"]},
        key=lambda x: int(x),
    )

    label_names = {}
    if dataset_json_path and os.path.isfile(dataset_json_path):
        with open(dataset_json_path) as f:
            dataset = json.load(f)
        for name, value in dataset.get("labels", {}).items():
            if isinstance(value, (list, tuple)):
                continue
            label_names[str(value)] = name

    return {k: label_names.get(str(k), f"Label {int(k):02d}") for k in labels}


def get_per_structure_values(summary, field):
    """{label_key: [per-case value of ``field``]} over all cases."""

    per_structure = {}
    for case in summary["metric_per_case"]:
        for label, metrics in case["metrics"].items():
            per_structure.setdefault(label, []).append(metrics[field])
    return per_structure


def get_per_case_summary(summary, field):
    """Per-case mean of ``field`` across that case's structures (the top 'overall' row)."""

    return [
        float(np.mean([m[field] for m in case["metrics"].values()]))
        for case in summary["metric_per_case"]
    ]


def simplify_count(value: float) -> str:
    """Converts millions to M, converts thousands to K for ez reading"""

    for threshold, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(value) >= threshold:
            return f"{value / threshold:.1f}{suffix}"
    return f"{value:.0f}"


def float2dec(mean, std):
    return f"{mean:.2f} ± {std:.2f}"


def float2simplified_dec(mean, std):
    return f"{simplify_count(mean)} ± {simplify_count(std)}"


def _draw_dot_bar_figure(rows, output_path, xlabel, title, value_label, subtitle="", logx=False, xlim=None):
    """Render figure with scatter + mean with bar.

    rows: list of (name, values, is_summary_row) ordered top -> bottom.
    """

    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, 1.6 + 0.42 * n))
    rng = np.random.default_rng(0)

    all_values = np.concatenate([np.asarray(v, float) for _, v, _ in rows])
    if xlim is None:
        if logx:
            lo = max(all_values.min() * 0.7, 1.0)
            xlim = (lo, all_values.max() * 1.4)
        else:
            xlim = (0.0, 1.0)
    bar_left = xlim[0] if logx else 0.0

    yticks, yticklabels = [], []
    for i, (name, values, is_summary) in enumerate(rows):
        y = n - 1 - i  # first row at the top
        vals = np.asarray(values, float)
        mean, std = float(vals.mean()), float(vals.std())

        bar_color = SUMMARY_BAR_COLOUR if is_summary else BAR_COLOUR
        mean_color = SUMMARY_MEAN_COLOUR if is_summary else MEAN_COLOUR

        ax.barh(y, mean - bar_left, left=bar_left, height=0.62, color=bar_color,
                alpha=0.55, zorder=1)
        jitter = (rng.random(len(vals)) - 0.5) * 0.34
        ax.scatter(vals, y + jitter, s=24, color=POINT_COLOUR, alpha=0.8,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.plot([mean, mean], [y - 0.33, y + 0.33], color=mean_color, lw=2.6, zorder=4)
        ax.annotate(value_label(mean, std), xy=(1.015, y),
                    xycoords=("axes fraction", "data"), va="center", ha="left",
                    fontsize=8.5, color="#333333")

        yticks.append(y)
        yticklabels.append(name)
        if is_summary:  # divider under the overall row
            ax.axhline(y - 0.5, color="#bbbbbb", lw=0.8, ls="--", zorder=0)

    if logx:
        ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=9, color="#555555")
    ax.grid(axis="x", color="#e6e6e6", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.2, right=0.86, top=0.9, bottom=0.1)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_validation_summary(summary_json_path, output_dir, dataset_json_path=None):
    """Generate metrics (Dice, num pixels) figures from nnunet summary json file (summary_json_path).

    Expects the post-processed cross-validation summary.json; reads label names from
    dataset_json_path.

    Returns list of written figure paths for mlflow logging.
    """

    os.makedirs(output_dir, exist_ok=True)
    summary = load_json(summary_json_path)
    names = get_label_names(summary, dataset_json_path=dataset_json_path)

    dice_per_structure = get_per_structure_values(summary, "Dice")
    size_per_structure = get_per_structure_values(summary, "n_ref")

    # shared structure alphabetical order (both figs)
    order = sorted(dice_per_structure, key=lambda k: names[k].lower())

    subtitle = f"{len(summary["metric_per_case"])} validation cases"

    dice_rows = [("Foreground mean", get_per_case_summary(summary, "Dice"), True)]
    dice_rows += [(names[k], dice_per_structure[k], False) for k in order]
    dice_path = _draw_dot_bar_figure(
        dice_rows, os.path.join(output_dir, "cross_validation_dice.png"),
        xlabel="Dice", title="Validation Dice per structure", subtitle=subtitle,
        value_label=float2dec, logx=False, xlim=(0.0, 1.0),
    )

    size_rows = [("All structures (mean)", get_per_case_summary(summary, "n_ref"), True)]
    size_rows += [(names[k], size_per_structure[k], False) for k in order]
    size_path = _draw_dot_bar_figure(
        size_rows, os.path.join(output_dir, "cross_validation_structure_size.png"),
        xlabel="Number of ground truth pixels",
        title="Validation structure size per structure", subtitle=subtitle,
        value_label=float2simplified_dec, logx=True,
    )

    return [dice_path, size_path]
