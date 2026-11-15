"""Regenerate README figures from the versioned seed-13 CSV."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
COLORS = ["#425b76", "#24968b", "#df9a39"]


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7fafc", "axes.facecolor": "#f7fafc"})
    with (ROOT / "results" / "seed13" / "summary_test.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = ["A · Base", "B · Random", "C · Moderate"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")
    panels = [("Test accuracy", [100 * float(row["accuracy"]) for row in rows], "%", 80),
              ("Answer format compliance", [100 * float(row["strict_format_rate"]) for row in rows], "%", 118),
              ("Scoring + training", [60 * (float(row["score_minutes"]) + float(row["train_minutes"])) for row in rows], "s", 140)]
    for axis, (title, values, unit, ceiling) in zip(axes, panels):
        bars = axis.bar(labels, values, color=COLORS, width=0.58)
        axis.bar_label(bars, labels=[f"{value:.2f}{unit}" for value in values], padding=7, fontsize=11)
        axis.set(title=title, ylim=(0, ceiling), ylabel="Percent" if unit == "%" else "Seconds")
        axis.spines[["left", "bottom"]].set_color("#ccd6df")
        axis.yaxis.grid(True, alpha=0.15)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", length=0)
    fig.suptitle("GSM8K · 1,319 test questions · Seed 13 · ~0.5M training tokens", fontsize=15, weight="bold")
    fig.savefig(ASSETS / "results.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(13.5, 4.7), layout="constrained")
    axis.set(xlim=(0, 13.5), ylim=(0, 4.7))
    axis.axis("off")
    axis.text(0.2, 4.15, "Same model. Matched budget. Two selection strategies.", fontsize=19, weight="bold", color="#21394f")
    axis.text(0.2, 3.67, "Qwen2.5-1.5B-Instruct  /  GSM8K  /  BF16 LoRA", color="#61788b", fontsize=12)

    def box(x, y, width, height, title, detail, color):
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.12,rounding_size=0.14",
                                     facecolor=color, edgecolor="none"))
        axis.text(x + width/2, y + height*0.65, title, ha="center", va="center", color="white", weight="bold", fontsize=12)
        axis.text(x + width/2, y + height*0.29, detail, ha="center", va="center", color="white", fontsize=10)

    def arrow(start, end):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color="#8a9dad", linewidth=1.8))

    box(0.3, 1.28, 2.25, 1.28, "Prepare", "6,973 candidates", COLORS[0])
    box(3.25, 1.28, 2.25, 1.28, "Score + bucket", "Answer NLL / length", COLORS[0])
    box(6.25, 2.12, 2.6, 1.0, "B · Random", "Length-matched sampling", COLORS[1])
    box(6.25, 0.45, 2.6, 1.0, "C · Moderate", "Middle 50% of losses", COLORS[2])
    box(9.75, 1.28, 3.0, 1.28, "Train + evaluate", "~0.5M tokens / 80 updates", COLORS[0])
    arrow((2.7, 1.92), (3.07, 1.92))
    arrow((5.65, 2.15), (6.08, 2.59))
    arrow((5.65, 1.68), (6.08, 0.98))
    arrow((9.0, 2.59), (9.58, 2.13))
    arrow((9.0, 0.98), (9.58, 1.68))
    axis.text(0.3, 0.15, "A · Base is evaluated with the same prompt, decoding settings, and held-out questions.", fontsize=11, color="#61788b")
    fig.savefig(ASSETS / "pipeline.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
