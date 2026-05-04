"""
STAGE 8: Generate publication plots.

Reads stages 03-07 output, produces matplotlib figures for the paper.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils_pipeline import RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def plot_layer_sweep():
    with open(RESULTS_DIR / "03_layer_sweep.json") as f:
        data = json.load(f)
    fig, ax = plt.subplots(figsize=(10, 6))
    for task_name, info in data.items():
        per_layer = info["per_layer"]
        layers = sorted([int(l) for l in per_layer])
        accs = [per_layer[str(l)] for l in layers]
        ax.plot(layers, accs, marker="o", label=task_name)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Zero-shot patch accuracy")
    ax.set_title("Per-task layer sweep (residual-stream FV)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_layer_sweep.pdf")
    plt.savefig(PLOTS_DIR / "03_layer_sweep.png", dpi=150)
    plt.close()
    print("  Saved 03_layer_sweep")


def plot_cie_heatmaps():
    with open(RESULTS_DIR / "04_cie_heatmaps.json") as f:
        data = json.load(f)
    n_tasks = len(data)
    cols = 4
    rows = (n_tasks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    for ax, (task_name, info) in zip(axes, data.items()):
        m = np.array(info["cie_matrix"])
        vmax = max(abs(m.min()), abs(m.max()))
        im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(task_name, fontsize=9)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        plt.colorbar(im, ax=ax)
    for ax in axes[len(data):]:
        ax.axis("off")
    plt.suptitle("Causal Indirect Effect: per-head heatmaps")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_cie_heatmaps.pdf")
    plt.savefig(PLOTS_DIR / "04_cie_heatmaps.png", dpi=150)
    plt.close()
    print("  Saved 04_cie_heatmaps")


def plot_fv_validation():
    with open(RESULTS_DIR / "05_cie_fv_validation.json") as f:
        data = json.load(f)
    tasks = list(data.keys())
    cie_acc = [data[t]["results"]["cie_fv"]["acc"] for t in tasks]
    resid_acc = [data[t]["results"]["residual_fv"]["acc"] for t in tasks]
    rand_acc = [data[t]["results"]["random"]["acc"] for t in tasks]
    x = np.arange(len(tasks))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, cie_acc, w, label="CIE-FV (top-K heads)")
    ax.bar(x,     resid_acc, w, label="Residual-stream FV")
    ax.bar(x + w, rand_acc, w, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Zero-shot patch accuracy")
    ax.set_title("FV extraction method comparison")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_fv_validation.pdf")
    plt.savefig(PLOTS_DIR / "05_fv_validation.png", dpi=150)
    plt.close()
    print("  Saved 05_fv_validation")


def plot_composition_analysis():
    with open(RESULTS_DIR / "07_composition_analysis.json") as f:
        data = json.load(f)
    comps = list(data.keys())
    labels = ["baseline", "v_h", "v_f", "v_g", "v_f+v_g", "v_random"]
    correct_rates = {l: [data[c]["causal"][l]["correct_comp_rate"] for c in comps] for l in labels}
    f_only_rates = {l: [data[c]["causal"][l]["f_only_rate"] for c in comps] for l in labels}
    g_only_rates = {l: [data[c]["causal"][l]["g_only_rate"] for c in comps] for l in labels}
    x = np.arange(len(comps))
    w = 0.13
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, l in enumerate(labels):
        ax.bar(x + (i - len(labels)/2)*w, correct_rates[l], w, label=l)
    ax.set_xticks(x)
    ax.set_xticklabels(comps, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("CORRECT_COMP rate")
    ax.set_title("Causal validation: which patch produces the composition?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_composition_correct.pdf")
    plt.savefig(PLOTS_DIR / "07_composition_correct.png", dpi=150)
    plt.close()
    print("  Saved 07_composition_correct")

    # Cosine geometry plot
    fig, ax = plt.subplots(figsize=(12, 5))
    cos_f = [data[c]["geometry"]["cos_h_f_corrected"] for c in comps]
    cos_g = [data[c]["geometry"]["cos_h_g_corrected"] for c in comps]
    cos_sum = [data[c]["geometry"]["cos_h_sum_corrected"] for c in comps]
    ax.bar(x - w, cos_f, w, label="cos(v_h, v_f) corrected")
    ax.bar(x,     cos_g, w, label="cos(v_h, v_g) corrected")
    ax.bar(x + w, cos_sum, w, label="cos(v_h, v_f+v_g) corrected")
    ax.set_xticks(x)
    ax.set_xticklabels(comps, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Format-corrected cosine")
    ax.set_title("Geometric structure of composition FVs")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_composition_cosine.pdf")
    plt.savefig(PLOTS_DIR / "07_composition_cosine.png", dpi=150)
    plt.close()
    print("  Saved 07_composition_cosine")


def main():
    print("Generating plots...")
    plot_layer_sweep()
    plot_cie_heatmaps()
    plot_fv_validation()
    plot_composition_analysis()
    print(f"\nAll plots in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
