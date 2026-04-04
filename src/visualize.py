"""
Visualization module for Knowledge Tracing benchmark paper.

Generates publication-quality figures in APA-compatible style.
Sentence capitalization, no em dashes.

Figures:
  1. AUC-ROC grouped bar chart (primary result)
  2. PR-AUC grouped bar chart
  3. RMSE grouped bar chart
  4. Training time comparison
  5. Per-dataset AUC stability (fold variance within each dataset)
  6. Multi-metric radar/bar (macro-averaged across datasets)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_COLORS = {
    "BKT": "#4477AA",
    "PFA": "#66CCEE",
    "DKT": "#228833",
    "SAKT": "#CCBB44",
    "TransformerKT": "#EE6677",
}

MODEL_ORDER = ["BKT", "PFA", "DKT", "SAKT", "TransformerKT"]

DATASET_LABELS = {
    "assist2009": "ASSISTments 2009",
    "assist2015": "ASSISTments 2015",
    "assist2017": "ASSISTments 2017",
    "statics2011": "Statics 2011",
    "algebra2005": "Algebra 2005",
}


def load_results():
    with open(os.path.join(OUTPUT_DIR, "benchmark_results.json"), "r") as f:
        return json.load(f)


def _grouped_bar(results, metric_key, ylabel, title, filename, ylim=None,
                 lower_better=False):
    """Generic grouped bar chart across datasets and models."""
    datasets = list(DATASET_LABELS.keys())
    models = MODEL_ORDER

    fig, ax = plt.subplots(figsize=(10, 5))
    n_datasets = len(datasets)
    n_models = len(models)
    bar_width = 0.15
    x = np.arange(n_datasets)

    for i, model in enumerate(models):
        means, stds = [], []
        for ds in datasets:
            if ds in results and model in results[ds]:
                m = results[ds][model]
                means.append(m.get(f"{metric_key}_mean", 0))
                stds.append(m.get(f"{metric_key}_std", 0))
            else:
                means.append(0)
                stds.append(0)

        ax.bar(
            x + i * bar_width, means, bar_width,
            yerr=stds, label=model, color=MODEL_COLORS[model],
            edgecolor="white", linewidth=0.5,
            capsize=3, error_kw={"linewidth": 0.8},
        )

    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], rotation=15, ha="right")
    if lower_better:
        ax.legend(loc="upper right", framealpha=0.9)
    else:
        ax.legend(loc="lower right", framealpha=0.9)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.png"))
    plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.pdf"))
    plt.close()
    print(f"  Saved {filename}")


def fig1_auc(results):
    _grouped_bar(
        results, "auc", "AUC-ROC",
        "Figure 1. Area under the ROC curve by model and dataset",
        "fig1_auc_comparison", ylim=(0.45, 0.90),
    )


def fig2_prauc(results):
    _grouped_bar(
        results, "pr_auc", "PR-AUC",
        "Figure 2. Area under the precision-recall curve by model and dataset",
        "fig2_prauc_comparison", ylim=(0.45, 0.95),
    )


def fig3_rmse(results):
    _grouped_bar(
        results, "rmse", "RMSE (lower is better)",
        "Figure 3. Root mean square error by model and dataset",
        "fig3_rmse_comparison", lower_better=True,
    )


def fig4_training_time(results):
    """Figure 4: Training time comparison (per-dataset bars)."""
    datasets = list(DATASET_LABELS.keys())
    models = MODEL_ORDER

    fig, ax = plt.subplots(figsize=(10, 5))
    n_datasets = len(datasets)
    bar_width = 0.15
    x = np.arange(n_datasets)

    for i, model in enumerate(models):
        means = []
        for ds in datasets:
            if ds in results and model in results[ds]:
                means.append(results[ds][model].get("train_time_seconds_mean", 0))
            else:
                means.append(0)
        ax.bar(
            x + i * bar_width, means, bar_width,
            label=model, color=MODEL_COLORS[model],
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Training time per fold (seconds)")
    ax.set_title("Figure 4. Average training time per fold by model and dataset")
    ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], rotation=15, ha="right")
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_training_time.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_training_time.pdf"))
    plt.close()
    print("  Saved fig4_training_time")


def fig5_per_dataset_stability(results):
    """Figure 5: Per-dataset AUC fold variance (subplots per dataset)."""
    datasets = list(DATASET_LABELS.keys())
    models = MODEL_ORDER

    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)

    for ax, ds in zip(axes, datasets):
        fold_aucs = []
        labels = []
        colors = []
        for model in models:
            if ds in results and model in results[ds]:
                details = results[ds][model].get("fold_details", [])
                aucs = [f["auc"] for f in details if f.get("auc") is not None]
                if aucs:
                    fold_aucs.append(aucs)
                    labels.append(model)
                    colors.append(MODEL_COLORS[model])

        if fold_aucs:
            bp = ax.boxplot(
                fold_aucs, tick_labels=labels, patch_artist=True,
                widths=0.5, showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "black", "markersize": 4},
                medianprops={"color": "black", "linewidth": 1.2},
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

        ax.set_title(DATASET_LABELS[ds], fontsize=10)
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("AUC-ROC")
    fig.suptitle("Figure 5. Distribution of AUC-ROC scores across folds for each dataset",
                 fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_stability.png"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_stability.pdf"), bbox_inches="tight")
    plt.close()
    print("  Saved fig5_stability")


def fig6_macro_average(results):
    """Figure 6: Macro-averaged metrics across all datasets."""
    metrics_keys = ["auc", "pr_auc", "accuracy", "f1"]
    metric_labels = ["AUC-ROC", "PR-AUC", "Accuracy", "F1"]
    models = MODEL_ORDER

    avg_metrics = {}
    for model in models:
        model_vals = {m: [] for m in metrics_keys}
        for ds in results:
            if model in results[ds]:
                for m in metrics_keys:
                    val = results[ds][model].get(f"{m}_mean")
                    if val is not None:
                        model_vals[m].append(val)
        avg_metrics[model] = {m: np.mean(v) if v else 0 for m, v in model_vals.items()}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics_keys))
    bar_width = 0.15

    for i, model in enumerate(models):
        values = [avg_metrics[model][m] for m in metrics_keys]
        ax.bar(x + i * bar_width, values, bar_width,
               label=model, color=MODEL_COLORS[model],
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Macro-averaged score across datasets")
    ax.set_title("Figure 6. Macro-averaged performance across all five datasets")
    ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0.35, 0.90)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_macro_average.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_macro_average.pdf"))
    plt.close()
    print("  Saved fig6_macro_average")


def generate_all_figures():
    print("Loading results...")
    results = load_results()
    print("Generating figures...")
    fig1_auc(results)
    fig2_prauc(results)
    fig3_rmse(results)
    fig4_training_time(results)
    fig5_per_dataset_stability(results)
    fig6_macro_average(results)
    print("\nAll figures generated.")


if __name__ == "__main__":
    generate_all_figures()
