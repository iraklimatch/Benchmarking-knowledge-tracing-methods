"""
Extract results from benchmark_results.json and print formatted tables
for paper.md and generate_docx.js updates.
"""

import json
import os
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

DATASETS = ["assist2009", "assist2015", "assist2017", "statics2011", "algebra2005"]
DATASET_LABELS = {
    "assist2009": "ASSISTments 2009",
    "assist2015": "ASSISTments 2015",
    "assist2017": "ASSISTments 2017",
    "statics2011": "Statics 2011",
    "algebra2005": "Algebra 2005",
}
MODELS = ["BKT", "PFA", "DKT", "SAKT", "TransformerKT"]


def load_results():
    with open(os.path.join(OUTPUT_DIR, "benchmark_results.json")) as f:
        return json.load(f)


def fmt(val, digits=3):
    """Format a float as .XXX"""
    if val is None:
        return "N/A"
    return f".{val:.3f}"[1:]


def main():
    results = load_results()

    # Verify n_predictions match across all models per dataset per fold
    print("=" * 60)
    print("VERIFICATION: n_predictions consistency")
    print("=" * 60)
    all_ok = True
    for ds in DATASETS:
        if ds not in results:
            print(f"  {ds}: MISSING")
            continue
        for fold_idx in range(5):
            counts = {}
            for model in MODELS:
                if model in results[ds]:
                    details = results[ds][model].get("fold_details", [])
                    if fold_idx < len(details):
                        counts[model] = details[fold_idx].get("n_predictions", 0)
            unique_counts = set(counts.values())
            if len(unique_counts) > 1:
                print(f"  {ds} fold {fold_idx}: MISMATCH {counts}")
                all_ok = False
        print(f"  {ds}: OK (all folds match)" if all_ok else "")

    # ================================================================
    # Table 2: AUC-ROC by model and dataset
    # ================================================================
    print("\n" + "=" * 60)
    print("TABLE 2: AUC-ROC (paper.md format)")
    print("=" * 60)
    macro_aucs = {m: [] for m in MODELS}

    for ds in DATASETS:
        row = f"| {DATASET_LABELS[ds]} |"
        best_model = max(MODELS, key=lambda m: results[ds][m].get("auc_mean", 0))
        for model in MODELS:
            r = results[ds][model]
            auc = r.get("auc_mean")
            std = r.get("auc_std")
            if auc is not None:
                macro_aucs[model].append(auc)
            cell = f" {fmt(auc)} ({fmt(std)}) "
            if model == best_model:
                cell = f" **{fmt(auc)} ({fmt(std)})** "
            row += cell + "|"
        print(row)

    # Macro average row
    row = "| **Macro avg.** |"
    for model in MODELS:
        avg = np.mean(macro_aucs[model]) if macro_aucs[model] else 0
        row += f" **{fmt(avg)}** |"
    print(row)

    # ================================================================
    # Table 3: Multi-metric (paper.md format)
    # ================================================================
    print("\n" + "=" * 60)
    print("TABLE 3: Multi-metric (paper.md format)")
    print("=" * 60)
    for ds in DATASETS:
        for i, model in enumerate(MODELS):
            r = results[ds][model]
            ds_label = DATASET_LABELS[ds] if i == 0 else ""
            auc = fmt(r.get("auc_mean"))
            pr = fmt(r.get("pr_auc_mean"))
            acc = fmt(r.get("accuracy_mean"))
            f1 = fmt(r.get("f1_mean"))
            rmse = fmt(r.get("rmse_mean"))
            print(f"| {ds_label} | {model} | {auc} | {pr} | {acc} | {f1} | {rmse} |")

    # ================================================================
    # Table 4: Training time
    # ================================================================
    print("\n" + "=" * 60)
    print("TABLE 4: Training time (paper.md format)")
    print("=" * 60)
    for model in MODELS:
        row = f"| {model} |"
        times = []
        for ds in DATASETS:
            r = results[ds][model]
            t = r.get("train_time_seconds_mean", 0)
            times.append(t)
            row += f" {t:.1f} |"
        row += f" {np.mean(times):.1f} |"
        print(row)

    # ================================================================
    # Table 5: Confidence intervals
    # ================================================================
    print("\n" + "=" * 60)
    print("TABLE 5: AUC-ROC CIs (paper.md format)")
    print("=" * 60)
    for ds in DATASETS:
        row = f"| {DATASET_LABELS[ds]} |"
        for model in MODELS:
            r = results[ds][model]
            ci = r.get("auc_ci_mean")
            if ci:
                row += f" [{fmt(ci[0])}, {fmt(ci[1])}] |"
            else:
                row += " N/A |"
        print(row)

    # ================================================================
    # generate_docx.js table data
    # ================================================================
    print("\n" + "=" * 60)
    print("GENERATE_DOCX.JS: Table 2 data (copy-paste)")
    print("=" * 60)
    for ds in DATASETS:
        row_items = [f'"{DATASET_LABELS[ds]}"']
        for model in MODELS:
            r = results[ds][model]
            auc = fmt(r.get("auc_mean"))
            std = fmt(r.get("auc_std"))
            row_items.append(f'"{auc} ({std})"')
        print(f'          [{", ".join(row_items)}],')
    # Macro
    macro_items = ['"Macro avg."']
    for model in MODELS:
        avg = np.mean(macro_aucs[model]) if macro_aucs[model] else 0
        macro_items.append(f'"{fmt(avg)}"')
    print(f'          [{", ".join(macro_items)}],')

    print("\n" + "=" * 60)
    print("GENERATE_DOCX.JS: Table 3 data (copy-paste)")
    print("=" * 60)
    for ds in DATASETS:
        for i, model in enumerate(MODELS):
            r = results[ds][model]
            ds_label = DATASET_LABELS[ds] if i == 0 else ""
            auc = fmt(r.get("auc_mean"))
            pr = fmt(r.get("pr_auc_mean"))
            acc = fmt(r.get("accuracy_mean"))
            f1 = fmt(r.get("f1_mean"))
            rmse = fmt(r.get("rmse_mean"))
            print(f'          ["{ds_label}", "{model}", "{auc}", "{pr}", "{acc}", "{f1}", "{rmse}"],')

    print("\n" + "=" * 60)
    print("GENERATE_DOCX.JS: Table 4 data (copy-paste)")
    print("=" * 60)
    for model in MODELS:
        items = [f'"{model}"']
        times = []
        for ds in DATASETS:
            r = results[ds][model]
            t = r.get("train_time_seconds_mean", 0)
            times.append(t)
            items.append(f'"{t:.1f}"')
        items.append(f'"{np.mean(times):.1f}"')
        print(f'          [{", ".join(items)}],')

    print("\n" + "=" * 60)
    print("GENERATE_DOCX.JS: Table 5 data (copy-paste)")
    print("=" * 60)
    for ds in DATASETS:
        items = [f'"{DATASET_LABELS[ds]}"']
        for model in MODELS:
            r = results[ds][model]
            ci = r.get("auc_ci_mean")
            if ci:
                items.append(f'"[{fmt(ci[0])}, {fmt(ci[1])}]"')
            else:
                items.append('"N/A"')
        print(f'          [{", ".join(items)}],')

    # ================================================================
    # Key numbers for prose updates
    # ================================================================
    print("\n" + "=" * 60)
    print("KEY NUMBERS FOR PROSE")
    print("=" * 60)
    for model in MODELS:
        avg = np.mean(macro_aucs[model]) if macro_aucs[model] else 0
        print(f"  {model} macro AUC: {avg:.3f}")

    dkt_macro = np.mean(macro_aucs["DKT"])
    bkt_macro = np.mean(macro_aucs["BKT"])
    pfa_macro = np.mean(macro_aucs["PFA"])
    tkt_macro = np.mean(macro_aucs["TransformerKT"])
    sakt_macro = np.mean(macro_aucs["SAKT"])
    print(f"\n  DKT advantage over BKT/PFA: {(dkt_macro - bkt_macro)*100:.1f} / {(dkt_macro - pfa_macro)*100:.1f} pp")
    print(f"  DKT advantage over TransformerKT: {(dkt_macro - tkt_macro)*100:.1f} pp")
    print(f"  DKT advantage over SAKT: {(dkt_macro - sakt_macro)*100:.1f} pp")
    print(f"  TransformerKT/DKT time ratio: ", end="")

    tkt_times = []
    dkt_times = []
    for ds in DATASETS:
        tkt_times.append(results[ds]["TransformerKT"].get("train_time_seconds_mean", 0))
        dkt_times.append(results[ds]["DKT"].get("train_time_seconds_mean", 0))
    print(f"{np.mean(tkt_times) / np.mean(dkt_times):.1f}x")


if __name__ == "__main__":
    main()
