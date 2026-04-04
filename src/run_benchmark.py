"""
Unified benchmark runner for Knowledge Tracing experiments (v2).

Key design decisions:
  - Classical models (BKT, PFA) receive full unchunked interaction histories
  - DL models (DKT, SAKT, TransformerKT) receive chunked sequences
  - All models evaluated on the identical common prediction mask:
    skip first encounter of each (user, skill) pair
  - Real elapsed training time measured for every model
  - Full seed control for reproducibility
  - PR-AUC metric and bootstrap 95% CIs
"""

import os
import sys
import json
import time
import traceback
import warnings
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import prepare_dataset, DATASETS, DATA_DIR, compute_chunk_start_positions
from models_v2 import (
    set_seed, evaluate_predictions, bootstrap_ci,
    compute_common_prediction_mask,
    BKTModel, PFAModel, DKTModel, SAKTModel, TransformerKTModel,
    KTDataset,
)
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from torch.utils.data import DataLoader

os.chdir(DATA_DIR)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_EPOCHS = {
    "assist2009": 15,
    "assist2015": 10,
    "assist2017": 8,
    "statics2011": 15,
    "algebra2005": 10,
}

N_FOLDS = 5
SEED = 42

CLASSICAL_MODELS = {"BKT", "PFA"}
DL_MODELS = {"DKT", "SAKT", "TransformerKT"}
MODEL_ORDER = ["BKT", "PFA", "DKT", "SAKT", "TransformerKT"]


def predict_dl_on_interactions(model, test_seqs, test_user_ids, test_skills,
                                test_corrects, common_mask):
    """Map DL sequence predictions back to interaction-level and apply common mask.

    Strategy: iterate through test sequences in the same order they were created
    (by user, then by chunk). For each sequence, the DL model predicts positions
    1..length-1 (next-step). We track which global interaction index each
    prediction corresponds to, then filter by the common mask.
    """
    # Step 1: Get per-sequence predictions
    dataset = KTDataset(test_seqs)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.net.eval()

    seq_predictions = []  # list of (preds_array, targets_array) per sequence
    with torch.no_grad():
        batch_seq_idx = 0
        for batch in loader:
            skills_b = batch["skills"].to(model.device)
            corrects_b = batch["corrects"].to(model.device)
            targets_b = batch["targets"].to(model.device)
            mask_b = batch["mask"].to(model.device)

            pred = model.net(skills_b, corrects_b, mask_b)
            target_shifted = targets_b[:, 1:]
            mask_shifted = mask_b[:, 1:]

            for b in range(pred.shape[0]):
                length = int(mask_shifted[b].sum().item())
                p = pred[b, :length].cpu().numpy()
                t = target_shifted[b, :length].cpu().numpy()
                seq_predictions.append((p, t))

    # Step 2: Map predictions to global interaction indices
    # Sequences are created per-user in timestamp order, chunked.
    # pred[i] in a sequence of length L corresponds to the interaction
    # at position (chunk_offset + i + 1) within that user's history.
    pred_by_global_idx = {}
    user_chunk_offsets = {}

    for seq_idx, seq in enumerate(test_seqs):
        uid = seq["user_id"]
        length = seq["length"]
        if uid not in user_chunk_offsets:
            user_chunk_offsets[uid] = 0
        offset = user_chunk_offsets[uid]

        if seq_idx < len(seq_predictions):
            preds, targets = seq_predictions[seq_idx]
            for i in range(len(preds)):
                global_pos = offset + i + 1  # +1 because next-step prediction
                pred_by_global_idx[(uid, global_pos)] = preds[i]

        user_chunk_offsets[uid] = offset + length

    # Step 3: Iterate test interactions, collect predictions at masked positions
    y_true_out = []
    y_pred_out = []
    uid_out = []
    user_pos = {}

    for i in range(len(test_user_ids)):
        uid = test_user_ids[i]
        if uid not in user_pos:
            user_pos[uid] = 0
        pos = user_pos[uid]
        user_pos[uid] = pos + 1

        if common_mask[i]:
            key = (uid, pos)
            if key in pred_by_global_idx:
                y_true_out.append(test_corrects[i])
                y_pred_out.append(pred_by_global_idx[key])
                uid_out.append(uid)

    return np.array(y_true_out), np.array(y_pred_out), np.array(uid_out)


def make_model(model_name, dataset_name):
    epochs = DATASET_EPOCHS.get(dataset_name, 10)
    if model_name == "BKT":
        return BKTModel(n_iter=50, tol=1e-4)
    elif model_name == "PFA":
        return PFAModel()
    elif model_name == "DKT":
        return DKTModel(hidden_dim=100, lr=0.001, epochs=epochs, batch_size=64)
    elif model_name == "SAKT":
        return SAKTModel(embed_dim=64, lr=0.001, epochs=epochs, batch_size=64)
    elif model_name == "TransformerKT":
        return TransformerKTModel(
            embed_dim=64, n_heads=4, n_layers=2,
            lr=0.001, epochs=epochs, batch_size=64,
        )


def run_single_fold(model_name, dataset_name, fold_data, fold_idx):
    """Run a single model on a single fold."""
    train_seqs = fold_data["train_seqs"]
    test_seqs = fold_data["test_seqs"]
    train_uids, train_skills, train_corrects, train_items = fold_data["train_interactions"]
    test_uids, test_skills, test_corrects, test_items = fold_data["test_interactions"]
    n_skills = fold_data["n_skills"]
    n_items = fold_data["n_items"]

    print(f"    Fold {fold_idx + 1}: {len(train_uids)} train, "
          f"{len(test_uids)} test interactions", flush=True)

    set_seed(SEED)
    model = make_model(model_name, dataset_name)
    start_time = time.time()

    try:
        # Compute common mask on test interactions:
        # 1. Exclude first (user, skill) encounters
        # 2. Exclude chunk-start positions (0, 200, 400... per user) so DL
        #    and classical models are evaluated on the identical subset
        common_mask = compute_common_prediction_mask(test_uids, test_skills)
        chunk_starts = compute_chunk_start_positions(test_uids)
        common_mask = common_mask & ~chunk_starts

        if model_name in CLASSICAL_MODELS:
            # Classical: fit on interaction arrays
            if model_name == "BKT":
                model.fit(train_uids, train_skills, train_corrects, train_items)
            elif model_name == "PFA":
                model.fit(train_uids, train_skills, train_corrects, train_items,
                          n_skills=n_skills)

            train_time = time.time() - start_time

            # Pass expanded mask so classical models skip the same positions as DL
            y_true, y_pred, _ = model.predict(
                test_uids, test_skills, test_corrects, test_items,
                mask=common_mask,
            )
            # Student IDs aligned to predictions
            pred_uids = test_uids[common_mask]
        else:
            # DL: fit on chunked sequences
            model.fit(train_seqs, n_skills, n_items)
            train_time = time.time() - start_time

            # Map DL predictions to common mask; returns aligned student IDs
            y_true, y_pred, pred_uids = predict_dl_on_interactions(
                model, test_seqs, test_uids, test_skills,
                test_corrects, common_mask,
            )

        if len(y_true) == 0:
            raise ValueError("No predictions produced")

        # Verify prediction count (DL may miss a few due to short trailing chunks)
        expected = int(common_mask.sum())
        actual = len(y_true)
        if actual < expected:
            pct_diff = 100 * (expected - actual) / expected
            if pct_diff > 0.1:
                print(f"      WARNING: {expected - actual} predictions missing "
                      f"({pct_diff:.2f}%)", flush=True)

        metrics = evaluate_predictions(y_true, y_pred)
        metrics["train_time_seconds"] = round(train_time, 2)
        metrics["n_predictions"] = len(y_true)

        # Student-clustered bootstrap CIs
        y_t = np.array(y_true)
        y_p = np.array(y_pred)
        auc_lo, auc_hi = bootstrap_ci(
            y_t, y_p, roc_auc_score, seed=SEED, student_ids=pred_uids)
        prauc_lo, prauc_hi = bootstrap_ci(
            y_t, y_p, average_precision_score, seed=SEED, student_ids=pred_uids)
        metrics["auc_ci"] = [round(auc_lo, 4), round(auc_hi, 4)]
        metrics["pr_auc_ci"] = [round(prauc_lo, 4), round(prauc_hi, 4)]

        print(f"      AUC={metrics['auc']:.4f} [{auc_lo:.3f},{auc_hi:.3f}] "
              f"PR-AUC={metrics['pr_auc']:.4f} "
              f"RMSE={metrics['rmse']:.4f} n={len(y_true)} Time={train_time:.1f}s",
              flush=True)
        return metrics

    except Exception as e:
        train_time = time.time() - start_time
        print(f"      ERROR: {str(e)}", flush=True)
        traceback.print_exc()
        return {
            "auc": None, "pr_auc": None, "accuracy": None,
            "f1": None, "rmse": None,
            "train_time_seconds": round(train_time, 2),
            "n_predictions": 0, "error": str(e),
        }


def run_benchmark():
    """Run full benchmark: all models x all datasets x 5 folds."""
    all_results = {}

    for dataset_name in DATASETS:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Dataset: {dataset_name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        print(f"  Preparing {N_FOLDS}-fold CV splits...", flush=True)
        fold_data_list = prepare_dataset(
            dataset_name, min_interactions=5, n_folds=N_FOLDS, seed=SEED,
        )

        dataset_results = {}
        for model_name in MODEL_ORDER:
            print(f"\n  Model: {model_name}", flush=True)
            fold_metrics = []

            for fold_idx, fold_data in enumerate(fold_data_list):
                metrics = run_single_fold(model_name, dataset_name, fold_data, fold_idx)
                fold_metrics.append(metrics)

            # Aggregate
            valid = [m for m in fold_metrics if m.get("auc") is not None]
            if valid:
                agg = {}
                for key in ["auc", "pr_auc", "accuracy", "f1", "rmse",
                            "train_time_seconds"]:
                    vals = [m[key] for m in valid if m.get(key) is not None]
                    if vals:
                        agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
                        agg[f"{key}_std"] = round(float(np.std(vals)), 4)

                # Aggregate CI bounds (mean of per-fold CIs)
                for ci_key in ["auc_ci", "pr_auc_ci"]:
                    ci_vals = [m[ci_key] for m in valid if ci_key in m]
                    if ci_vals:
                        lo_mean = round(float(np.mean([c[0] for c in ci_vals])), 4)
                        hi_mean = round(float(np.mean([c[1] for c in ci_vals])), 4)
                        agg[f"{ci_key}_mean"] = [lo_mean, hi_mean]

                agg["n_valid_folds"] = len(valid)
                agg["fold_details"] = fold_metrics
            else:
                agg = {"error": "All folds failed", "fold_details": fold_metrics}

            dataset_results[model_name] = agg
            print(f"  {model_name} mean AUC: {agg.get('auc_mean', 'N/A')}", flush=True)

        all_results[dataset_name] = dataset_results

        with open(os.path.join(OUTPUT_DIR, "benchmark_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved.", flush=True)

    with open(os.path.join(OUTPUT_DIR, "benchmark_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE. Results saved to output/benchmark_results.json")
    return all_results


if __name__ == "__main__":
    results = run_benchmark()
