"""
Preprocessing module for Knowledge Tracing benchmark datasets.
Handles loading, cleaning, sequence creation, and train/test splitting.

Supports two data formats:
  - Chunked sequences (max_len=200) for DL models
  - Full interaction-level arrays for classical models (BKT, PFA)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

DATASETS = {
    "assist2009": "assist2009.csv",
    "assist2015": "assist2015.csv",
    "assist2017": "assist2017.csv",
    "statics2011": "statics2011.csv",
    "algebra2005": "algebra2005.csv",
}

MAX_SEQ_LEN = 200


def load_dataset(name):
    """Load a dataset by name, return cleaned DataFrame."""
    fpath = os.path.join(DATA_DIR, DATASETS[name])
    df = pd.read_csv(fpath, sep="\t")
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["correct"] = df["correct"].astype(int)
    df["skill_id"] = df["skill_id"].astype(int)
    return df


def filter_short_sequences(df, min_interactions=5):
    """Remove students with fewer than min_interactions attempts."""
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    return df[df["user_id"].isin(valid_users)].reset_index(drop=True)


def remap_ids(df):
    """Remap skill_id and item_id to contiguous integers starting from 0."""
    skill_map = {s: i for i, s in enumerate(sorted(df["skill_id"].unique()))}
    item_map = {it: i for i, it in enumerate(sorted(df["item_id"].unique()))}
    df = df.copy()
    df["skill_id"] = df["skill_id"].map(skill_map)
    df["item_id"] = df["item_id"].map(item_map)
    return df, len(skill_map), len(item_map)


def create_sequences(df, max_len=MAX_SEQ_LEN):
    """Convert interaction-level data to chunked student sequences for DL models."""
    sequences = []
    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp")
        skills = group["skill_id"].values
        corrects = group["correct"].values
        items = group["item_id"].values

        for start in range(0, len(skills), max_len):
            end = min(start + max_len, len(skills))
            if end - start >= 3:
                sequences.append({
                    "user_id": user_id,
                    "skills": skills[start:end],
                    "corrects": corrects[start:end],
                    "items": items[start:end],
                    "length": end - start,
                })
    return sequences


def compute_chunk_start_positions(user_ids, max_len=MAX_SEQ_LEN):
    """Boolean mask marking chunk-start positions for each user.

    These are the interactions at positions 0, max_len, 2*max_len, etc.
    within each user's history. DL models cannot predict these positions
    (they are the first position of each chunked sequence), so they must
    be excluded from evaluation for all models to ensure identical
    evaluation subsets.

    Returns a boolean array where True = chunk-start position (to exclude).
    """
    chunk_starts = np.zeros(len(user_ids), dtype=bool)
    user_pos = {}
    for i in range(len(user_ids)):
        uid = user_ids[i]
        if uid not in user_pos:
            user_pos[uid] = 0
        pos = user_pos[uid]
        if pos % max_len == 0:
            chunk_starts[i] = True
        user_pos[uid] = pos + 1
    return chunk_starts


def create_interaction_arrays(df):
    """Convert DataFrame to interaction-level arrays for classical models.

    Returns user_ids, skills, corrects, items as numpy arrays,
    ordered by user then timestamp (preserving temporal order within each user).
    """
    rows = []
    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp")
        for _, row in group.iterrows():
            rows.append((user_id, row["skill_id"], row["correct"], row["item_id"]))
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1].astype(int), arr[:, 2].astype(int), arr[:, 3].astype(int)


def get_folds(df, n_folds=5, seed=42):
    """Create student-level k-fold splits."""
    users = df["user_id"].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in kf.split(users):
        train_users = set(users[train_idx])
        test_users = set(users[test_idx])
        train_df = df[df["user_id"].isin(train_users)]
        test_df = df[df["user_id"].isin(test_users)]
        folds.append((train_df, test_df))
    return folds


def prepare_dataset(name, min_interactions=5, n_folds=5, seed=42):
    """Full preprocessing pipeline.

    Returns list of fold dicts, each containing:
      - train_seqs, test_seqs: chunked sequences for DL models
      - train_interactions, test_interactions: (user_ids, skills, corrects, items) tuples
      - n_skills, n_items: vocabulary sizes
    """
    df = load_dataset(name)
    df = filter_short_sequences(df, min_interactions=min_interactions)
    df, n_skills, n_items = remap_ids(df)
    folds = get_folds(df, n_folds=n_folds, seed=seed)

    result = []
    for train_df, test_df in folds:
        train_seqs = create_sequences(train_df)
        test_seqs = create_sequences(test_df)
        train_interactions = create_interaction_arrays(train_df)
        test_interactions = create_interaction_arrays(test_df)
        result.append({
            "train_seqs": train_seqs,
            "test_seqs": test_seqs,
            "train_interactions": train_interactions,
            "test_interactions": test_interactions,
            "n_skills": n_skills,
            "n_items": n_items,
        })
    return result


if __name__ == "__main__":
    for name in DATASETS:
        print(f"\nProcessing {name}...")
        df = load_dataset(name)
        df_filtered = filter_short_sequences(df)
        print(f"  Before filtering: {len(df)} interactions, {df['user_id'].nunique()} students")
        print(f"  After filtering:  {len(df_filtered)} interactions, {df_filtered['user_id'].nunique()} students")
        print(f"  Skills: {df_filtered['skill_id'].nunique()}")
