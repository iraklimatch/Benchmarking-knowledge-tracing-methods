"""
Knowledge Tracing model implementations v2.

Fixes from review:
1. PFA: proper categorical skill encoding with skill-specific intercepts
   and skill-specific success/failure slopes (Pavlik et al., 2009)
2. BKT: correct forward-backward EM (Corbett & Anderson, 1995)
3. DKT: unchanged (Piech et al., 2015)
4. SAKT: unchanged (Pandey & Karypis, 2019)
5. AKT renamed to TransformerKT: generic causal Transformer baseline,
   no longer claiming to be full AKT (Ghosh et al., 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, mean_squared_error,
    average_precision_score,
)
from scipy import sparse
import time


# ============================================================
# Seed control for reproducibility
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Evaluation utilities
# ============================================================

def evaluate_predictions(y_true, y_pred_proba):
    """Compute all evaluation metrics."""
    y_true = np.array(y_true, dtype=int)
    y_pred_proba = np.clip(np.array(y_pred_proba, dtype=float), 1e-7, 1 - 1e-7)
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "f1": f1_score(y_true, y_pred_binary),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred_proba)),
    }


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=42, alpha=0.05,
                  student_ids=None):
    """Compute bootstrap 95% confidence interval for a metric.

    If student_ids is provided, uses student-clustered bootstrap (resamples
    students with replacement, then concatenates all interactions for the
    sampled students). This accounts for within-student correlation and
    produces appropriately wider CIs.
    """
    rng = np.random.RandomState(seed)
    scores = []

    if student_ids is not None:
        # Build index: student -> list of row positions
        from collections import defaultdict
        student_index = defaultdict(list)
        for i, sid in enumerate(student_ids):
            student_index[sid].append(i)
        unique_students = np.array(list(student_index.keys()))
        n_students = len(unique_students)

        for _ in range(n_boot):
            sampled = rng.choice(unique_students, size=n_students, replace=True)
            idx = np.concatenate([student_index[s] for s in sampled])
            try:
                scores.append(metric_fn(y_true[idx], y_pred[idx]))
            except ValueError:
                continue
    else:
        # Fallback: standard interaction-level bootstrap
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            try:
                scores.append(metric_fn(y_true[idx], y_pred[idx]))
            except ValueError:
                continue

    scores = np.array(scores)
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return lo, hi


# ============================================================
# Shared: interaction-level data structure
# ============================================================

def flatten_sequences_to_interactions(sequences):
    """Convert sequence-based data back to interaction-level arrays.

    Returns: user_ids, skills, corrects, items (all numpy arrays).
    Each array has one entry per interaction, ordered by user then timestamp.
    """
    user_ids, skills, corrects, items = [], [], [], []
    for seq in sequences:
        n = seq["length"]
        user_ids.extend([seq["user_id"]] * n)
        skills.extend(seq["skills"][:n].tolist())
        corrects.extend(seq["corrects"][:n].tolist())
        items.extend(seq["items"][:n].tolist())
    return (np.array(user_ids), np.array(skills, dtype=int),
            np.array(corrects, dtype=int), np.array(items, dtype=int))


def compute_common_prediction_mask(user_ids, skills):
    """Compute a boolean mask selecting interactions that are NOT the first
    occurrence of a (user, skill) pair. This is the common prediction target
    across all models: we only predict from the second encounter onward."""
    seen = set()
    mask = np.zeros(len(user_ids), dtype=bool)
    for i in range(len(user_ids)):
        key = (user_ids[i], skills[i])
        if key in seen:
            mask[i] = True
        else:
            seen.add(key)
    return mask


# ============================================================
# 1. BKT -- correct forward-backward EM
# ============================================================

class BKTModel:
    """Standard BKT with forward-backward EM per skill.

    Parameters per skill:
        p_L0: initial mastery probability
        p_T:  learning (transition) probability
        p_G:  guess probability
        p_S:  slip probability

    EM uses the standard forward-backward algorithm for HMMs
    (Rabiner, 1989) applied to the two-state BKT formulation.
    """

    def __init__(self, n_iter=50, tol=1e-4):
        self.n_iter = n_iter
        self.tol = tol
        self.params = {}
        self.default_params = (0.5, 0.1, 0.2, 0.1)

    def _forward_backward(self, obs, p_L0, p_T, p_G, p_S):
        """Run forward-backward on a single observation sequence.

        Returns: gamma (posterior P(L_t | all obs)), xi (transition posteriors).
        """
        T = len(obs)
        # Emission probabilities: P(o_t | state)
        # State 0 = not learned, State 1 = learned
        emit = np.zeros((T, 2))
        for t in range(T):
            if obs[t] == 1:  # correct
                emit[t, 0] = p_G          # P(correct | not learned)
                emit[t, 1] = 1.0 - p_S    # P(correct | learned)
            else:             # incorrect
                emit[t, 0] = 1.0 - p_G
                emit[t, 1] = p_S

        # Transition matrix: P(state_t+1 | state_t)
        # From not-learned: can stay not-learned or learn
        # From learned: stays learned (no forgetting)
        A = np.array([
            [1.0 - p_T, p_T],   # from not-learned
            [0.0,       1.0],    # from learned (absorbing)
        ])

        # Prior
        pi = np.array([1.0 - p_L0, p_L0])

        # Forward pass
        alpha = np.zeros((T, 2))
        alpha[0] = pi * emit[0]
        c = np.zeros(T)
        c[0] = alpha[0].sum()
        if c[0] > 0:
            alpha[0] /= c[0]
        else:
            alpha[0] = 0.5
            c[0] = 1.0

        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ A) * emit[t]
            c[t] = alpha[t].sum()
            if c[t] > 0:
                alpha[t] /= c[t]
            else:
                alpha[t] = 0.5
                c[t] = 1.0

        # Backward pass
        beta = np.zeros((T, 2))
        beta[T-1] = 1.0

        for t in range(T-2, -1, -1):
            beta[t] = A @ (emit[t+1] * beta[t+1])
            if c[t+1] > 0:
                beta[t] /= c[t+1]

        # Posteriors
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1.0
        gamma = gamma / gamma_sum

        # Transition posteriors: xi[t] = P(state_t, state_{t+1} | all obs)
        xi = np.zeros((T-1, 2, 2))
        for t in range(T-1):
            numerator = np.outer(alpha[t], emit[t+1] * beta[t+1]) * A
            denom = numerator.sum()
            if denom > 0:
                xi[t] = numerator / denom

        return gamma, xi

    def _em_single_skill(self, sequences):
        """Run EM for a single skill."""
        p_L0, p_T, p_G, p_S = 0.3, 0.1, 0.25, 0.1

        for iteration in range(self.n_iter):
            # Accumulators for M-step
            gamma_0_sum = np.zeros(2)
            xi_sum = np.zeros((2, 2))
            emit_num = np.zeros((2, 2))  # [state, obs] counts

            for obs_seq in sequences:
                if len(obs_seq) < 2:
                    continue
                gamma, xi = self._forward_backward(obs_seq, p_L0, p_T, p_G, p_S)

                # Initial state
                gamma_0_sum += gamma[0]

                # Transitions
                if len(xi) > 0:
                    xi_sum += xi.sum(axis=0)

                # Emissions
                for t in range(len(obs_seq)):
                    o = obs_seq[t]
                    emit_num[:, o] += gamma[t]

            # M-step
            g0_total = gamma_0_sum.sum()
            new_p_L0 = gamma_0_sum[1] / g0_total if g0_total > 0 else p_L0

            # Transition from not-learned
            row0 = xi_sum[0].sum()
            new_p_T = xi_sum[0, 1] / row0 if row0 > 0 else p_T

            # Emissions
            # P(G) = P(correct | not learned) = emit_num[0,1] / (emit_num[0,0] + emit_num[0,1])
            nL_total = emit_num[0].sum()
            new_p_G = emit_num[0, 1] / nL_total if nL_total > 0 else p_G

            # P(S) = P(incorrect | learned) = emit_num[1,0] / (emit_num[1,0] + emit_num[1,1])
            L_total = emit_num[1].sum()
            new_p_S = emit_num[1, 0] / L_total if L_total > 0 else p_S

            # Constrain
            new_p_L0 = np.clip(new_p_L0, 0.001, 0.999)
            new_p_T = np.clip(new_p_T, 0.001, 0.999)
            new_p_G = np.clip(new_p_G, 0.001, 0.40)
            new_p_S = np.clip(new_p_S, 0.001, 0.40)

            # Check convergence
            delta = abs(new_p_L0 - p_L0) + abs(new_p_T - p_T) + abs(new_p_G - p_G) + abs(new_p_S - p_S)
            p_L0, p_T, p_G, p_S = new_p_L0, new_p_T, new_p_G, new_p_S

            if delta < self.tol:
                break

        return p_L0, p_T, p_G, p_S

    def fit(self, user_ids, skills, corrects, items=None):
        """Fit BKT on full (unchunked) interaction-level data."""
        # Group by skill, then by user within each skill
        from collections import defaultdict
        skill_user_seqs = defaultdict(lambda: defaultdict(list))
        for i in range(len(skills)):
            skill_user_seqs[skills[i]][user_ids[i]].append(corrects[i])

        for skill, user_seqs in skill_user_seqs.items():
            seqs = [np.array(s) for s in user_seqs.values() if len(s) >= 2]
            if len(seqs) >= 5:
                self.params[skill] = self._em_single_skill(seqs)

        # Default: population average
        if self.params:
            self.default_params = tuple(np.mean(list(self.params.values()), axis=0))

    def predict(self, user_ids, skills, corrects, items=None, mask=None):
        """Predict on interaction-level data.

        Returns predictions aligned to the common mask (second encounter onward).
        If mask is provided externally, uses that instead of computing its own.
        """
        if mask is None:
            mask = compute_common_prediction_mask(user_ids, skills)
        y_true, y_pred = [], []

        # Track knowledge state per (user, skill)
        state = {}  # (user, skill) -> P(L)

        for i in range(len(skills)):
            uid, skill, correct = user_ids[i], skills[i], corrects[i]
            p_L0, p_T, p_G, p_S = self.params.get(skill, self.default_params)
            key = (uid, skill)

            if key not in state:
                p_L = p_L0
            else:
                p_L = state[key] + (1.0 - state[key]) * p_T

            p_correct = p_L * (1.0 - p_S) + (1.0 - p_L) * p_G

            if mask[i]:
                y_true.append(correct)
                y_pred.append(np.clip(p_correct, 1e-7, 1 - 1e-7))

            # Bayesian update
            if correct == 1:
                p_o_L = 1.0 - p_S
                p_o_nL = p_G
            else:
                p_o_L = p_S
                p_o_nL = 1.0 - p_G
            posterior_L = p_L * p_o_L
            posterior_nL = (1.0 - p_L) * p_o_nL
            total = posterior_L + posterior_nL
            state[key] = posterior_L / total if total > 0 else p_L

        return np.array(y_true), np.array(y_pred), mask


# ============================================================
# 2. PFA -- proper categorical skill encoding
# ============================================================

class PFAModel:
    """Performance Factors Analysis (Pavlik et al., 2009).

    Logistic regression with:
      - Skill-specific intercepts (one-hot encoded skill)
      - Skill-specific success count slopes
      - Skill-specific failure count slopes

    This is the standard PFA formulation where each skill has its own
    difficulty intercept and its own learning rate parameters for
    successes and failures.
    """

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(
            max_iter=5000, C=1.0, solver="lbfgs", penalty="l2",
        )
        self.n_skills = None

    def _extract_features(self, user_ids, skills, corrects):
        """Build the PFA feature matrix.

        For each interaction i, the features are:
        - One-hot skill indicator (n_skills columns): skill intercept
        - Skill-gated success count: n_skills columns, where column k =
          cumulative successes on skill k for this user (0 for other skills)
        - Skill-gated failure count: same structure for failures.

        Total features: 3 * n_skills
        """
        n = len(skills)
        # We build a sparse matrix for efficiency
        rows, cols, vals = [], [], []
        y = np.zeros(n, dtype=int)

        # Track per-(user, skill) counts
        success_counts = {}
        failure_counts = {}

        for i in range(n):
            uid, skill, correct = user_ids[i], skills[i], corrects[i]
            key = (uid, skill)

            s_count = success_counts.get(key, 0)
            f_count = failure_counts.get(key, 0)

            # Skill intercept (columns 0..n_skills-1)
            rows.append(i); cols.append(skill); vals.append(1.0)
            # Skill-gated success count (columns n_skills..2*n_skills-1)
            if s_count > 0:
                rows.append(i); cols.append(self.n_skills + skill); vals.append(float(s_count))
            # Skill-gated failure count (columns 2*n_skills..3*n_skills-1)
            if f_count > 0:
                rows.append(i); cols.append(2 * self.n_skills + skill); vals.append(float(f_count))

            y[i] = correct

            if correct == 1:
                success_counts[key] = s_count + 1
            else:
                failure_counts[key] = f_count + 1

        X = sparse.csr_matrix((vals, (rows, cols)), shape=(n, 3 * self.n_skills))
        return X, y

    def fit(self, user_ids, skills, corrects, items=None, n_skills=None):
        self.n_skills = n_skills or (skills.max() + 1)
        X, y = self._extract_features(user_ids, skills, corrects)
        self.model.fit(X, y)

    def predict(self, user_ids, skills, corrects, items=None, mask=None):
        """Predict, returning only predictions for the common mask.

        If mask is provided externally, uses that instead of computing its own.
        """
        if mask is None:
            mask = compute_common_prediction_mask(user_ids, skills)
        X, y_all = self._extract_features(user_ids, skills, corrects)

        y_pred_all = self.model.predict_proba(X)[:, 1]

        return y_all[mask], y_pred_all[mask], mask


# ============================================================
# PyTorch Dataset for sequence models
# ============================================================

class KTDataset(Dataset):
    def __init__(self, sequences, max_len=200):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        length = min(seq["length"], self.max_len)

        skills = np.zeros(self.max_len, dtype=np.int64)
        corrects = np.zeros(self.max_len, dtype=np.float32)
        targets = np.zeros(self.max_len, dtype=np.float32)
        mask = np.zeros(self.max_len, dtype=np.float32)

        skills[:length] = seq["skills"][:length]
        corrects[:length] = seq["corrects"][:length]
        targets[:length] = seq["corrects"][:length]
        mask[:length] = 1.0

        return {
            "skills": torch.LongTensor(skills),
            "corrects": torch.FloatTensor(corrects),
            "targets": torch.FloatTensor(targets),
            "mask": torch.FloatTensor(mask),
            "length": length,
        }


# ============================================================
# DL model base: shared train/predict logic
# ============================================================

class DLModelBase:
    """Shared training and evaluation for DL sequence models."""

    def __init__(self, lr=0.001, epochs=15, batch_size=64):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.net = None

    def _build_net(self, n_skills, n_items):
        raise NotImplementedError

    def fit(self, train_seqs, n_skills, n_items=None):
        set_seed(42)
        self.net = self._build_net(n_skills, n_items or n_skills).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        dataset = KTDataset(train_seqs)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            worker_init_fn=lambda w: np.random.seed(42 + w),
            generator=torch.Generator().manual_seed(42),
        )

        self.net.train()
        for epoch in range(self.epochs):
            for batch in loader:
                skills = batch["skills"].to(self.device)
                corrects = batch["corrects"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)

                pred = self.net(skills, corrects, mask)
                target_shifted = targets[:, 1:]
                mask_shifted = mask[:, 1:]

                loss = F.binary_cross_entropy(
                    pred * mask_shifted, target_shifted * mask_shifted,
                    reduction="sum",
                ) / mask_shifted.sum().clamp(min=1)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

    def predict_sequences(self, test_seqs):
        """Get raw predictions from DL model on chunked sequences.

        Returns per-interaction (skill, correct, predicted) triples,
        aligned to the shifted (next-step) prediction target.
        """
        dataset = KTDataset(test_seqs)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.net.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                skills = batch["skills"].to(self.device)
                corrects = batch["corrects"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)

                pred = self.net(skills, corrects, mask)
                target_shifted = targets[:, 1:]
                mask_shifted = mask[:, 1:]

                pred_flat = pred[mask_shifted.bool()].cpu().numpy()
                target_flat = target_shifted[mask_shifted.bool()].cpu().numpy()
                all_preds.extend(pred_flat)
                all_targets.extend(target_flat)

        return np.array(all_targets), np.array(all_preds)


# ============================================================
# 3. DKT (Deep Knowledge Tracing) -- Piech et al. (2015)
# ============================================================

class DKTNet(nn.Module):
    def __init__(self, n_skills, hidden_dim=100, dropout=0.2):
        super().__init__()
        self.n_skills = n_skills
        self.input_dim = n_skills * 2
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_skills)

    def forward(self, skills, corrects, mask):
        batch_size, seq_len = skills.shape
        x = torch.zeros(batch_size, seq_len, self.input_dim, device=skills.device)
        skill_idx = skills + corrects.long() * self.n_skills
        x.scatter_(2, skill_idx.unsqueeze(2), 1.0)

        h, _ = self.lstm(x)
        h = self.dropout(h)
        output = torch.sigmoid(self.fc(h))

        pred = output[:, :-1, :]
        next_skills = skills[:, 1:]
        pred = pred.gather(2, next_skills.unsqueeze(2)).squeeze(2)
        return pred


class DKTModel(DLModelBase):
    def __init__(self, hidden_dim=100, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def _build_net(self, n_skills, n_items):
        return DKTNet(n_skills, self.hidden_dim)


# ============================================================
# 4. SAKT (Self-Attentive KT) -- Pandey & Karypis (2019)
# ============================================================

class SAKTNet(nn.Module):
    def __init__(self, n_skills, embed_dim=64, max_len=200, dropout=0.2):
        super().__init__()
        self.n_skills = n_skills
        self.interaction_embed = nn.Embedding(n_skills * 2, embed_dim)
        self.skill_embed = nn.Embedding(n_skills, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads=1, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, skills, corrects, mask):
        batch_size, seq_len = skills.shape
        interactions = skills + corrects.long() * self.n_skills
        interaction_emb = self.interaction_embed(interactions)
        positions = torch.arange(seq_len, device=skills.device).unsqueeze(0).expand(batch_size, -1)
        interaction_emb = interaction_emb + self.pos_embed(positions)
        skill_emb = self.skill_embed(skills)

        query = skill_emb[:, 1:, :]
        key = interaction_emb[:, :-1, :]
        value = interaction_emb[:, :-1, :]
        causal = torch.triu(torch.ones(seq_len - 1, seq_len - 1, device=skills.device), diagonal=1).bool()

        attn_out, _ = self.attn(query, key, value, attn_mask=causal)
        attn_out = self.layer_norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(attn_out)
        ffn_out = self.layer_norm2(attn_out + ffn_out)
        return torch.sigmoid(self.output_layer(ffn_out).squeeze(-1))


class SAKTModel(DLModelBase):
    def __init__(self, embed_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def _build_net(self, n_skills, n_items):
        return SAKTNet(n_skills, self.embed_dim)


# ============================================================
# 5. TransformerKT -- generic causal Transformer baseline
#    (NOT full AKT; does not implement Rasch embeddings or
#     monotonic attention from Ghosh et al., 2020)
# ============================================================

class TransformerKTNet(nn.Module):
    """Generic causal Transformer encoder for knowledge tracing.

    Architecture: skill + correctness + positional embeddings fed into
    a standard Transformer encoder with causal masking. Output at
    position t predicts correctness at t+1.
    """

    def __init__(self, n_skills, embed_dim=64, n_heads=4, n_layers=2, max_len=200, dropout=0.2):
        super().__init__()
        self.skill_embed = nn.Embedding(n_skills, embed_dim)
        self.interaction_embed = nn.Embedding(2, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, 1),
        )

    def forward(self, skills, corrects, mask):
        batch_size, seq_len = skills.shape
        skill_emb = self.skill_embed(skills)
        correct_emb = self.interaction_embed(corrects.long())
        positions = torch.arange(seq_len, device=skills.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)

        x = skill_emb + correct_emb + pos_emb
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=skills.device)
        pad_mask = (mask == 0)

        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)

        context = encoded[:, :-1, :]
        next_skill_emb = skill_emb[:, 1:, :]
        combined = torch.cat([context, next_skill_emb], dim=-1)
        return torch.sigmoid(self.output_layer(combined).squeeze(-1))


class TransformerKTModel(DLModelBase):
    def __init__(self, embed_dim=64, n_heads=4, n_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

    def _build_net(self, n_skills, n_items):
        return TransformerKTNet(n_skills, self.embed_dim, self.n_heads, self.n_layers)
