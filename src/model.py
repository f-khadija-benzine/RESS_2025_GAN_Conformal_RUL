"""
model.py — Multi-task BiLSTM for RUL Prediction (Step 6)

This file starts with the BASELINE: a 2-layer bidirectional LSTM with a
temporal-attention pooling layer and a single RUL regression head. No
augmentation, no stage-classification head — this establishes the anchor
RMSE against which every later contribution (GAN augmentation, the stage
head, RUL-loss masking) is measured.

The stage-classification head and the multi-task loss are added in a later
iteration; the encoder is written so that head can be attached without
changing the shared trunk.

Design:
  * Shared encoder: 2-layer BiLSTM over the 32x20 window, then temporal
    attention that pools the 32 time steps into one context vector by a
    learned weighted average (so the model can emphasise the informative
    end of the window rather than treating all steps equally).
  * RUL head: two dense layers -> scalar in [0, 1] (sigmoid), matching the
    per-bearing normalised RUL target.

Reporting helper `rul_metrics` converts normalised predictions back to
minutes per bearing (primary metric, comparable to Lu et al. and the
XJTU-SY literature) and also returns the normalised RMSE.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn


# ── Configuration ────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # data shape
    window_size: int = 32
    n_features: int = 20

    # encoder
    hidden: int = 48            # was 64 — fewer params to memorise with
    num_layers: int = 2
    dropout: float = 0.4        # was 0.2 — stronger regularisation

    # heads
    rul_hidden: int = 48

    # training
    epochs: int = 150           # was 100 — more room now that val can improve
    batch_size: int = 64
    lr: float = 5e-4            # was 1e-3 — slower, less prone to snap-memorise
    weight_decay: float = 5e-4  # was 1e-5 — much stronger L2
    patience: int = 25          # was 15 — give generalisation time to appear
    grad_clip: float = 5.0

    seed: int = 42
    device: str = 'cuda'


# ── Temporal attention pooling ───────────────────────────────────────

class TemporalAttention(nn.Module):
    """Pool a sequence (B, T, H) into (B, H) by a learned weighted average.

    A small scoring network assigns each time step a scalar weight; the
    weights are softmax-normalised over time and used to average the step
    representations. This lets the model concentrate on the informative
    part of the window (typically the most recent steps) instead of giving
    every step equal say.
    """

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):                     # x: (B, T, H)
        w = self.score(x)                     # (B, T, 1)
        w = torch.softmax(w, dim=1)           # weights over time
        context = (w * x).sum(dim=1)          # (B, H)
        return context, w.squeeze(-1)         # also return weights (for inspection)


# ── Shared encoder ───────────────────────────────────────────────────

class BiLSTMEncoder(nn.Module):
    """2-layer BiLSTM + temporal attention -> context vector."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.attn = TemporalAttention(2 * cfg.hidden)   # bidirectional -> 2H

    def forward(self, x):                     # x: (B, T, F)
        seq, _ = self.lstm(x)                 # (B, T, 2H)
        context, attn_w = self.attn(seq)      # (B, 2H)
        return context, attn_w


# ── Baseline model: encoder + RUL head ───────────────────────────────

class RULModel(nn.Module):
    """Baseline single-task model: shared encoder + RUL regression head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = BiLSTMEncoder(cfg)
        self.rul_head = nn.Sequential(
            nn.Linear(2 * cfg.hidden, cfg.rul_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.rul_hidden, 1),
            nn.Sigmoid(),                     # RUL target is normalised to [0, 1]
        )

    def forward(self, x):
        context, attn_w = self.encoder(x)
        rul = self.rul_head(context).squeeze(-1)   # (B,)
        return rul


# ── Training / evaluation ────────────────────────────────────────────

class RULTrainer:
    """Trains the baseline RUL model for one fold with early stopping."""

    def __init__(self, cfg: Optional[ModelConfig] = None):
        self.cfg = cfg or ModelConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.device = torch.device(
            self.cfg.device if torch.cuda.is_available() else 'cpu')
        self.model = RULModel(self.cfg).to(self.device)
        self.history = {'train_rmse': [], 'val_rmse': []}

    def _loader(self, X, y, shuffle):
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=shuffle)

    def fit(self, X_train, y_train, X_val, y_val,
            stage_train=None, stage_val=None, mask_healthy=True, verbose=True):
        """Train the baseline RUL model for one fold.

        Args:
            X_train, y_train: training windows and normalised RUL
            X_val, y_val: validation windows and RUL (real, for early stopping)
            stage_train: (N,) 0-indexed stage labels for training windows.
                Required when mask_healthy=True.
            stage_val: (N,) 0-indexed stage labels for validation windows.
                When provided with mask_healthy, early stopping tracks
                validation RMSE on post-FPT windows only — the same regime the
                model is trained on — rather than on all windows (which would
                include healthy windows the RUL head never learned).
            mask_healthy: if True, the RUL regression loss is computed only on
                post-FPT windows (stage index >= 1). Pre-FPT healthy windows
                have a flat, stationary signal but a steadily changing RUL
                label, so the mapping from window to RUL is ill-posed there;
                including them biases the regressor toward predicting the mean
                (and, because healthy windows carry high RUL, toward
                over-predicting remaining life). Masking restricts the
                regression to the regime where degradation is observable.
        """
        cfg = self.cfg
        X = torch.as_tensor(X_train, dtype=torch.float32)
        y = torch.as_tensor(y_train, dtype=torch.float32)

        # For early stopping, evaluate on post-FPT validation windows when
        # masking, so the stopping signal matches the training objective.
        Xv_eval, yv_eval = X_val, y_val
        if mask_healthy and stage_val is not None:
            vkeep = np.asarray(stage_val) >= 1
            if vkeep.sum() > 0:
                Xv_eval = np.asarray(X_val)[vkeep]
                yv_eval = np.asarray(y_val)[vkeep]

        if mask_healthy:
            if stage_train is None:
                raise ValueError("mask_healthy=True requires stage_train")
            # post-FPT = stage index >= 1 (0=healthy, 1=early, 2=near-failure)
            keep = np.asarray(stage_train) >= 1
            X, y = X[keep], y[keep]
            if verbose:
                print(f"  RUL masking: {keep.sum()}/{len(keep)} post-FPT "
                      f"windows kept for regression")

        ds = torch.utils.data.TensorDataset(X, y)
        tr = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float('inf')
        best_state = None
        bad_epochs = 0

        for epoch in range(cfg.epochs):
            self.model.train()
            sq_err, n = 0.0, 0
            for xb, yb in tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                opt.step()
                sq_err += ((pred - yb) ** 2).sum().item(); n += len(yb)
            train_rmse = (sq_err / n) ** 0.5

            val_rmse = self.evaluate_rmse(Xv_eval, yv_eval)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)

            if val_rmse < best_val - 1e-5:
                best_val = val_rmse
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
                print(f"  epoch {epoch:3d} | train RMSE {train_rmse:.4f} | "
                      f"val RMSE {val_rmse:.4f} | best {best_val:.4f}")

            if bad_epochs >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} "
                          f"(no val improvement for {cfg.patience})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        preds = []
        for i in range(0, len(X), self.cfg.batch_size):
            preds.append(self.model(X[i:i + self.cfg.batch_size]).cpu().numpy())
        return np.concatenate(preds) if preds else np.empty(0)

    def evaluate_rmse(self, X, y):
        """Normalised RMSE (the quantity early stopping tracks)."""
        if len(X) == 0:
            return float('nan')
        pred = self.predict(X)
        return float(np.sqrt(np.mean((pred - np.asarray(y)) ** 2)))


# ── Metrics ──────────────────────────────────────────────────────────

def rul_metrics(y_true_norm, y_pred_norm, lifetimes_min):
    """Pooled RUL error over all supplied windows.

    The model predicts normalised RUL in [0, 1]; multiplying by each window's
    bearing lifetime recovers minutes. This pools every window together, so
    the minutes-RMSE is dominated by the longest-lived bearing in the set;
    it is a valid aggregate but not directly comparable to the per-bearing
    cumulative RMSE reported by Lu et al. (use `cumulative_rmse` for that).

    Args:
        y_true_norm:   (N,) true normalised RUL
        y_pred_norm:   (N,) predicted normalised RUL
        lifetimes_min: (N,) total lifetime in minutes of each window's bearing

    Returns:
        dict with rmse_min, mae_min (minutes), rmse_norm, mae_norm
        (normalised 0-1 scale), and r2 (on normalised RUL)
    """
    yt = np.asarray(y_true_norm, dtype=float)
    yp = np.asarray(y_pred_norm, dtype=float)
    life = np.asarray(lifetimes_min, dtype=float)

    err_min = (yp - yt) * life
    rmse_min = float(np.sqrt(np.mean(err_min ** 2)))
    mae_min = float(np.mean(np.abs(err_min)))
    rmse_norm = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae_norm = float(np.mean(np.abs(yp - yt)))

    # R^2 on the normalised RUL: fraction of variance explained.
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return {'rmse_min': rmse_min, 'mae_min': mae_min,
            'rmse_norm': rmse_norm, 'mae_norm': mae_norm, 'r2': r2}


def per_bearing_rmse(y_true_norm, y_pred_norm, bearing_ids, lifetimes_min,
                     stages=None, post_fpt_only=False):
    """RMSE in minutes computed separately for each bearing.

    Pooling all windows into one RMSE lets the longest bearing dominate.
    Computing per bearing and then averaging weights every bearing equally,
    which is what the XJTU-SY literature generally reports.

    Args:
        y_true_norm, y_pred_norm: (N,) normalised RUL
        bearing_ids:  (N,) the bearing each window belongs to
        lifetimes_min:(N,) bearing lifetime in minutes per window
        stages:       (N,) stage label per window (1/2/3); required if post_fpt_only
        post_fpt_only: if True, score only windows past the FPT (stage >= 2),
                       matching Lu et al., who evaluate over the FPT-EOL span

    Returns:
        dict: {bearing_id: rmse_min, ...}, plus '_mean'
    """
    yt = np.asarray(y_true_norm, dtype=float)
    yp = np.asarray(y_pred_norm, dtype=float)
    bids = np.asarray(bearing_ids)
    life = np.asarray(lifetimes_min, dtype=float)

    mask = np.ones(len(yt), dtype=bool)
    if post_fpt_only:
        if stages is None:
            raise ValueError("post_fpt_only requires stages")
        mask = np.asarray(stages) >= 2         # stage 2 or 3 = past FPT

    out = {}
    for b in np.unique(bids):
        sel = mask & (bids == b)
        if sel.sum() == 0:
            continue
        err_min = (yp[sel] - yt[sel]) * life[sel]
        out[str(b)] = float(np.sqrt(np.mean(err_min ** 2)))

    if out:
        out['_mean'] = float(np.mean([v for k, v in out.items() if k != '_mean']))
    return out


def phm_score(y_true_norm, y_pred_norm, bearing_ids):
    """PHM 2012 / PRONOSTIA scoring function, per-bearing then averaged.

    This is the bearing-challenge form used across the XJTU-SY literature,
    based on percentage error rather than raw error. For each bearing the
    percentage error is

        Er = 100 * (RUL_true - RUL_pred) / RUL_true

    so Er > 0 means the prediction is EARLY (predicted less life than
    remained) and Er < 0 means LATE (predicted more life than remained).
    Late predictions are penalised more steeply, reflecting their greater
    operational risk. The per-bearing accuracy is

        A = exp(-ln(0.5) * Er / 5)    if Er <= 0   (late)
        A = exp(+ln(0.5) * Er / 20)   if Er >  0   (early)

    and the Score is the mean of A over bearings. A is in (0, 1]; HIGHER is
    BETTER, with 1.0 a perfect prediction.

    The score is evaluated at each bearing's failure point, i.e. on the
    window whose true RUL is smallest (closest to EOL), following the
    challenge convention of scoring the final RUL estimate per unit.

    Args:
        y_true_norm, y_pred_norm: (N,) normalised RUL
        bearing_ids: (N,) bearing per window

    Returns:
        dict {bearing_id: A, ..., '_score': mean A}
    """
    yt = np.asarray(y_true_norm, dtype=float)
    yp = np.asarray(y_pred_norm, dtype=float)
    bids = np.asarray(bearing_ids)

    ln_half = np.log(0.5)
    out = {}
    for b in np.unique(bids):
        sel = bids == b
        idx = np.where(sel)[0]
        # Score at the failure point: the window nearest EOL. Normalised RUL
        # reaches 0 at EOL, which makes the percentage-error denominator
        # undefined, so we take the window nearest EOL whose true RUL is still
        # above a small floor.
        cand = idx[yt[idx] >= 0.02]
        if len(cand) == 0:
            continue
        j = cand[np.argmin(yt[cand])]
        true_val = yt[j]
        er = 100.0 * (true_val - yp[j]) / true_val
        if er <= 0:                        # over-prediction (late) — penalise hard
            a = np.exp(-ln_half * er / 5.0)
        else:                              # under-prediction (early) — penalise gently
            a = np.exp(+ln_half * er / 20.0)
        out[str(b)] = float(a)

    if out:
        out['_score'] = float(np.mean([v for k, v in out.items() if k != '_score']))
    return out


def cumulative_rmse(per_bearing, prognostic_durations):
    """Lu et al.'s cumulative RMSE: per-bearing RMSE weighted by prognostic
    duration Delta_T = t_EOL - t_FPT + 1.

    This is the exact quantity in their Table 6 (cumulative row, 21.90 for
    HP-JT), so a value computed here is directly comparable to their
    published figure.

    Args:
        per_bearing:  {bearing_id: rmse_min} (from per_bearing_rmse, drop '_mean')
        prognostic_durations: {bearing_id: Delta_T in minutes}

    Returns:
        float cumulative RMSE
    """
    ids = [b for b in per_bearing if b != '_mean' and b in prognostic_durations]
    if not ids:
        return float('nan')
    weights = np.array([prognostic_durations[b] for b in ids], dtype=float)
    rmses = np.array([per_bearing[b] for b in ids], dtype=float)
    return float(np.sum(weights * rmses) / np.sum(weights))


# ── Standard evaluation grid ─────────────────────────────────────────

# Stage index convention (0-indexed, as stored in y_stage):
#   0 = healthy (S1) | 1 = early (S2) | 2 = near-failure (S3)
_STAGE_ROWS = [('healthy', [0]), ('early', [1]), ('nearfail', [2]),
               ('postFPT', [1, 2]), ('overall', [0, 1, 2])]


def evaluate_grid(y_true, y_pred, stages, bearing_ids):
    """Standard normalised-RMSE grid for one fold, computed two ways.

    For each window subset (healthy / early / near-failure / post-FPT /
    overall) returns RMSE on the normalised RUL, computed both:

      pooled       : all windows in the subset thrown together (weights each
                     WINDOW equally; long bearings dominate).
      per_bearing  : RMSE per bearing, then averaged (weights each BEARING
                     equally; matches the XJTU-SY literature).

    Args:
        y_true, y_pred: (N,) normalised RUL
        stages: (N,) 0-indexed stage labels (0/1/2)
        bearing_ids: (N,) bearing identifier per window

    Returns:
        dict: {subset: {'pooled': rmse, 'per_bearing': rmse, 'n': n_windows}}
    """
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    st = np.asarray(stages)
    bids = np.asarray(bearing_ids)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float('nan')

    grid = {}
    for name, stage_vals in _STAGE_ROWS:
        sel = np.isin(st, stage_vals)
        n = int(sel.sum())
        if n == 0:
            grid[name] = {'pooled': float('nan'),
                          'per_bearing': float('nan'), 'n': 0}
            continue
        pooled = rmse(yp[sel], yt[sel])
        # per-bearing: RMSE for each bearing present in this subset, then mean
        per = []
        for b in np.unique(bids[sel]):
            bsel = sel & (bids == b)
            if bsel.sum() > 0:
                per.append(rmse(yp[bsel], yt[bsel]))
        per_bearing = float(np.mean(per)) if per else float('nan')
        grid[name] = {'pooled': pooled, 'per_bearing': per_bearing, 'n': n}
    return grid


def print_grid(grid, title=''):
    """Pretty-print one fold's evaluation grid."""
    if title:
        print(title)
    print(f"  {'subset':10s} {'n':>6s} {'pooled':>10s} {'per-bearing':>12s}")
    print(f"  {'-'*40}")
    for name, _ in _STAGE_ROWS:
        g = grid[name]
        print(f"  {name:10s} {g['n']:6d} {g['pooled']:10.4f} "
              f"{g['per_bearing']:12.4f}")


def average_grids(grids):
    """Average a list of per-fold grids into one grid (mean over folds).

    Each subset's pooled / per_bearing RMSE is averaged across folds. This is
    the standard 5-fold cross-validated result to report.
    """
    out = {}
    for name, _ in _STAGE_ROWS:
        pooled = np.nanmean([g[name]['pooled'] for g in grids])
        perb = np.nanmean([g[name]['per_bearing'] for g in grids])
        ntot = int(np.sum([g[name]['n'] for g in grids]))
        out[name] = {'pooled': float(pooled),
                     'per_bearing': float(perb), 'n': ntot}
    return out


# ═════════════════════════════════════════════════════════════════════
#  RESULTS CONSISTENCY CONTRACT  — read before adding any new evaluation
# ═════════════════════════════════════════════════════════════════════
#
#  Every experiment (baseline, +GAN, +oversample, +multitask, conformal)
#  MUST report results through the SAME functions so numbers are
#  comparable across notebooks and across the paper. Do not hand-roll
#  stage splits or ad-hoc RMSE in a notebook cell — call these:
#
#    evaluate_grid(y_true, y_pred, stages, bearing_ids)
#        -> normalised-RMSE grid, 5 subsets x {pooled, per_bearing}
#        subsets: healthy / early / nearfail / postFPT / overall
#    print_grid(grid, title)          -> standard printout
#    average_grids([grids])           -> mean over folds
#    secondary_metrics(...)           -> PHM score, cumulative-min, R2
#
#  ALWAYS state three things with any number: SCALE (normalised [0,1] or
#  minutes), SUBSET (which of the 5 rows), AGGREGATION (pooled or
#  per-bearing). The paper's primary table is the per_bearing column of
#  the grid; pooled is kept for diagnosis. Secondary metrics go in a
#  SEPARATE table with explicit units.
#
#  RUN-TO-RUN NOISE: LSTM training on GPU is non-deterministic. A single
#  run's RMSE wobbles by roughly +/-0.02-0.03. Before claiming any config
#  beats another, run MULTIPLE SEEDS and report mean +/- std; a difference
#  smaller than the std is not a real effect.
# ═════════════════════════════════════════════════════════════════════


def secondary_metrics(y_true_norm, y_pred_norm, stages, bearing_ids,
                      lifetimes_min, prognostic_durations=None):
    """Secondary metrics, reported once per configuration (units explicit).

    Returns a dict with:
      r2                : R^2 on normalised RUL, post-FPT windows (the regime
                          the model is meant to predict)
      phm_score         : PHM 2012 score (higher better, penalises late harder)
      cumulative_min    : dT-weighted per-bearing RMSE in MINUTES, post-FPT,
                          directly comparable to Lu et al. (needs
                          prognostic_durations {bearing_id: dT}); else nan
    """
    yt = np.asarray(y_true_norm, float)
    yp = np.asarray(y_pred_norm, float)
    st = np.asarray(stages)
    bids = np.asarray(bearing_ids)
    life = np.asarray(lifetimes_min, float)

    post = st >= 1
    # R2 on post-FPT
    if post.sum() > 1:
        ss_res = np.sum((yt[post] - yp[post]) ** 2)
        ss_tot = np.sum((yt[post] - yt[post].mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    else:
        r2 = float('nan')

    sc = phm_score(yt, yp, bids).get('_score', float('nan'))

    cum = float('nan')
    if prognostic_durations is not None:
        pb = per_bearing_rmse(yt, yp, bids, life,
                              stages=st + 1, post_fpt_only=True)
        cum = cumulative_rmse(pb, prognostic_durations)

    return {'r2': r2, 'phm_score': sc, 'cumulative_min': cum}
