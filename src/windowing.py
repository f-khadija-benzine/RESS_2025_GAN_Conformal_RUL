"""
windowing.py — Sliding Window Segmentation and Dataset Splitting

Step 3 of the RESS pipeline.

Takes the per-bearing feature matrices + stage labels produced by
health_indicator.py and turns them into model-ready windows:

    (n_windows, window_size, n_features) + RUL target + stage label

Splitting is done at the BEARING level, not the window level. With
stride=1 consecutive windows overlap by 31/32 time steps, so a random
window-level split would place near-identical windows in both train and
calibration. That leaks information and invalidates the exchangeability
assumption underpinning the conformal coverage guarantee. Holding out
whole bearings keeps the calibration set genuinely independent.

Usage:
    from windowing import make_windows, build_folds, prepare_fold
    folds = build_folds()
    data = prepare_fold(results, folds[0])
"""

import numpy as np
from typing import Dict, List, Tuple


# ── Configuration ────────────────────────────────────────────────────

WINDOW_SIZE = 32      # time steps per window (power of 2 — GAN-friendly)
STRIDE = 1            # slide one recording at a time (max data from short bearings)
N_FEATURES = 20       # 10 time-domain features x 2 channels
N_CAL_BEARINGS = 2    # bearings held out of training for conformal calibration
N_VAL_BEARINGS = 1    # bearings held out for early stopping / model selection


# ── RUL target ───────────────────────────────────────────────────────

def compute_rul_normalized(n_recordings):
    """RUL normalized to [0, 1] as fraction of life remaining.

    RUL(t) = (EOL - t) / EOL

    Gives 1.0 at the first recording and 0.0 at failure. Normalizing
    per bearing is standard on XJTU-SY because raw lifetimes span
    42–2538 minutes; an absolute target in minutes would let long-lived
    bearings dominate the loss. It also makes RMSE directly comparable
    to the published XJTU-SY baselines, which report normalized errors.

    Args:
        n_recordings: total recordings for this bearing

    Returns:
        1D array of shape (n_recordings,), descending from 1.0 to 0.0
    """
    eol = n_recordings - 1
    if eol == 0:
        return np.zeros(1, dtype=np.float32)
    t = np.arange(n_recordings, dtype=np.float32)
    return ((eol - t) / eol).astype(np.float32)


# ── Windowing ────────────────────────────────────────────────────────

def make_windows(features, stage_labels, window_size=WINDOW_SIZE, stride=STRIDE):
    """Slide a fixed-length window over one bearing's feature matrix.

    Each window is labeled by its LAST time step: the RUL and stage at
    the window's right edge. This is the causal convention — the model
    sees the past `window_size` recordings and predicts the state at the
    current moment, never using future information.

    Args:
        features: (n_recordings, 20) feature matrix
        stage_labels: (n_recordings,) integer stage labels (1, 2, 3)
        window_size: number of time steps per window
        stride: step between consecutive windows

    Returns:
        X:     (n_windows, window_size, 20) float32
        y_rul: (n_windows,) float32 in [0, 1]
        y_stage: (n_windows,) int64 in {0, 1, 2}  (zero-indexed for PyTorch)
    """
    n_rec = features.shape[0]

    # Bearings shorter than one window cannot produce any sample
    if n_rec < window_size:
        return (np.empty((0, window_size, features.shape[1]), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64))

    rul = compute_rul_normalized(n_rec)

    starts = np.arange(0, n_rec - window_size + 1, stride)
    n_win = len(starts)

    X = np.zeros((n_win, window_size, features.shape[1]), dtype=np.float32)
    y_rul = np.zeros(n_win, dtype=np.float32)
    y_stage = np.zeros(n_win, dtype=np.int64)

    for i, s in enumerate(starts):
        e = s + window_size
        X[i] = features[s:e]
        y_rul[i] = rul[e - 1]                  # label = last time step
        y_stage[i] = stage_labels[e - 1] - 1   # 1,2,3 -> 0,1,2

    return X, y_rul, y_stage


# ── Fold construction ────────────────────────────────────────────────

MIN_CAL_S3_WINDOWS = 100   # minimum near-failure windows per calibration set
MIN_VAL_WINDOWS = 300      # minimum windows in the validation set


def count_windows(results, bearing_id, window_size=WINDOW_SIZE):
    """Total windows this bearing contributes."""
    if bearing_id not in results:
        return 0
    n_rec = results[bearing_id]['features'].shape[0]
    return max(0, n_rec - window_size + 1)


def count_stage_windows(results, bearing_id, stage, window_size=WINDOW_SIZE):
    """Number of windows of a given stage this bearing would contribute.

    A window inherits the stage of its LAST time step, so the count is the
    number of recordings of that stage that fall at or beyond index
    (window_size - 1).
    """
    if bearing_id not in results:
        return 0
    labels = results[bearing_id]['stage_labels']
    if len(labels) < window_size:
        return 0
    return int((labels[window_size - 1:] == stage).sum())


def build_folds(results=None):
    """Build 5 folds, each testing on one bearing per operating condition.

    Fold k tests on Bearings 1-k, 2-k, 3-k. Every fold therefore spans all
    three speed/load settings, preventing evaluation bias toward any single
    operating condition (Lu et al. 2022 protocol). From the 12 remaining
    bearings, 2 are held out for conformal calibration and 1 for
    validation, leaving 9 for training.

    CALIBRATION SELECTION IS STAGE-AWARE
    ------------------------------------
    Near-failure data is extremely concentrated in this dataset: Bearing
    3-2 alone holds ~51% of all Stage-3 windows, and five bearings hold
    fewer than ten each. Selecting calibration bearings by index rotation
    therefore produces wildly uneven near-failure representation -- one
    fold ended up with 29 Stage-3 calibration windows while another had
    636.

    That matters because the conformal interval width is the (1-alpha)
    quantile of the calibration nonconformity scores, and the scores that
    set the upper tail come predominantly from near-failure windows, where
    the model is least certain. A quantile estimated from 29 samples is
    noise, not a quantile; and per-stage coverage (PICP on Stage 3) cannot
    be claimed at all from so few points.

    Calibration bearings are therefore chosen to satisfy two requirements:
      1. At least MIN_CAL_S3_WINDOWS near-failure windows, so the
         near-failure quantile is estimable.
      2. A stage composition as close as possible to that fold's TEST set,
         since split conformal prediction assumes exchangeability between
         calibration and test. A calibration set of mostly-healthy windows
         paired with a mostly-degraded test set yields intervals calibrated
         on easy examples and applied to hard ones, producing systematic
         under-coverage.

    This is a constraint on the DATA SPLIT, imposed before any model is
    trained; it is not tuned against model performance. Note also that GAN
    augmentation cannot substitute for it: synthetic windows enter the
    TRAINING set only. Calibration must remain real data, or the
    distribution-free coverage guarantee has no basis.

    Args:
        results: dict from HealthIndicatorPipeline.process_all(). If None,
                 falls back to the legacy index-rotation split (kept only
                 for reproducing earlier runs).

    Returns:
        list of 5 dicts with keys 'fold', 'test', 'cal', 'val', 'train'
    """
    all_bearings = [f"Bearing{c}_{j}" for c in (1, 2, 3) for j in range(1, 6)]

    folds = []
    for k in range(1, 6):
        test = [f"Bearing{c}_{k}" for c in (1, 2, 3)]
        remaining = [b for b in all_bearings if b not in test]

        if results is None:
            # Legacy path: arbitrary index rotation (stage-blind)
            cal = [remaining[k % len(remaining)],
                   remaining[(k + 6) % len(remaining)]]
            cal = list(dict.fromkeys(cal))
            while len(cal) < N_CAL_BEARINGS:
                for b in remaining:
                    if b not in cal:
                        cal.append(b)
                        break
        else:
            cal = _select_calibration_bearings(results, remaining, test)

        rest = [b for b in remaining if b not in cal]

        if results is None:
            val = [rest[k % len(rest)]]
        else:
            val = _select_validation_bearing(results, rest, k)

        train = [b for b in rest if b not in val]

        folds.append({'fold': k, 'test': test, 'cal': cal,
                      'val': val, 'train': train})
    return folds


def _select_validation_bearing(results, candidates, fold_k,
                               window_size=WINDOW_SIZE):
    """Pick the validation bearing, requiring a usable number of windows.

    Validation drives early stopping: training halts when the validation
    RMSE stops improving. That signal is only meaningful if it is computed
    over enough windows to be stable. Under index rotation the validation
    bearing was whichever one the counter landed on, which repeatedly
    produced sets of ~90 windows -- far too few, so the epoch-to-epoch
    RMSE would fluctuate on noise alone and the stopping point would be
    effectively arbitrary.

    Candidates below MIN_VAL_WINDOWS are therefore excluded. Among those
    that qualify, the SMALLEST is chosen: validation only needs to be large
    enough to be stable, and every window spent on it is a window taken
    from training. If no candidate reaches the floor (all remaining
    bearings are short), the largest available is used and the caller is
    warned rather than the split failing outright.
    """
    sized = [(b, count_windows(results, b, window_size)) for b in candidates]
    eligible = [(b, n) for b, n in sized if n >= MIN_VAL_WINDOWS]

    if eligible:
        # Smallest bearing that still clears the floor -> preserves training data
        return [min(eligible, key=lambda t: t[1])[0]]

    best, n = max(sized, key=lambda t: t[1])
    print(f"  ! Fold {fold_k}: no validation bearing reaches "
          f"{MIN_VAL_WINDOWS} windows; using {best} ({n} windows). "
          f"Early stopping will be unreliable for this fold.")
    return [best]


def _stage_profile(results, bearings, window_size=WINDOW_SIZE):
    """Fractional stage composition of a set of bearings, as (p1, p2, p3)."""
    counts = np.array([
        sum(count_stage_windows(results, b, s, window_size) for b in bearings)
        for s in (1, 2, 3)
    ], dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def _select_calibration_bearings(results, candidates, test_bearings,
                                 window_size=WINDOW_SIZE):
    """Pick the calibration pair: enough Stage-3 data, and matched to test.

    Scores every candidate pair on how closely its stage profile matches
    the test set's, subject to a hard floor on near-failure windows. If no
    pair clears the floor (possible when the near-failure-rich bearings are
    all in the test set), the floor is relaxed and the pair with the most
    Stage-3 windows is taken, so the split never fails outright.
    """
    from itertools import combinations

    test_profile = _stage_profile(results, test_bearings, window_size)

    scored = []
    for pair in combinations(candidates, N_CAL_BEARINGS):
        s3 = sum(count_stage_windows(results, b, 3, window_size) for b in pair)
        profile = _stage_profile(results, pair, window_size)
        # L1 distance between calibration and test stage composition
        divergence = float(np.abs(profile - test_profile).sum())
        scored.append((pair, s3, divergence))

    eligible = [s for s in scored if s[1] >= MIN_CAL_S3_WINDOWS]

    if eligible:
        # Among pairs with enough near-failure data, take the best match to test
        best = min(eligible, key=lambda s: s[2])
    else:
        # Floor unreachable: maximise near-failure data instead
        best = max(scored, key=lambda s: s[1])

    return list(best[0])


# ── Normalization ────────────────────────────────────────────────────

# Feature order per channel (see health_indicator.FEATURE_NAMES_PER_CHANNEL):
#   0 rms, 1 peak, 2 peak_to_peak, 3 std, 4 kurtosis, 5 skewness,
#   6 crest_factor, 7 shape_factor, 8 impulse_factor, 9 clearance_factor
# repeated for horizontal (0-9) and vertical (10-19) channels.
#
# The amplitude features (rms, peak, peak_to_peak, std), the dimensionless
# impulse features (crest, shape, impulse, clearance) AND kurtosis are
# strictly positive and span several orders of magnitude: they sit near a
# small baseline for most of a bearing's life, then rise sharply at failure
# (kurtosis in particular spikes on impulsive faults — up to ~144 observed).
# Standardising them directly fails, because the mean and standard deviation
# are dominated by the abundant healthy windows, so a near-failure spike
# lands tens of standard deviations out. A log transform compresses this
# multiplicative growth into an additive range before standardisation, which
# both stabilises the scaler and lets the generator reach the near-failure
# region of feature space.
#
# Skewness (5) is excluded: it is bounded, small in magnitude (|skew| < 0.5
# here), and can be negative, so it neither needs nor suits a log transform.

_LOG_FEATURES_PER_CHANNEL = [0, 1, 2, 3, 4, 6, 7, 8, 9]   # all but skewness (5)
LOG_FEATURE_IDX = np.array(
    _LOG_FEATURES_PER_CHANNEL + [i + 10 for i in _LOG_FEATURES_PER_CHANNEL]
)


def log_transform(X):
    """Apply log1p to the amplitude/impulse features, in place-safe fashion.

    Uses log1p(|x|)*sign(x); the features so transformed are non-negative in
    practice, so this reduces to log1p(x), but the sign guard keeps it robust
    to any small negative numerical artefacts.

    Args:
        X: (n_windows, window_size, n_features)

    Returns:
        transformed copy of X
    """
    Xt = np.asarray(X, dtype=np.float32).copy()
    if Xt.size == 0:
        return Xt
    sel = Xt[..., LOG_FEATURE_IDX]
    Xt[..., LOG_FEATURE_IDX] = np.sign(sel) * np.log1p(np.abs(sel))
    return Xt


def fit_scaler(X_train):
    """Compute per-feature mean and std from the TRAINING windows only.

    Expects X_train to have already been log-transformed. Fitting on training
    data alone is essential: using calibration or test statistics would leak
    information. The same scaler is then applied to every split.

    Args:
        X_train: (n_windows, window_size, n_features), log-transformed

    Returns:
        mu:    (n_features,)
        sigma: (n_features,)
    """
    flat = X_train.reshape(-1, X_train.shape[-1])
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0) + 1e-8   # guard against constant features
    return mu.astype(np.float32), sigma.astype(np.float32)


def apply_scaler(X, mu, sigma):
    """Z-score windows using training-set statistics."""
    return ((X - mu) / sigma).astype(np.float32)


# ── Fold preparation ─────────────────────────────────────────────────

def prepare_fold(results, fold, window_size=WINDOW_SIZE, stride=STRIDE,
                 verbose=True):
    """Window and split one fold into train / val / cal / test tensors.

    Args:
        results: dict from HealthIndicatorPipeline.process_all()
        fold: one dict from build_folds()
        window_size, stride: windowing parameters
        verbose: print split summary

    Returns:
        dict with 'X_train', 'y_rul_train', 'y_stage_train', and the
        same for _val, _cal, _test; plus 'scaler' and 'fold'
    """
    split_data = {}

    for split in ('train', 'val', 'cal', 'test'):
        Xs, rs, ss = [], [], []
        for bid in fold[split]:
            if bid not in results:
                continue
            r = results[bid]
            X, y_rul, y_stage = make_windows(
                r['features'], r['stage_labels'], window_size, stride)
            if len(X) == 0:
                if verbose:
                    print(f"  ! {bid} skipped: shorter than window ({window_size})")
                continue
            Xs.append(X); rs.append(y_rul); ss.append(y_stage)

        split_data[split] = (
            np.concatenate(Xs) if Xs else np.empty((0, window_size, N_FEATURES), np.float32),
            np.concatenate(rs) if rs else np.empty(0, np.float32),
            np.concatenate(ss) if ss else np.empty(0, np.int64),
        )

    # Log-transform amplitude/impulse features BEFORE scaling, on every split.
    # The transform is a fixed, data-independent function (no fitted
    # parameters), so applying it to all splits leaks no information.
    for split in ('train', 'val', 'cal', 'test'):
        X, y_rul, y_stage = split_data[split]
        split_data[split] = (log_transform(X), y_rul, y_stage)

    # Scaler fitted on (log-transformed) training windows only, applied everywhere
    mu, sigma = fit_scaler(split_data['train'][0])

    out = {'fold': fold['fold'], 'scaler': (mu, sigma)}
    for split in ('train', 'val', 'cal', 'test'):
        X, y_rul, y_stage = split_data[split]
        out[f'X_{split}'] = apply_scaler(X, mu, sigma) if len(X) else X
        out[f'y_rul_{split}'] = y_rul
        out[f'y_stage_{split}'] = y_stage

    if verbose:
        print(f"\n=== FOLD {fold['fold']} ===")
        for split in ('train', 'val', 'cal', 'test'):
            X = out[f'X_{split}']
            ys = out[f'y_stage_{split}']
            counts = [int((ys == s).sum()) for s in (0, 1, 2)]
            pct = [100 * c / max(len(ys), 1) for c in counts]
            print(f"  {split:5s} {str(fold[split]):45s} "
                  f"{len(X):5d} windows | "
                  f"S1 {counts[0]:5d} ({pct[0]:4.1f}%)  "
                  f"S2 {counts[1]:5d} ({pct[1]:4.1f}%)  "
                  f"S3 {counts[2]:5d} ({pct[2]:4.1f}%)")

    return out


def prepare_all_folds(results, window_size=WINDOW_SIZE, stride=STRIDE,
                      verbose=True):
    """Prepare all 5 folds using stage-aware calibration selection.

    Returns:
        list of 5 fold dicts from prepare_fold()
    """
    folds = build_folds(results)
    return [prepare_fold(results, f, window_size, stride, verbose)
            for f in folds]
