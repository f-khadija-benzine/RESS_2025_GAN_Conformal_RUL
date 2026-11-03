"""
cvplus.py — Nested cross-conformal (CV+) calibration for C3

WHY
---
Split conformal calibrates each outer fold on only N_CAL_BEARINGS = 2
bearings. Conformal coverage rests on calibration and test data being
exchangeable, and two bearings cannot represent three different test
bearings: even with the true stage, overall coverage was 0.79-0.82 at a
nominal 0.90 (results/c3_offline).

CV+ (Barber, Candes, Ramdas & Tibshirani, 2021, "Predictive inference with
the jackknife+", Annals of Statistics 49(1)) uses every non-test bearing for
calibration without letting any model score data it was trained on:

  * Outer fold k: the 3 test bearings are set aside, exactly as before.
  * The 12 remaining bearings are split into K_INNER = 4 groups of 3 (one per
    operating condition, mirroring the outer protocol).
  * Inner model j is trained on the other 9 bearings (1 of them used for
    early stopping) and predicts its held-out group -> out-of-fold residuals
    R_i for all 12 bearings.
  * For a test window x, with i ranging over calibration windows and j(i) the
    inner model that did not see i:
        lower = floor(alpha (n+1))-th smallest of  mu_{j(i)}(x) - R_i
        upper = ceil((1-alpha)(n+1))-th smallest of mu_{j(i)}(x) + R_i
    Coverage is >= 1 - 2 alpha in the worst case and close to 1 - alpha in
    practice (Barber et al. 2021, Thm. 4).
  * Mondrian variant: i is restricted to calibration windows of the test
    window's group (predicted stage, true stage, or none).

Everything an inner model sees (scaler, GAN, early stopping) comes from its
own 9 training bearings only.
"""

import numpy as np

from windowing import (make_windows, log_transform, fit_scaler, apply_scaler,
                       _select_validation_bearing, WINDOW_SIZE, STRIDE, N_FEATURES)

K_INNER = 4
ALL_BEARINGS = [f"Bearing{c}_{j}" for c in (1, 2, 3) for j in range(1, 6)]
BEARING_INDEX = {b: i for i, b in enumerate(ALL_BEARINGS)}


# ── Splits ───────────────────────────────────────────────────────────

def build_inner_folds(results, outer_fold):
    """Split the 12 non-test bearings of an outer fold into K_INNER groups.

    Group j holds the j-th remaining bearing of each operating condition, so
    every inner model is trained on all three conditions.

    Returns:
        list of dicts {'inner', 'holdout', 'val', 'train'}
    """
    test = set(outer_fold['test'])
    remaining = [b for b in ALL_BEARINGS if b not in test]
    by_cond = {c: [b for b in remaining if b.startswith(f"Bearing{c}_")] for c in (1, 2, 3)}
    assert all(len(v) == K_INNER for v in by_cond.values()), by_cond
    inner = []
    for j in range(K_INNER):
        holdout = [by_cond[c][j] for c in (1, 2, 3)]
        rest = [b for b in remaining if b not in holdout]
        val = _select_validation_bearing(results, rest, f"{outer_fold['fold']}.{j + 1}")
        train = [b for b in rest if b not in val]
        inner.append({'inner': j + 1, 'holdout': holdout, 'val': val, 'train': train})
    return inner


def prepare_inner(results, outer_fold, inner_fold, scaling_method='log_clip',
                  rul_target='piecewise'):
    """Window one inner model's data; scaler fitted on its training bearings.

    The outer test bearings are scaled with this inner model's scaler, since
    every inner model predicts them.

    Returns:
        dict with X_/y_rul_/y_stage_/bid_ for train, val, hold, test
    """
    splits = {'train': inner_fold['train'], 'val': inner_fold['val'],
              'hold': inner_fold['holdout'], 'test': outer_fold['test']}
    raw = {}
    for name, bearings in splits.items():
        Xs, rs, ss, bs = [], [], [], []
        for bid in bearings:
            if bid not in results:
                continue
            r = results[bid]
            X, y, s = make_windows(r['features'], r['stage_labels'], WINDOW_SIZE, STRIDE,
                                   fpt=r.get('fpt_idx', None), rul_target=rul_target)
            if len(X) == 0:
                continue
            Xs.append(X); rs.append(y); ss.append(s)
            bs.append(np.full(len(X), BEARING_INDEX[bid], np.int64))
        X = np.concatenate(Xs) if Xs else np.empty((0, WINDOW_SIZE, N_FEATURES), np.float32)
        if scaling_method in ('log_clip', 'log_robust') and len(X):
            X = log_transform(X)
        raw[name] = (X,
                     np.concatenate(rs) if rs else np.empty(0, np.float32),
                     np.concatenate(ss) if ss else np.empty(0, np.int64),
                     np.concatenate(bs) if bs else np.empty(0, np.int64))
    scaler = fit_scaler(raw['train'][0], method=scaling_method)
    out = {'outer': outer_fold['fold'], 'inner': inner_fold['inner']}
    for name, (X, y, s, b) in raw.items():
        out[f'X_{name}'] = apply_scaler(X, scaler)
        out[f'y_rul_{name}'] = y
        out[f'y_stage_{name}'] = s
        out[f'bid_{name}'] = b
    return out


# ── CV+ intervals ────────────────────────────────────────────────────

def _kth(values, k):
    """k-th smallest (1-indexed) along axis 1; -inf / +inf outside 1..n."""
    n = values.shape[1]
    if k < 1:
        return np.full(values.shape[0], -np.inf)
    if k > n:
        return np.full(values.shape[0], np.inf)
    return np.partition(values, k - 1, axis=1)[:, k - 1]


def cvplus_intervals(mu_test, cal_resid, cal_inner, cal_group, test_group,
                     alpha=0.1, chunk=512):
    """Mondrian CV+ intervals.

    Args:
        mu_test:    (m, K) predictions of each inner model on the test windows
        cal_resid:  (n,) out-of-fold residuals |y_i - mu_{j(i)}(x_i)|
        cal_inner:  (n,) index j(i) in 0..K-1 of the model that did not see i
        cal_group:  (n,) group of each calibration window
        test_group: (m,) group of each test window

    Returns:
        lower, upper: (m,) interval bounds (not clipped)
    """
    m = len(mu_test)
    lower = np.full(m, -np.inf); upper = np.full(m, np.inf)
    for g in np.unique(test_group):
        ti = np.flatnonzero(test_group == g)
        ci = np.flatnonzero(cal_group == g)
        n = len(ci)
        if n == 0:
            continue                       # no calibration data: unbounded
        k_lo = int(np.floor(alpha * (n + 1)))
        k_hi = int(np.ceil((1 - alpha) * (n + 1)))
        R = cal_resid[ci]; J = cal_inner[ci]
        for a in range(0, len(ti), chunk):
            t = ti[a:a + chunk]
            mu = mu_test[t][:, J]          # (chunk, n): mu_{j(i)}(x)
            lower[t] = _kth(mu - R, k_lo)
            upper[t] = _kth(mu + R, k_hi)
    return lower, upper


def interval_metrics(y, lower, upper, subsets):
    """PICP, raw MPIW and MPIW after clipping to the RUL range [0, 1]."""
    cov = (y >= lower) & (y <= upper)
    w_raw = upper - lower
    w_clip = np.clip(upper, 0, 1) - np.clip(lower, 0, 1)
    out = {}
    for name, sel in subsets.items():
        if sel.sum() == 0:
            out[name] = {'picp': np.nan, 'mpiw_raw': np.nan, 'mpiw': np.nan, 'n': 0}
        else:
            out[name] = {'picp': float(cov[sel].mean()),
                         'mpiw_raw': float(np.mean(w_raw[sel])),
                         'mpiw': float(np.mean(w_clip[sel])),
                         'n': int(sel.sum())}
    return out


def stage_subsets(stages):
    st = np.asarray(stages)
    return {'healthy': st == 0, 'early': st == 1, 'nearfail': st == 2,
            'postFPT': st >= 1, 'overall': np.ones(len(st), bool)}


def evaluate_outer(runs, alpha=0.1):
    """CV+ evaluation of one (config, seed, outer fold) from its K inner runs.

    Args:
        runs: list of K dicts loaded from the saved npz files, ordered by inner
              index, each with yp_hold, y_hold, s_hold, shat_hold, yp_test,
              y_test, s_test, pstage_test (K-th model's stage probabilities
              on test), bid_test.

    Returns:
        dict: {'point': (y_test, yhat_test, s_test, bid_test),
               scheme: interval metrics} for schemes none / pred / true
    """
    K = len(runs)
    y_test, s_test, bid_test = runs[0]['y_test'], runs[0]['s_test'], runs[0]['bid_test']
    mu_test = np.stack([r['yp_test'] for r in runs], axis=1)
    resid = np.concatenate([np.abs(r['y_hold'] - r['yp_hold']) for r in runs])
    inner = np.concatenate([np.full(len(r['y_hold']), j) for j, r in enumerate(runs)])
    g_true = np.concatenate([r['s_hold'] for r in runs])
    g_pred = np.concatenate([r['shat_hold'] for r in runs])
    # test stage: average the K stage classifiers' probabilities
    shat_test = np.mean([r['pstage_test'] for r in runs], axis=0).argmax(1)

    subsets = stage_subsets(s_test)
    out = {'point': (y_test, mu_test.mean(1), s_test, bid_test),
           'stage_acc_test': float(np.mean(shat_test == s_test)),
           'nf_recall_test': float(np.mean(shat_test[s_test == 2] == 2)) if (s_test == 2).any() else np.nan}
    schemes = {'none': (np.zeros_like(g_true), np.zeros_like(s_test)),
               'pred': (g_pred, shat_test),
               'true': (g_true, s_test)}
    for name, (gc, gt) in schemes.items():
        lo, hi = cvplus_intervals(mu_test, resid, inner, gc, gt, alpha)
        out[name] = interval_metrics(y_test, lo, hi, subsets)
    return out
