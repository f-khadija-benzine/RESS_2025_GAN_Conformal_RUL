"""
conformal.py — Split Conformal Prediction for RUL intervals (Step 7, C3)

Distribution-free, coverage-guaranteed prediction intervals around the point
RUL predictions. On exchangeable data, the interval

    [ y_hat - q,  y_hat + q ]

with q the finite-sample-corrected (1-alpha) quantile of the calibration
absolute residuals, satisfies  P(y in interval) >= 1 - alpha  marginally.

Three things this module is careful about:

  1. Calibration residuals come from REAL calibration windows only. Synthetic
     windows never enter calibration — `assert_real_calibration` guards this.
  2. The quantile uses the finite-sample correction ceil((n+1)(1-alpha))/n,
     not the plain (1-alpha) quantile. The +1 is what delivers the guarantee.
  3. Per-stage PICP and MPIW are reported, not just the global figures, so a
     near-failure coverage failure cannot hide behind a healthy-dominated
     average. Near-failure is where calibration/test exchangeability is most
     strained, so per-stage coverage is the honest diagnostic.

Coverage (PICP) must always be read ALONGSIDE width (MPIW): a narrower
interval that under-covers is worse, not better. Comparisons of MPIW across
configurations are only meaningful at matched PICP.
"""

import numpy as np


def conformal_quantile(residuals, alpha):
    """Finite-sample-corrected (1 - alpha) quantile of calibration residuals.

    Uses the k-th smallest residual with k = ceil((n + 1)(1 - alpha)), which
    is the value that gives split conformal its finite-sample coverage
    guarantee. If k > n (possible when n is small relative to 1/alpha), the
    interval is unbounded in principle; we return the max residual and flag it.

    Args:
        residuals: (n,) non-negative calibration residuals |y - y_hat|
        alpha: miscoverage level (e.g. 0.1 for 90% intervals)

    Returns:
        q: the half-width
        exact: bool, False if n too small for an exact finite-sample quantile
    """
    r = np.sort(np.asarray(residuals, dtype=float))
    n = len(r)
    if n == 0:
        return float('nan'), False
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        # not enough calibration points for an exact guarantee at this alpha
        return float(r[-1]), False
    return float(r[k - 1]), True     # k-th smallest (1-indexed)


def assert_real_calibration(n_cal_expected, y_cal, tol=0):
    """Guard: the calibration set must be exactly the real calibration windows.

    Any augmentation must touch training only. If the calibration tensor's
    length differs from what prepare_fold produced, synthetic (or otherwise
    altered) data may have leaked in, which silently invalidates every
    coverage number. Call this immediately before calibrating.
    """
    n = len(y_cal)
    if abs(n - n_cal_expected) > tol:
        raise AssertionError(
            f"Calibration set has {n} windows, expected {n_cal_expected} real "
            f"windows. Synthetic data may have leaked into calibration — this "
            f"invalidates the coverage guarantee. Augmentation must touch "
            f"training only.")


def calibrate_and_evaluate(y_cal_true, y_cal_pred,
                           y_test_true, y_test_pred, test_stages,
                           alpha=0.1):
    """Full split-conformal pass for one fold at one alpha.

    Calibrates q on the calibration residuals, forms intervals on the test
    set, and reports PICP (coverage) and MPIW (mean width) overall and per
    stage. All widths are on the normalised RUL scale.

    Args:
        y_cal_true, y_cal_pred:   (n_cal,) real calibration RUL + predictions
        y_test_true, y_test_pred: (n_test,) test RUL + predictions
        test_stages: (n_test,) 0-indexed stage labels for the test windows
        alpha: miscoverage level

    Returns:
        dict: q, exact, and per-subset {picp, mpiw, n} for
              healthy/early/nearfail/postFPT/overall
    """
    cal_res = np.abs(np.asarray(y_cal_true) - np.asarray(y_cal_pred))
    q, exact = conformal_quantile(cal_res, alpha)

    yt = np.asarray(y_test_true, float)
    yp = np.asarray(y_test_pred, float)
    st = np.asarray(test_stages)

    lo, hi = yp - q, yp + q
    covered = (yt >= lo) & (yt <= hi)
    width = hi - lo                       # = 2q everywhere, but kept general

    subsets = [('healthy', [0]), ('early', [1]), ('nearfail', [2]),
               ('postFPT', [1, 2]), ('overall', [0, 1, 2])]
    out = {'q': q, 'exact': exact, 'alpha': alpha}
    for name, vals in subsets:
        sel = np.isin(st, vals)
        n = int(sel.sum())
        if n == 0:
            out[name] = {'picp': float('nan'), 'mpiw': float('nan'), 'n': 0}
        else:
            out[name] = {'picp': float(covered[sel].mean()),
                         'mpiw': float(width[sel].mean()), 'n': n}
    return out


def average_conformal(results):
    """Average per-fold conformal results (mean PICP, MPIW per subset)."""
    subsets = ['healthy', 'early', 'nearfail', 'postFPT', 'overall']
    out = {'q': float(np.nanmean([r['q'] for r in results]))}
    for name in subsets:
        out[name] = {
            'picp': float(np.nanmean([r[name]['picp'] for r in results])),
            'mpiw': float(np.nanmean([r[name]['mpiw'] for r in results])),
            'n': int(np.sum([r[name]['n'] for r in results])),
        }
    return out


def print_conformal(res, title='', target=None):
    """Pretty-print a conformal result. `target` = 1-alpha for a coverage ref."""
    if title:
        print(title)
    if target is not None:
        print(f"  (target coverage {target:.0%})")
    print(f"  {'subset':10s} {'n':>6s} {'PICP':>8s} {'MPIW':>8s}")
    print(f"  {'-'*36}")
    for name in ['healthy', 'early', 'nearfail', 'postFPT', 'overall']:
        s = res[name]
        print(f"  {name:10s} {s['n']:6d} {s['picp']:8.3f} {s['mpiw']:8.4f}")


# ── Stage-conditional (Mondrian) conformal ───────────────────────────

def calibrate_and_evaluate_mondrian(y_cal_true, y_cal_pred, cal_stages,
                                    y_test_true, y_test_pred, test_stages,
                                    alpha=0.1):
    """Split conformal with a SEPARATE quantile per degradation stage.

    Plain split conformal uses one q for all windows, so its coverage
    guarantee is marginal: it holds on average but can fail within a stage
    when the calibration and test stage-mixes differ, or when one stage's
    residuals are much larger than another's. Here the healthy windows are
    predicted poorly (the model under-predicts a constant RUL=1.0 target),
    inflating and destabilising a shared q while still under-covering.

    Mondrian conformal partitions the calibration set by stage and computes
    q_s from stage s's residuals alone, then applies q_s to test windows of
    stage s. This restores a coverage guarantee *within each stage* — in
    particular the near-failure stage, which is the operationally important
    one. It is a standard, published conformal variant, not a workaround.

    Args:
        y_cal_true, y_cal_pred: (n_cal,) real calibration RUL + predictions
        cal_stages: (n_cal,) 0-indexed stage labels for calibration windows
        y_test_true, y_test_pred, test_stages: test equivalents
        alpha: miscoverage level

    Returns:
        dict: per-subset {picp, mpiw, n, q}. q is now stage-specific; the
              'postFPT' and 'overall' rows aggregate windows that were each
              covered by their own stage's q, so their reported q is the mean.
    """
    yct = np.asarray(y_cal_true, float); ycp = np.asarray(y_cal_pred, float)
    cst = np.asarray(cal_stages)
    ytt = np.asarray(y_test_true, float); ytp = np.asarray(y_test_pred, float)
    tst = np.asarray(test_stages)

    # one q per stage, from that stage's calibration residuals
    q_by_stage = {}
    for s in (0, 1, 2):
        m = cst == s
        if m.sum() == 0:
            q_by_stage[s] = float('nan')
            continue
        res_s = np.abs(yct[m] - ycp[m])
        q_by_stage[s], _ = conformal_quantile(res_s, alpha)

    # apply each test window's own stage q
    q_test = np.array([q_by_stage.get(int(s), np.nan) for s in tst])
    covered = np.abs(ytt - ytp) <= q_test
    width = 2 * q_test

    subsets = [('healthy', [0]), ('early', [1]), ('nearfail', [2]),
               ('postFPT', [1, 2]), ('overall', [0, 1, 2])]
    out = {'q_by_stage': q_by_stage, 'alpha': alpha}
    for name, vals in subsets:
        sel = np.isin(tst, vals)
        n = int(sel.sum())
        if n == 0:
            out[name] = {'picp': float('nan'), 'mpiw': float('nan'),
                         'n': 0, 'q': float('nan')}
        else:
            out[name] = {'picp': float(covered[sel].mean()),
                         'mpiw': float(np.nanmean(width[sel])),
                         'n': n, 'q': float(np.nanmean(q_test[sel]))}
    return out


def average_conformal_mondrian(results):
    """Average per-fold Mondrian results (mean PICP, MPIW per subset)."""
    subsets = ['healthy', 'early', 'nearfail', 'postFPT', 'overall']
    out = {}
    for name in subsets:
        out[name] = {
            'picp': float(np.nanmean([r[name]['picp'] for r in results])),
            'mpiw': float(np.nanmean([r[name]['mpiw'] for r in results])),
            'q': float(np.nanmean([r[name]['q'] for r in results])),
            'n': int(np.sum([r[name]['n'] for r in results])),
        }
    return out
