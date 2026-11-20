"""Per-bearing coverage of the CV+ run and two variants (offline, no retraining).

1. Coverage per test bearing (each bearing is tested once over the 5 outer
   folds), mean over seeds: shows whether a few bearings fail while the
   average looks fine.
2. Bearing-balanced CV+: each calibration window weighted 1 / (windows of its
   bearing within its group), so one long bearing (e.g. Bearing3_2) cannot
   dominate a quantile (weighted conformal, Tibshirani et al. 2019).
3. For the multi-task configs: groups from the models' OWN stage heads instead
   of the shared stage classifier.

Usage: python3 -I cvplus_per_bearing.py <repo_src_dir> <predictions_dir> <out_md>
"""
import sys
import numpy as np

SRC, PRED, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, SRC)
from cvplus import cvplus_intervals, ALL_BEARINGS, K_INNER  # noqa: E402

ALPHA = 0.10
SEEDS = [1, 2, 3, 4, 5]
OUTER = [1, 2, 3, 4, 5]
TAGS = {'baseline': 'nogan_st', '+GAN': 'gan_st', '+MT': 'nogan_mt', '+GAN+MT': 'gan_mt'}


def load(tag, k, j, s):
    z = np.load(f'{PRED}/{tag}_outer{k}_inner{j}_s{s}.npz', allow_pickle=False)
    return {key: z[key] for key in z}


def weighted_cvplus(mu_test, R, J, gc, gt, w, alpha, chunk=256):
    """CV+ with calibration weights w (weighted conformal quantiles).

    The test point carries the mean weight of its group and sits at +/- inf,
    which keeps the finite-sample correction of the unweighted version.
    """
    m = len(mu_test)
    lo = np.full(m, -np.inf); hi = np.full(m, np.inf)
    for g in np.unique(gt):
        ti = np.flatnonzero(gt == g); ci = np.flatnonzero(gc == g)
        if len(ci) == 0:
            continue
        Rg, Jg, wg = R[ci], J[ci], w[ci]
        p = wg / (wg.sum() + wg.mean())          # last share = test point at inf
        for a in range(0, len(ti), chunk):
            t = ti[a:a + chunk]
            mu = mu_test[t][:, Jg]
            for side, vals in (('hi', mu + Rg), ('lo', mu - Rg)):
                order = np.argsort(vals, axis=1)
                v = np.take_along_axis(vals, order, 1)
                cp = np.cumsum(p[order], axis=1)
                if side == 'hi':                    # smallest v with cum >= 1 - alpha
                    idx = (cp < 1 - alpha - 1e-9).sum(1)
                    ok = idx < v.shape[1]
                    hi[t] = np.where(ok, v[np.arange(len(t)), np.minimum(idx, v.shape[1] - 1)], np.inf)
                else:                               # mirror: largest v with upper-cum >= 1 - alpha
                    up = p.sum() - (cp - p[order])  # calibration mass at or above each value
                    idx = (up >= 1 - alpha - 1e-9).sum(1) - 1
                    ok = idx >= 0
                    lo[t] = np.where(ok, v[np.arange(len(t)), np.maximum(idx, 0)], -np.inf)
    return lo, hi


def outer_eval(tag, k, s, schemes):
    runs = [load(tag, k, j, s) for j in range(1, K_INNER + 1)]
    y, st, bid = runs[0]['y_test'], runs[0]['s_test'], runs[0]['bid_test']
    mu = np.stack([r['yp_test'] for r in runs], 1)
    R = np.concatenate([np.abs(r['y_hold'] - r['yp_hold']) for r in runs])
    J = np.concatenate([np.full(len(r['y_hold']), j) for j, r in enumerate(runs)])
    bh = np.concatenate([r['bid_hold'] for r in runs])
    groups = {
        'none': (np.zeros(len(R), int), np.zeros(len(y), int)),
        'pred': (np.concatenate([r['shat_hold'] for r in runs]),
                 np.mean([r['pstage_test'] for r in runs], 0).argmax(1)),
        'true': (np.concatenate([r['s_hold'] for r in runs]), st),
    }
    if 'own_pstage_hold' in runs[0]:
        groups['own'] = (np.concatenate([r['own_pstage_hold'].argmax(1) for r in runs]),
                         np.mean([r['own_pstage_test'] for r in runs], 0).argmax(1))
    out = {}
    for name in schemes:
        if name.endswith('_bw'):
            base = name[:-3]
            if base not in groups:
                continue
            gc, gt = groups[base]
            # weight = 1 / windows of the same bearing in the same group
            w = np.ones(len(R))
            for g in np.unique(gc):
                for b in np.unique(bh[gc == g]):
                    sel = (gc == g) & (bh == b)
                    w[sel] = 1.0 / sel.sum()
            lo, hi = weighted_cvplus(mu, R, J, gc, gt, w, ALPHA)
        else:
            if name not in groups:
                continue
            gc, gt = groups[name]
            lo, hi = cvplus_intervals(mu, R, J, gc, gt, ALPHA)
        cov = (y >= lo) & (y <= hi)
        wid = np.clip(hi, 0, 1) - np.clip(lo, 0, 1)
        out[name] = (cov, wid)
    return y, st, bid, out


SCHEMES = ['none', 'pred', 'true', 'own', 'pred_bw', 'own_bw']
# per config: scheme -> bearing -> list over seeds of (cov_all, cov_nf, width_all, n, n_nf)
per = {c: {sc: {} for sc in SCHEMES} for c in TAGS}
agg = {c: {sc: [] for sc in SCHEMES} for c in TAGS}   # per seed: mean over outer folds
for c, tag in TAGS.items():
    for s in SEEDS:
        fold_stats = {sc: [] for sc in SCHEMES}
        for k in OUTER:
            y, st, bid, out = outer_eval(tag, k, s, SCHEMES)
            for sc, (cov, wid) in out.items():
                nf = st == 2
                fold_stats[sc].append((cov.mean(), cov[nf].mean() if nf.any() else np.nan,
                                       cov[st >= 1].mean(), wid.mean(), wid[nf].mean() if nf.any() else np.nan))
                for b in np.unique(bid):
                    m = bid == b
                    mn = m & nf
                    per[c][sc].setdefault(int(b), []).append(
                        (cov[m].mean(), cov[mn].mean() if mn.any() else np.nan, wid[m].mean(), int(m.sum()), int(mn.sum())))
        for sc in SCHEMES:
            if fold_stats[sc]:
                agg[c][sc].append(np.nanmean(np.array(fold_stats[sc]), 0))

LAB = {'none': 'single q', 'pred': 'predicted stage (shared classifier)', 'true': 'true stage (oracle)',
       'own': "predicted stage (model's own head)", 'pred_bw': 'predicted stage, bearing-balanced',
       'own_bw': "own head, bearing-balanced"}
L = ['# CV+ per-bearing coverage and variants (alpha = 0.10, 5 seeds, offline)\n']
L.append('## Averages (mean ± std over seeds of the fold means)\n')
L.append('PICP all / PICP near-failure / PICP post-FPT / width all / width near-failure (clipped)\n')
L.append('| variant | ' + ' | '.join(TAGS) + ' |')
L.append('|---|' + '---|' * len(TAGS))
for sc in SCHEMES:
    cells = []
    for c in TAGS:
        a = np.array(agg[c][sc])
        if len(a) == 0:
            cells.append('—'); continue
        m, sd = a.mean(0), a.std(0)
        cells.append(f'{m[0]:.3f}±{sd[0]:.3f} / {m[1]:.3f} / {m[2]:.3f} / {m[3]:.2f} / {m[4]:.2f}')
    L.append(f'| {LAB[sc]} | ' + ' | '.join(cells) + ' |')

for c in ('+GAN+MT', 'baseline'):
    L.append(f'\n## Per test bearing — {c} (mean over 5 seeds)\n')
    L.append('Coverage of all windows / of near-failure windows; n = windows (near-failure)\n')
    show = [sc for sc in ('none', 'pred', 'true', 'pred_bw') if per[c][sc]]
    L.append('| bearing | n (NF) | ' + ' | '.join(LAB[sc] for sc in show) + ' |')
    L.append('|---|---|' + '---|' * len(show))
    for b in sorted(per[c]['pred']):
        rows = {sc: np.array(per[c][sc][b], float) for sc in show}
        n, nn = int(rows['pred'][0, 3]), int(rows['pred'][0, 4])
        cells = []
        for sc in show:
            r = rows[sc]
            nf = np.nanmean(r[:, 1]) if nn else np.nan
            cells.append(f'{np.mean(r[:, 0]):.2f} / ' + (f'{nf:.2f}' if nn else '—'))
        L.append(f'| {ALL_BEARINGS[b]} | {n} ({nn}) | ' + ' | '.join(cells) + ' |')

open(OUT, 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
