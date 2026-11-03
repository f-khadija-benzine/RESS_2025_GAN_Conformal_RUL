"""Offline C3 analysis from saved predictions (no retraining).

Usage: python3 -I c3_offline.py <predictions_dir> <out_dir>
"""
import sys, glob, os, re, json
import numpy as np

PRED, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
ALPHA = 0.10
CONFIGS = {'baseline': 'nogan_st', '+GAN': 'gan_st', '+MT': 'nogan_mt', '+GAN+MT': 'gan_mt'}


def load(tag, f, s):
    z = np.load(f'{PRED}/{tag}_f{f}_s{s}.npz', allow_pickle=False)
    return {k: z[k] for k in z}


def segments(y, st):
    """Bearing boundaries: the true stage drops or the RUL jumps up."""
    b = np.zeros(len(y), bool); b[0] = True
    b[1:] = (st[1:] < st[:-1]) | (y[1:] > y[:-1] + 0.2)
    idx = np.flatnonzero(b)
    return list(zip(idx, list(idx[1:]) + [len(y)]))


def ratchet(pred, segs, k):
    """Causal, monotone stage: move up to stage j only after the last k
    predictions are all >= j; never move back. Uses past windows only."""
    out = np.empty_like(pred)
    for a, b in segs:
        state = 0
        for t in range(a, b):
            lo = max(a, t - k + 1)
            if t - lo + 1 == k:
                state = max(state, int(pred[lo:t + 1].min()))
            out[t] = state
    return out


def qhat(r, alpha=ALPHA):
    n = len(r)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(np.sort(r)[k - 1])


def conformal(yc, pc, gc, yt, pt, gt, groups):
    """Mondrian over `groups`; returns covered flags, raw and clipped widths."""
    q = {g: qhat(np.abs(yc[gc == g] - pc[gc == g])) for g in groups}
    qt = np.array([q.get(g, np.nan) for g in gt])
    cov = np.abs(yt - pt) <= qt
    w_raw = 2 * qt
    w_clip = np.minimum(1, pt + qt) - np.maximum(0, pt - qt)
    return cov, w_raw, w_clip, q


def rul_bins(p, edges):
    return np.digitize(p, edges)


# ---------- stage-classifier diagnosis (shared across configs) ----------
conf_mat = np.zeros((3, 3), int)
for f in range(1, 6):
    for s in range(1, 6):
        d = load('gan_mt', f, s)
        for t, p in zip(d['s_test'], d['s_test_hat']):
            conf_mat[t, p] += 1

# ratchet effect on stage accuracy
def stage_metrics(variant):
    acc, rec_nf, prec_nf = [], [], []
    for f in range(1, 6):
        for s in range(1, 6):
            d = load('gan_mt', f, s)
            g = variant(d, 'test')
            st = d['s_test']
            acc.append(np.mean(g == st))
            rec_nf.append(np.mean(g[st == 2] == 2))
            prec_nf.append(np.mean(st[g == 2] == 2) if (g == 2).any() else np.nan)
    return np.mean(acc), np.mean(rec_nf), np.nanmean(prec_nf)


def v_raw(d, part):
    return d[f's_{part}_hat']

def make_ratchet(k):
    def v(d, part):
        segs = segments(d[f'y_{part}'], d[f's_{part}'])
        return ratchet(d[f's_{part}_hat'], segs, k)
    return v

def v_true(d, part):
    return d[f's_{part}']

# ---------- conformal variants ----------
EDGES = {'rul3': [0.3, 0.7], 'rul4': [0.25, 0.5, 0.75], 'rul5': [0.2, 0.4, 0.6, 0.8]}

def variants():
    V = {'single q (no groups)': ('const', None),
         'true stage (oracle)': ('stage', v_true),
         'predicted stage': ('stage', v_raw)}
    for k in (5, 10, 20, 30):
        V[f'predicted stage + ratchet k={k}'] = ('stage', make_ratchet(k))
    for name, e in EDGES.items():
        V[f'predicted-RUL bins {name}'] = ('rul', e)
    return V


def run_variant(kind, arg, tag):
    per_seed = []
    for s in range(1, 6):
        fold_res = []
        for f in range(1, 6):
            d = load(tag, f, s)
            yc, pc, yt, pt, st = d['y_cal'], d['yp_cal'], d['y_test'], d['yp_test'], d['s_test']
            if kind == 'const':
                gc, gt, groups = np.zeros(len(yc), int), np.zeros(len(yt), int), [0]
            elif kind == 'stage':
                gc, gt, groups = arg(d, 'cal'), arg(d, 'test'), [0, 1, 2]
            else:
                gc, gt = rul_bins(pc, arg), rul_bins(pt, arg); groups = list(range(len(arg) + 1))
            cov, wr, wc, _ = conformal(yc, pc, gc, yt, pt, gt, groups)
            r = {}
            for name, sel in (('nf', st == 2), ('postFPT', st >= 1), ('overall', np.ones(len(st), bool))):
                r[name] = (cov[sel].mean(), np.nanmean(wr[sel]), np.nanmean(wc[sel]))
            fold_res.append(r)
        per_seed.append({k: np.mean([fr[k] for fr in fold_res], axis=0) for k in fold_res[0]})
    out = {}
    for k in per_seed[0]:
        a = np.array([p[k] for p in per_seed])
        out[k] = (a.mean(0), a.std(0))
    return out


results = {}
for vname, (kind, arg) in variants().items():
    results[vname] = {c: run_variant(kind, arg, tag) for c, tag in CONFIGS.items()}

stage_rows = {'raw': stage_metrics(v_raw)}
for k in (5, 10, 20, 30):
    stage_rows[f'ratchet k={k}'] = stage_metrics(make_ratchet(k))

# own stage head of the MT models is not saved; the shared classifier is used throughout.

# ---------- report ----------
L = []
L.append('# Offline C3 analysis (100 saved prediction files, no retraining)\n')
L.append('## Shared stage classifier, test windows pooled over folds and seeds\n')
L.append('Confusion matrix (rows = true stage, columns = predicted), row %:\n')
L.append('| true \\ pred | healthy | early | near-failure | n |')
L.append('|---|---|---|---|---|')
for i, nm in enumerate(['healthy', 'early', 'near-failure']):
    row = conf_mat[i] / conf_mat[i].sum() * 100
    L.append(f'| {nm} | {row[0]:.1f} | {row[1]:.1f} | {row[2]:.1f} | {conf_mat[i].sum()} |')
L.append('\n| stage assignment | accuracy | near-failure recall | near-failure precision |')
L.append('|---|---|---|---|')
for k, (a, r, p) in stage_rows.items():
    L.append(f'| {k} | {a:.3f} | {r:.3f} | {p:.3f} |')

for subset, label in (('nf', 'near-failure windows (true stage)'), ('overall', 'all test windows'), ('postFPT', 'post-FPT windows')):
    L.append(f'\n## Coverage / width on {label}, alpha = {ALPHA}\n')
    L.append('PICP | MPIW raw | MPIW clipped to [0,1] — mean ± std over 5 seeds\n')
    L.append('| variant | ' + ' | '.join(CONFIGS) + ' |')
    L.append('|---|' + '---|' * len(CONFIGS))
    for vname, byc in results.items():
        cells = []
        for c in CONFIGS:
            m, sd = byc[c][subset]
            cells.append(f'{m[0]:.3f}±{sd[0]:.3f} / {m[1]:.2f} / {m[2]:.2f}')
        L.append(f'| {vname} | ' + ' | '.join(cells) + ' |')

open(f'{OUT}/c3_offline_report.md', 'w').write('\n'.join(L) + '\n')
json.dump({v: {c: {k: [list(map(float, x[0])), list(map(float, x[1]))] for k, x in byc[c].items()}
               for c in byc} for v, byc in results.items()},
          open(f'{OUT}/c3_offline_results.json', 'w'), indent=1)
print('\n'.join(L))
