"""Does C3 under-cover because each fold calibrates on only 2 bearings?

Sensitivity check from saved predictions (no retraining). For test fold k,
calibration residuals are taken from the fold's own 2 calibration bearings
PLUS the test bearings of the other 4 folds, each predicted by its own fold
model (out-of-fold residuals, 14 bearings in total). This mimics a
cross-conformal / CV+ calibration; it is a diagnostic, not the final method.

Usage: python3 -I c3_calsize.py <predictions_dir> <out_md>
"""
import sys
import numpy as np

PRED, OUT = sys.argv[1], sys.argv[2]
ALPHA = 0.10
CONFIGS = {'baseline': 'nogan_st', '+GAN': 'gan_st', '+MT': 'nogan_mt', '+GAN+MT': 'gan_mt'}


def load(tag, f, s):
    z = np.load(f'{PRED}/{tag}_f{f}_s{s}.npz', allow_pickle=False)
    return {k: z[k] for k in z}


def qhat(r):
    n = len(r)
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    return np.inf if (n == 0 or k > n) else float(np.sort(r)[k - 1])


def evaluate(tag, scheme, grouping):
    per_seed = []
    for s in range(1, 6):
        D = {f: load(tag, f, s) for f in range(1, 6)}
        folds = []
        for k in range(1, 6):
            d = D[k]
            r_c = [np.abs(d['y_cal'] - d['yp_cal'])]
            g_c = [d['s_cal'] if grouping == 'true' else d['s_cal_hat']]
            if scheme == 'pooled':
                for j in range(1, 6):
                    if j != k:
                        r_c.append(np.abs(D[j]['y_test'] - D[j]['yp_test']))
                        g_c.append(D[j]['s_test'] if grouping == 'true' else D[j]['s_test_hat'])
            r_c, g_c = np.concatenate(r_c), np.concatenate(g_c)
            if grouping == 'none':
                g_c = np.zeros_like(g_c)
            g_t = {'true': d['s_test'], 'pred': d['s_test_hat'], 'none': np.zeros_like(d['s_test'])}[grouping]
            q = {g: qhat(r_c[g_c == g]) for g in np.unique(g_c)}
            qt = np.array([q.get(g, np.inf) for g in g_t])
            pt, yt, st = d['yp_test'], d['y_test'], d['s_test']
            cov = np.abs(yt - pt) <= qt
            wc = np.minimum(1, pt + qt) - np.maximum(0, pt - qt)
            folds.append({n: (cov[m].mean(), wc[m].mean())
                          for n, m in (('nf', st == 2), ('overall', np.ones(len(st), bool)))})
        per_seed.append({n: np.mean([f[n] for f in folds], axis=0) for n in ('nf', 'overall')})
    return {n: (np.mean([p[n] for p in per_seed], 0), np.std([p[n] for p in per_seed], 0))
            for n in ('nf', 'overall')}


L = ['# Calibration-size check: 2 calibration bearings vs 14 (out-of-fold pooled)\n',
     'PICP±std / clipped MPIW, alpha = 0.10\n']
for subset, lab in (('nf', 'near-failure windows'), ('overall', 'all test windows')):
    L.append(f'\n## {lab}\n')
    L.append('| calibration | grouping | ' + ' | '.join(CONFIGS) + ' |')
    L.append('|---|---|' + '---|' * len(CONFIGS))
    for scheme, sl in (('own', '2 bearings (current)'), ('pooled', '14 bearings (pooled OOF)')):
        for grouping, gl in (('none', 'single q'), ('true', 'true stage'), ('pred', 'predicted stage')):
            cells = []
            for c, tag in CONFIGS.items():
                m, sd = evaluate(tag, scheme, grouping)[subset]
                cells.append(f'{m[0]:.3f}±{sd[0]:.3f} / {m[1]:.2f}')
            L.append(f'| {sl} | {gl} | ' + ' | '.join(cells) + ' |')
open(OUT, 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
