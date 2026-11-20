"""Numbers for the paper's results section, from the CV+ predictions (notebook 09).

Deployable grouping = the stage predicted from the window: the multi-task
model's own stage head (+MT, +GAN+MT) or the shared stage classifier
(baseline, +GAN, which have no stage head).

Usage: python3 -I paper_tables.py <repo_src_dir> <predictions_dir> <out_json>
"""
import sys, json
import numpy as np

SRC, PRED, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, SRC)
from cvplus import cvplus_intervals, ALL_BEARINGS, K_INNER  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]; OUTER = [1, 2, 3, 4, 5]
TAGS = {'baseline': 'nogan_st', '+GAN': 'gan_st', '+MT': 'nogan_mt', '+GAN+MT': 'gan_mt'}
ALPHAS = [0.05, 0.10, 0.15, 0.20]
SUBS = ['healthy', 'early', 'nearfail', 'postFPT', 'overall']


def load(tag, k, j, s):
    z = np.load(f'{PRED}/{tag}_outer{k}_inner{j}_s{s}.npz', allow_pickle=False)
    return {key: z[key] for key in z}


def subsets(st):
    return {'healthy': st == 0, 'early': st == 1, 'nearfail': st == 2,
            'postFPT': st >= 1, 'overall': np.ones(len(st), bool)}


res = {}
per_bearing = {}
for c, tag in TAGS.items():
    mt = tag.endswith('_mt')
    seed_rows = []
    for s in SEEDS:
        folds = []
        for k in OUTER:
            runs = [load(tag, k, j, s) for j in range(1, K_INNER + 1)]
            y, st, bid = runs[0]['y_test'], runs[0]['s_test'], runs[0]['bid_test']
            mu = np.stack([r['yp_test'] for r in runs], 1)
            R = np.concatenate([np.abs(r['y_hold'] - r['yp_hold']) for r in runs])
            J = np.concatenate([np.full(len(r['y_hold']), j) for j, r in enumerate(runs)])
            if mt:
                gc_pred = np.concatenate([r['own_pstage_hold'].argmax(1) for r in runs])
                gt_pred = np.mean([r['own_pstage_test'] for r in runs], 0).argmax(1)
            else:
                gc_pred = np.concatenate([r['shat_hold'] for r in runs])
                gt_pred = np.mean([r['pstage_test'] for r in runs], 0).argmax(1)
            groups = {'single': (np.zeros(len(R), int), np.zeros(len(y), int)),
                      'pred': (gc_pred, gt_pred),
                      'true': (np.concatenate([r['s_hold'] for r in runs]), st)}
            row = {'stage_acc': float(np.mean(gt_pred == st)),
                   'nf_recall': float(np.mean(gt_pred[st == 2] == 2)),
                   'nf_precision': float(np.mean(st[gt_pred == 2] == 2)) if (gt_pred == 2).any() else np.nan}
            sub = subsets(st)
            for g, (gc, gt) in groups.items():
                for a in ALPHAS:
                    if g != 'pred' and a != 0.10:
                        continue
                    lo, hi = cvplus_intervals(mu, R, J, gc, gt, a)
                    cov = (y >= lo) & (y <= hi)
                    wid = np.clip(hi, 0, 1) - np.clip(lo, 0, 1)
                    for n, m in sub.items():
                        if m.any():
                            row[f'{g}_a{a:.2f}_{n}_picp'] = float(cov[m].mean())
                            row[f'{g}_a{a:.2f}_{n}_mpiw'] = float(wid[m].mean())
                    if g == 'pred' and a == 0.10:
                        for b in np.unique(bid):
                            m = bid == b; mn = m & (st == 2)
                            per_bearing.setdefault(c, {}).setdefault(ALL_BEARINGS[b], []).append(
                                [float(cov[m].mean()), float(cov[mn].mean()) if mn.any() else np.nan,
                                 float(wid[m].mean()), int(m.sum()), int(mn.sum())])
            folds.append(row)
        keys = set().union(*folds)
        seed_rows.append({key: float(np.nanmean([f.get(key, np.nan) for f in folds])) for key in keys})
    res[c] = {key: [float(np.mean([r[key] for r in seed_rows])), float(np.std([r[key] for r in seed_rows]))]
              for key in seed_rows[0]}

pb = {c: {b: np.nanmean(np.array(v, float), 0).tolist() for b, v in d.items()} for c, d in per_bearing.items()}
json.dump({'agg': res, 'per_bearing': pb}, open(OUT, 'w'), indent=1)
for c in TAGS:
    r = res[c]
    print(c, 'acc %.3f recall %.3f prec %.3f' % (r['stage_acc'][0], r['nf_recall'][0], r['nf_precision'][0]))
    for g in ('single', 'pred', 'true'):
        print('  ', g, ' '.join(f"{n}:{r[f'{g}_a0.10_{n}_picp'][0]:.3f}±{r[f'{g}_a0.10_{n}_picp'][1]:.3f}/{r[f'{g}_a0.10_{n}_mpiw'][0]:.2f}" for n in SUBS))
    print('   alpha', ' '.join(f"{a}: {r[f'pred_a{a:.2f}_overall_picp'][0]:.3f}/{r[f'pred_a{a:.2f}_postFPT_picp'][0]:.3f}/{r[f'pred_a{a:.2f}_nearfail_picp'][0]:.3f} w{r[f'pred_a{a:.2f}_overall_mpiw'][0]:.2f}" for a in ALPHAS))
print('per-bearing +GAN+MT:')
for b, v in pb['+GAN+MT'].items():
    print(f'  {b}: all {v[0]:.2f} nf {v[1]:.2f} w {v[2]:.2f} n {int(v[3])} nnf {int(v[4])}')
