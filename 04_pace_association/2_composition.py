#!/usr/bin/env python3
import os, pickle, numpy as np, pandas as pd
from scipy import stats
B = '${PROJECT_DIR}/results'; D = f'{B}/PAPER_CRE_MECHANISM'; OUR = f'{B}/phase3b_ourcd4'
OUT = '${PAPER_DIR}/results/composition'; os.makedirs(OUT, exist_ok=True)
rs = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv')
ann = pd.read_csv(f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv', usecols=['cre_id', 'proximity']).drop_duplicates('cre_id')
prox = dict(zip(ann.cre_id, ann.proximity))
f3 = pd.read_csv(f'{D}/results/srcdata_fig2f_fig3i_canonical/Fig3i.csv', usecols=['cohort', 'donor', 'subset', 'age', 'pace', 'pace_adj']).drop_duplicates(['cohort', 'donor', 'subset'])
ts = pd.read_csv(f'{D}/results/tage_v110_check/tage_all_combined_v4.csv', usecols=['Donor_id', 'Batch', 'Age'])
base = ts.drop_duplicates().sort_values(['Donor_id', 'Age']).groupby('Donor_id').first().reset_index(); BK = set(zip(base.Donor_id, base.Batch))
def run(tag, pkl, first_only):
    d = pickle.load(open(pkl, 'rb')); cres = np.array(d['cres']); ci = {c: i for i, c in enumerate(cres)}
    isd = np.array([prox.get(c, '') == 'distal' for c in cres])
    sets = {}
    for ct, g in rs.groupby('cell_type'):
        sets[ct] = {p: np.array([ci[c] for c in g[g.program == p].cre_id if c in ci]) for p in ['up_distal', 'down_proximal']}
        sets[ct]['all_robust'] = np.array([ci[c] for c in g.cre_id if c in ci])
    agg = {}
    for k in d['expr_sum']:
        coh, don, bat, ct = k
        if first_only and (don, bat) not in BK: continue
        if d['n_cells'][k] < 10 or ct not in sets: continue
        v = np.asarray(d['expr_sum'][k], np.float64); u = float(d['umi_sum'][k]); n = d['n_cells'][k]
        s = sets[ct]; mask = np.ones(len(v), bool); mask[s['all_robust']] = False
        rec = np.array([v[s['up_distal']].sum() if len(s['up_distal']) else np.nan, v[s['down_proximal']].sum() if len(s['down_proximal']) else np.nan,
                        v[mask].sum(), v.sum(), v[isd].sum(), u, n])
        agg[(don, ct)] = agg.get((don, ct), 0) + rec
    del d
    rows = []
    for (don, ct), r in agg.items():
        rd, fp, bg, tot, dist, u, n = r
        rows.append(dict(cohort=tag, donor=don, subset=ct, rd_umi=rd / u * 1e6, fp_umi=fp / u * 1e6, rd_bg=rd / bg * 1e6, fp_bg=fp / bg * 1e6,
                         distal_share=dist / tot, distal_share_noFP=dist / (tot - fp) if fp == fp else np.nan, total_umi=u, n_cells=n, tcre_frac=tot / u))
    return pd.DataFrame(rows)
L = pd.concat([run('AIDA', f'{OUR}/aggregation_cache_aida_subcluster.pkl', False), run('Terekhova', f'{OUR}/aggregation_cache_synapse_subcluster.pkl', True)])
L = L.merge(f3, on=['cohort', 'donor', 'subset'], how='inner'); L.to_csv(f'{OUT}/R1_long.csv', index=False)
def pcor(x, y, C):
    Z = pd.DataFrame({'x': x, 'y': y, **{f'c{i}': c for i, c in enumerate(C)}}).replace([np.inf, -np.inf], np.nan).dropna(); R = Z.rank()
    X = np.column_stack([np.ones(len(R))] + [R[f'c{i}'] for i in range(len(C))])
    rx = R.x - X @ np.linalg.lstsq(X, R.x, rcond=None)[0]; ry = R.y - X @ np.linalg.lstsq(X, R.y, rcond=None)[0]
    return stats.pearsonr(rx, ry)[0], len(Z)
rows = []
for (c, s), g in L.groupby(['cohort', 'subset']):
    for m in ['rd_umi', 'fp_umi', 'rd_bg', 'fp_bg', 'distal_share', 'distal_share_noFP', 'tcre_frac']:
        x = np.log(g[m].clip(lower=1e-9)) if m not in ('distal_share', 'distal_share_noFP', 'tcre_frac') else g[m]
        if x.isna().all(): continue
        r, n = pcor(x, g.pace_adj, [np.log(g.total_umi), np.log(g.n_cells)])
        rows.append(dict(cohort=c, subset=s, metric=m, n=n, rho=stats.spearmanr(x, g.pace_adj, nan_policy='omit')[0], partial_rho=r,
                         rho_age=stats.spearmanr(x, g.age, nan_policy='omit')[0]))
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/R1_stats.csv', index=False)
S = R.groupby(['cohort', 'metric']).agg(n_ct=('partial_rho', 'size'), n_pos=('partial_rho', lambda v: int((v > 0).sum())), median_partial=('partial_rho', 'median'),
                                         median_rho=('rho', 'median'), median_rho_age=('rho_age', 'median')).reset_index()
S.to_csv(f'{OUT}/R1_summary.csv', index=False); print(S.round(3).to_string()); print(R.round(3).to_string())


import pandas as pd, numpy as np, os
from scipy import stats
L = pd.read_csv('${PAPER_DIR}/results/composition/R1_long.csv')
L['rd_cell'] = L.rd_umi * L.total_umi / 1e6 / L.n_cells; L['fp_cell'] = L.fp_umi * L.total_umi / 1e6 / L.n_cells; L['tot_cell'] = L.total_umi / L.n_cells
def pc(g, m):
    Z = g[[m, 'pace_adj', 'tot_cell', 'n_cells']].replace([np.inf, -np.inf], np.nan).dropna(); Z = Z[Z[m] > 0]
    R = pd.DataFrame({'x': np.log(Z[m]), 'y': Z.pace_adj, 'c1': np.log(Z.tot_cell), 'c2': np.log(Z.n_cells)}).rank()
    X = np.column_stack([np.ones(len(R)), R.c1, R.c2]); f = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    return stats.pearsonr(f(R.x), f(R.y))[0], stats.spearmanr(np.log(Z[m]), Z.pace_adj)[0]
rows = []
for (c, s), g in L.groupby(['cohort', 'subset']):
    for m in ['rd_cell', 'fp_cell', 'rd_bg', 'distal_share_noFP']:
        if g[m].notna().sum() < 20: continue
        p, r = pc(g, m) if m != 'distal_share_noFP' else (np.nan, stats.spearmanr(g[m], g.pace_adj, nan_policy='omit')[0])
        rows.append(dict(cohort=c, subset=s, metric=m, partial=p, rho=r))
R = pd.DataFrame(rows); R.to_csv('${PAPER_DIR}/results/composition/R1b_stats.csv', index=False)
print(R.groupby(['cohort', 'metric']).agg(n=('rho', 'size'), pos_partial=('partial', lambda v: int((v > 0).sum())), pos_rho=('rho', lambda v: int((v > 0).sum())),
      med_partial=('partial', 'median'), med_rho=('rho', 'median'), min_rho=('rho', 'min'), max_rho=('rho', 'max')).round(3).to_string())


import numpy as np, pandas as pd, os
B = '${PROJECT_DIR}/results'; D = f'{B}/PAPER_CRE_MECHANISM'
OUT = '${PAPER_DIR}/results/composition'; os.makedirs(OUT, exist_ok=True)
rs = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv')
info = pd.read_csv(f'{D}/results/shared_assets/tcre_info.csv').drop_duplicates('cre_id').set_index('cre_id')
rs['near_gene'] = rs.cre_id.map(info.near_gene)
p = rs.cre_id.str.rsplit('_', n=3, expand=True); rs['chrom'] = p[0]; rs['mid'] = (p[1].astype(int) + p[2].astype(int)) // 2
print(rs.program.value_counts().to_dict(), len(rs))

def both(df):
    n = 0
    for ct, g in df.groupby('cell_type'):
        a = set(g[g.program == 'up_distal'].near_gene.dropna()); b = set(g[g.program == 'down_proximal'].near_gene.dropna()); n += len(a & b)
    return n
obs = both(rs); rng = np.random.default_rng(0); null = []
for _ in range(1000):
    s = rs.copy(); s['program'] = s.groupby('cell_type').program.transform(lambda x: rng.permutation(x.values)); null.append(both(s))
print('3b both-programme gene x cell type: observed', obs, 'permutation expected', np.mean(null), '95% range', np.percentile(null, [2.5, 97.5]), 'p', (np.sum(np.array(null) <= obs) + 1) / 1001)

def near(df, test, W=100_000):
    hits = []
    for ct, g in df.groupby('cell_type'):
        rd = g[g.program == 'up_distal']
        for _, r in g[g.program == test].iterrows():
            x = rd[(rd.chrom == r.chrom) & ((rd.mid - r.mid).abs() <= W) & (rd.cre_id != r.cre_id)]
            hits.append(len(x) > 0)
    return np.array(hits)
for t in ['down_proximal', 'up_proximal']:
    h = near(rs, t); k = h.sum(); n = len(h)
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, method='wilson'); print(f'3c {t}: {k}/{n} = {k/n:.3f} (95% CI {lo:.3f}-{hi:.3f})')
from scipy.stats import fisher_exact
a = near(rs, 'down_proximal'); b = near(rs, 'up_proximal'); print('Fisher', fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]))
