#!/usr/bin/env python3
import os, sys, pickle, time
import numpy as np, pandas as pd
from scipy.stats import rankdata, t as tdist
from statsmodels.stats.multitest import multipletests

COHORT = sys.argv[1].lower()
assert COHORT in ('aida','synapse')
B   = '${PROJECT_DIR}/results'
P   = f'{B}/phase3b_ourcd4'
D   = f'{B}/PAPER_CRE_MECHANISM'
OUT = f'{D}/results/fixes/age_adjusted'
CD4 = f'{D}/results/cd4_redo/cd4_our_tage.csv'
SYN_TAGE_SC  = '${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/subcluster/tage_all_combined_v2.csv'
AIDA_TAGE_SC = '${DATA_DIR}/lnc_project_AIDA_dataset/by_subcluster/tage/tage_all_combined_v1.csv'
MIN_CELLS, MIN_SAMPLES = 10, 10
TV = 'yugene_diff_EN_tAge'

def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

def build_mc_meta():
    s = pd.read_csv(SYN_TAGE_SC)
    s = s[(s.model_type=='Chronoage') & (s['mode']=='EN')].copy().rename(columns={'Age':'age'})
    s['cohort'] = 'Synapse'
    pa = s[['Donor_id','Batch','age']].drop_duplicates()
    keep = pa.sort_values(['Donor_id','age']).groupby('Donor_id').head(1)
    s = s.merge(keep[['Donor_id','Batch']], on=['Donor_id','Batch'])
    a = pd.read_csv(AIDA_TAGE_SC)
    a = a[(a.model_type=='Chronoage') & (a['mode']=='EN')].copy()
    a = a.rename(columns={'Sample':'Donor_id','Subcluster':'CellType','sex':'Sex'})
    a['Batch'] = a['Donor_id']; a['cohort'] = 'AIDA'
    cols = ['cohort','Donor_id','Batch','CellType','age','Sex','yugene_diff_EN_tAge']
    comb = pd.concat([a[cols], s[cols]], ignore_index=True)
    comb = comb[~comb.CellType.isin(['CD4_Naive','CD4_Memory'])].copy()
    our = pd.read_csv(CD4)
    o = pd.DataFrame({'cohort':our.cohort,'Donor_id':our.Donor_id,'Batch':our.Batch,
                      'CellType':our.Group,'age':our.age,'Sex':our.Sex,
                      'yugene_diff_EN_tAge':our.yugene_diff_EN_tAge})
    return pd.concat([comb, o[cols]], ignore_index=True)

log(f'cohort = {COHORT}')
mc_meta = build_mc_meta()
mc_meta = mc_meta[mc_meta.cohort == ('AIDA' if COHORT=='aida' else 'Synapse')].copy()
log(f'tAge メタセル行: {len(mc_meta)}')

cache = f'{P}/aggregation_cache_{COHORT}_subcluster.pkl'
log(f'キャッシュ読み込み: {cache}')
t0 = time.time()
with open(cache,'rb') as f:
    agg = pickle.load(f)
log(f'  読み込み {time.time()-t0:.0f} s / tCRE {len(agg["cres"]):,} / メタセル {len(agg["expr_sum"]):,}')
cres = np.array(agg['cres'])
n_cre = len(cres)

keys_agg = set(agg['expr_sum'].keys())
mc_meta['key'] = list(zip(mc_meta.cohort, mc_meta.Donor_id, mc_meta.Batch, mc_meta.CellType))
mc = mc_meta[mc_meta.key.isin(keys_agg) & (mc_meta.CellType!='Other')].copy().reset_index(drop=True)
mc['agg_n_cells'] = [agg['n_cells'][k] for k in mc.key]
nb = len(mc); mc = mc[mc.agg_n_cells >= MIN_CELLS].copy().reset_index(drop=True)
log(f'  共通メタセル {nb} -> n_cells>={MIN_CELLS} で {len(mc)}')
log(f'  細胞種ごと: {mc.CellType.value_counts().to_dict()}')

def pearson_rows(xr, yv, dfree):
    xm = xr.mean(axis=1, keepdims=True); xs = xr.std(axis=1, ddof=0)
    ym = yv.mean(); ys = yv.std(ddof=0)
    cov = ((xr-xm)*(yv-ym)).mean(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where((xs>0)&(ys>0), cov/(xs*ys+1e-12), np.nan)
        t = r*np.sqrt(dfree/np.clip(1-r**2, 1e-12, None))
    return r, 2*tdist.sf(np.abs(t), dfree)

frames = []
for ct, gp in mc.groupby('CellType'):
    y = gp[TV].values.astype(np.float64)
    ag = gp['age'].values.astype(np.float64)
    ok = ~np.isnan(y) & ~np.isnan(ag)
    if ok.sum() < MIN_SAMPLES:
        log(f'  {ct}: n={ok.sum()} で skip'); continue
    sub = gp[ok]; n = int(ok.sum())
    t1 = time.time()
    X = np.zeros((n_cre, n), dtype=np.float32)
    for j, k in enumerate(sub.key):
        c = agg['expr_sum'][k].astype(np.float64); u = agg['umi_sum'][k]
        if u > 0: X[:, j] = (c/u*1e6).astype(np.float32)
    Xd = X.astype(np.float64); del X
    mean_expr = Xd.mean(axis=1)
    xr = rankdata(Xd, axis=1, method='average'); del Xd
    yv = y[ok]; av = ag[ok]
    yr = rankdata(yv, method='average'); ar = rankdata(av, method='average')

    r_plain, p_plain = pearson_rows(xr, yr, n-2)

    A = np.column_stack([np.ones(n), ar])
    AtA_inv = np.linalg.inv(A.T @ A)
    P_ = AtA_inv @ A.T
    yr_res = yr - A @ (P_ @ yr)
    xr_res = xr - (xr @ P_.T) @ A.T
    r_part, p_part = pearson_rows(xr_res, yr_res, n-3)
    del xr_res

    Xa = np.column_stack([np.ones(n), av])
    beta, *_ = np.linalg.lstsq(Xa, yv, rcond=None)
    yres_ols = yv - Xa @ beta
    r_ols, p_ols = pearson_rows(xr, rankdata(yres_ols, method='average'), n-2)
    del xr

    frames.append(pd.DataFrame({
        'cre_id': cres, 'cell_type': ct, 'n_samples': n,
        'mean_expression': mean_expr.astype(np.float32),
        'rho_plain': r_plain.astype(np.float32),   'p_plain': p_plain.astype(np.float32),
        'rho_partial': r_part.astype(np.float32),  'p_partial': p_part.astype(np.float32),
        'rho_resid': r_ols.astype(np.float32),     'p_resid': p_ols.astype(np.float32),
    }))
    log(f'  {ct}: n={n}  {time.time()-t1:.0f} s')

df = pd.concat(frames, ignore_index=True)
del agg
for col in ['plain','partial','resid']:
    df[f'fdr_{col}'] = np.nan
    for ct, g in df.groupby('cell_type'):
        pv = g[f'p_{col}'].values; ok = ~np.isnan(pv)
        f_ = np.full(len(pv), np.nan)
        if ok.sum(): f_[ok] = multipletests(pv[ok], method='fdr_bh')[1]
        df.loc[g.index, f'fdr_{col}'] = f_
    log(f'  FDR {col}: <0.05 = {int((df[f"fdr_{col}"]<0.05).sum()):,}')

out = f'{OUT}/tables/recorr_{COHORT}_subcluster.parquet'
df.to_parquet(out, index=False)
log(f'-> {out}  ({len(df):,} 行)')
