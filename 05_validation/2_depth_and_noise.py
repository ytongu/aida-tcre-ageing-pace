#!/usr/bin/env python3
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

D = '${PAPER_DIR}'
SRC = f'{D}/results/gene_mc_canonical'
OUT = f'{D}/results/noise_canonical'
os.makedirs(OUT, exist_ok=True)
MIN_DETECT_FRAC = 0.20
L = []
def w(s):
    print(s, flush=True); L.append(s)

meta = pd.read_csv(f'{SRC}/Metadata_metacells_canonical.csv')
meta = meta[meta.cohort == 'AIDA'].copy()
expr = pd.read_csv(f'{SRC}/Exprs_metacells_canonical.csv', index_col=0).astype(np.float32)
expr = expr[[c for c in expr.columns if c in set(meta.ID)]]
meta = meta[meta.ID.isin(expr.columns)].set_index('ID').loc[list(expr.columns)].reset_index()
meta = meta.rename(columns={'index': 'ID'}) if 'ID' not in meta.columns else meta
w(f'メタセル {expr.shape[1]:,} / 遺伝子 {expr.shape[0]:,}')

colsum = expr.sum(axis=0)
logcpm = np.log1p(expr.div(colsum, axis=1) * 1e6)
rows = []
for ct, sub in meta.groupby('Subcluster'):
    ids = sub.ID.tolist()
    X = logcpm[ids]
    keep = (expr[ids] > 0).mean(axis=1) >= MIN_DETECT_FRAC
    Xk = X.loc[keep].to_numpy()
    centroid = Xk.mean(axis=1)
    Xc = Xk - Xk.mean(axis=0, keepdims=True)
    cc = centroid - centroid.mean()
    num = (Xc * cc[:, None]).sum(axis=0)
    den = np.sqrt((Xc ** 2).sum(axis=0) * (cc ** 2).sum())
    df = sub.copy(); df['n_genes_used'] = int(keep.sum()); df['noise'] = 1.0 - num / den
    rows.append(df)
    w(f'  {ct:14s} n={len(ids):4d} genes={int(keep.sum()):6d} '
      f'noise 中央 {np.median(df.noise):.4f}')
res = pd.concat(rows, ignore_index=True)

pace = pd.read_csv(f'{D}/results/tage_canonical/pace_aida_canonical.csv')
res = res.merge(pace[['donor', 'subset', 'pace', 'age']].rename(
    columns={'donor': 'Sample', 'subset': 'Subcluster'}), on=['Sample', 'Subcluster'], how='left')
w(f'\npace 結合 {res.pace.notna().sum():,}/{len(res):,}')
res.to_csv(f'{OUT}/noise_table_canonical.csv', index=False)

def pcorr(x, y, Z):
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), 1)
    if m.sum() < 20: return np.nan, np.nan
    rx, ry = rankdata(x[m]), rankdata(y[m])
    RZ = np.c_[np.ones(m.sum()), np.apply_along_axis(rankdata, 0, Z[m])]
    B, *_ = np.linalg.lstsq(RZ, np.c_[rx, ry], rcond=None)
    e = np.c_[rx, ry] - RZ @ B
    r = spearmanr(e[:, 0], e[:, 1])
    return r.statistic, r.pvalue

w('\n=== ノイズ × pace（深度と細胞数を統制）===')
st = []
for ct, g in res.groupby('Subcluster'):
    g = g.dropna(subset=['pace'])
    if len(g) < 30: continue
    Z = np.c_[np.log10(g.Coverage + 1), np.log10(g.n_cells + 1)]
    rp, pp = pcorr(g.noise.values, g.pace.values, Z)
    ra, pa = pcorr(g.noise.values, g.age.values, Z)
    r0 = spearmanr(g.noise, g.pace)
    st.append(dict(Subcluster=ct, n=len(g), rho_pace_raw=r0.statistic, p_raw=r0.pvalue,
                   rho_pace_partial=rp, p_partial=pp, rho_age_partial=ra, p_age=pa))
ST = pd.DataFrame(st)
from statsmodels.stats.multitest import multipletests
ST['fdr_partial'] = multipletests(ST.p_partial.fillna(1), method='fdr_bh')[1]
ST.to_csv(f'{OUT}/noise_stats_canonical.csv', index=False)
for r in ST.sort_values('rho_pace_partial', ascending=False).itertuples():
    flag = ' *' if r.fdr_partial < 0.05 else ''
    w(f'  {r.Subcluster:14s} n={r.n:3d}  pace 偏 {r.rho_pace_partial:+.3f} '
      f'(FDR {r.fdr_partial:.3f}){flag}  暦年齢 偏 {r.rho_age_partial:+.3f}')
w(f'\n  FDR<0.05 の細胞種: {ST[ST.fdr_partial<0.05].Subcluster.tolist()}')

old = pd.read_csv(f'{D}/results/basic_figures/applied_noise/noise_stats.csv')
w(f'\n=== 旧版との比較 ===\n  旧版の列: {old.columns.tolist()[:6]}')
oc = [c for c in old.columns if 'Sub' in c or 'cell' in c.lower()]
if oc:
    m = ST.merge(old, left_on='Subcluster', right_on=oc[0], how='left', suffixes=('','_old'))
    m.to_csv(f'{OUT}/compare_vs_old.csv', index=False)
    w(f'  比較表を書き出し（{len(m)} 行）')
open(f'{OUT}/report.txt','w').write('\n'.join(L))
print(f'-> {OUT}')


import pickle, os, gzip
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, fisher_exact

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
TS = '_v4'
OUT = f'{D}/results/fixes/blocking_two'
os.makedirs(OUT, exist_ok=True)
TV = 'yugene_diff_EN_tAge'
TAGE = ('${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/'
        'subcluster/tage_all_combined_v2.csv')
GTF = '${DATA_DIR}/adachi/reference/GRCh38.113.gtf.gz'
rng = np.random.default_rng(0)
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

ann = pd.read_csv(f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv',
                  usecols=['cre_id', 'proximity']).drop_duplicates('cre_id')
prox_map = dict(zip(ann.cre_id, ann.proximity))

w('=' * 100)
w('[1] Synapse 単独 (単一センター)・粗指標 (遠位読み割合) の検定')
w('    バッチ固定効果 + 細胞種別 log10(Coverage) + log10(n_cells) を全部入れる')
w('=' * 100)
tg = pd.read_csv(TAGE)
tg = tg[(tg.model_type == 'Chronoage') & (tg['mode'] == 'EN')].copy()
tg = tg[['Donor_id', 'Batch', 'CellType', 'Age', 'Sex', 'Coverage', TV]].rename(
    columns={TV: 'tage'})
base = (tg[['Donor_id', 'Batch', 'Age']].drop_duplicates()
        .sort_values(['Donor_id', 'Age']).groupby('Donor_id').first().reset_index())
base_key = set(zip(base.Donor_id, base.Batch))
w(f'Synapse: ドナー {tg.Donor_id.nunique()} / baseline 訪問キー {len(base_key)}')

d = pickle.load(open(f'{OUR}/aggregation_cache_synapse_subcluster.pkl', 'rb'))
cres = np.array(d['cres']); keys = list(d['expr_sum'].keys())
prox_arr = np.array([prox_map.get(c, '') for c in cres])
is_dist = prox_arr == 'distal'
w(f'tCRE {len(cres):,} / 遠位 {int(is_dist.sum()):,}')

rows = []
for k in keys:
    coh, don, bat, ct = k
    if (don, bat) not in base_key: continue
    v = np.asarray(d['expr_sum'][k], dtype=np.float64)
    tot = v.sum()
    if tot < 1: continue
    rows.append(dict(Donor_id=don, Batch=bat, CellType=ct,
                     dist_frac=v[is_dist].sum() / tot,
                     umi=float(d['umi_sum'][k]), n_cells=int(d['n_cells'][k])))
mc = pd.DataFrame(rows)
mc = mc[mc.n_cells >= 10]
mc = mc.merge(tg, on=['Donor_id', 'Batch', 'CellType'], how='inner').dropna(subset=['tage'])
w(f'baseline メタセル (n_cells>=10, tAge あり): {len(mc):,}')

def partial_spear(df, xcol, ycol, covs, batch_col='Batch'):
    x = rankdata(df[xcol]); y = rankdata(df[ycol])
    Z = [np.ones(len(df))]
    for c in covs: Z.append(rankdata(df[c]))
    for b_ in sorted(df[batch_col].unique())[1:]:
        Z.append((df[batch_col] == b_).astype(float).values)
    Z = np.column_stack(Z)
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]; by = np.linalg.lstsq(Z, y, rcond=None)[0]
    r = spearmanr(x - Z @ bx, y - Z @ by)
    return r.statistic, r.pvalue

res1 = []
w(f'\n{"subcluster":14s}{"n":>5s}{"生rho":>8s}{"p":>9s}{"偏rho(深度+細胞数+バッチ)":>24s}{"p":>9s}')
for ct, g in mc.groupby('CellType'):
    if ct == 'Other' or len(g) < 60: continue
    g = g.copy()
    g['lcov'] = np.log10(g.Coverage.astype(float) + 1)
    g['lnc'] = np.log10(g.n_cells.astype(float) + 1)
    r0 = spearmanr(g.dist_frac, g.tage)
    rp, pp = partial_spear(g, 'dist_frac', 'tage', ['lcov', 'lnc'])
    w(f'{ct:14s}{len(g):5d}{r0.statistic:+8.3f}{r0.pvalue:9.4f}{rp:+24.3f}{pp:9.4f}')
    res1.append(dict(CellType=ct, n=len(g), rho_raw=r0.statistic, p_raw=r0.pvalue,
                     rho_partial=rp, p_partial=pp))
R1 = pd.DataFrame(res1)
R1.to_csv(f'{OUT}/synapse_only_crude_distal.csv', index=False)
npos_raw = int((R1.rho_raw > 0).sum()); npos = int((R1.rho_partial > 0).sum())
S_obs = float(R1.rho_partial.sum())
w(f'\n  正の細胞種: 生 {npos_raw}/{len(R1)} / 偏相関後 {npos}/{len(R1)}  S(偏rho 総和) = {S_obs:+.3f}')

donors = sorted(mc.Donor_id.unique())
tage_don = {}
for ct, g in mc.groupby('CellType'):
    tage_don[ct] = dict(zip(g.Donor_id, g.tage))
NPERM = 2000
null_S = np.zeros(NPERM)
sub = {ct: mc[mc.CellType == ct].copy() for ct in R1.CellType}
for it in range(NPERM):
    perm = dict(zip(donors, rng.permutation(donors)))
    S = 0.0
    for ct in R1.CellType:
        g = sub[ct]
        t2 = g.Donor_id.map(lambda x: tage_don[ct].get(perm[x], np.nan)).values
        m_ = np.isfinite(t2)
        if m_.sum() < 30: continue
        gg = g.loc[m_].copy(); gg['tage'] = t2[m_]
        gg['lcov'] = np.log10(gg.Coverage.astype(float) + 1)
        gg['lnc'] = np.log10(gg.n_cells.astype(float) + 1)
        rp, _ = partial_spear(gg, 'dist_frac', 'tage', ['lcov', 'lnc'])
        S += rp
    null_S[it] = S
p_one = float((null_S >= S_obs).mean()); p_two = float((np.abs(null_S) >= abs(S_obs)).mean())
w(f'  大域置換 ({NPERM} 回): 帰無 平均 {null_S.mean():+.3f} / SD {null_S.std():.3f} / 最大 {null_S.max():+.3f}')
w(f'  片側 p = {max(p_one, 1/NPERM):.4g} / 両側 p = {max(p_two, 1/NPERM):.4g}')
np.save(f'{OUT}/synapse_only_null_S.npy', null_S)
w('\n  ★判定: 偏相関後も正が優勢で置換検定が通れば、攻撃1 (深度・品質交絡) は Synapse で封じられる。')

res1b = []
for ct, g in mc.groupby('CellType'):
    if ct == 'Other' or len(g) < 60: continue
    g = g.copy()
    g['lcov'] = np.log10(g.Coverage.astype(float) + 1)
    g['lnc'] = np.log10(g.n_cells.astype(float) + 1)
    rp, pp = partial_spear(g, 'dist_frac', 'Age', ['lcov', 'lnc'])
    res1b.append(dict(CellType=ct, n=len(g), rho_partial_age=rp, p=pp))
R1b = pd.DataFrame(res1b)
R1b.to_csv(f'{OUT}/synapse_only_crude_distal_agecontrol.csv', index=False)
w(f'  対照 (暦年齢軸・同じ偏相関): 正 {int((R1b.rho_partial_age>0).sum())}/{len(R1b)}  '
  f'S = {R1b.rho_partial_age.sum():+.3f}')

w('\n' + '=' * 100)
w('[2] 同一遺伝子での共起 — 「別々の座位の 2 プログラム」の根拠を再計算して保存する')
w('=' * 100)
ca = pd.read_csv(f'{OUR}/cre_corr_aida_subcluster_significant_{TS}.csv',
                 usecols=['cre_id', 'cell_type', 'target_var', 'spearman_rho', 'fdr'])
ca = ca[(ca.target_var == TV) & (ca.fdr < 0.05)]
cs = pd.read_csv(f'{OUR}/cre_corr_synapse_subcluster_full_{TS}.csv',
                 usecols=['cre_id', 'cell_type', 'target_var', 'spearman_rho', 'fdr'])
cs = cs[cs.target_var == TV].rename(columns={'spearman_rho': 'rho_s', 'fdr': 'fdr_s'})
m = ca.merge(cs[['cre_id', 'cell_type', 'rho_s', 'fdr_s']], on=['cre_id', 'cell_type'])
m = m.dropna(subset=['spearman_rho', 'rho_s'])
rob = m[(m.fdr_s < 0.05) & (np.sign(m.spearman_rho) == np.sign(m.rho_s))].copy()
rob['proximity'] = rob.cre_id.map(prox_map)
rob = rob.dropna(subset=['proximity'])
w(f'頑健ペア: {len(rob):,} (異なり tCRE {rob.cre_id.nunique():,})')

tssd = {}
with gzip.open(GTF, 'rt') as f:
    for line in f:
        if line[0] == '#': continue
        p = line.split('\t')
        if p[2] != 'gene' or 'gene_biotype "protein_coding"' not in p[8]: continue
        i = p[8].find('gene_name "')
        if i < 0: continue
        gn = p[8][i + 11:p[8].find('"', i + 11)]
        ch = p[0] if p[0].startswith('chr') else 'chr' + p[0]
        tssd.setdefault(ch, []).append((int(p[3]) if p[6] == '+' else int(p[4]), gn))
for ch in tssd: tssd[ch].sort()
POS = {ch: np.array([x[0] for x in v]) for ch, v in tssd.items()}
def near_mid(cid):
    try:
        ch, s, e, st = cid.rsplit('_', 3); mid = (int(s) + int(e)) // 2
    except Exception:
        return None, None, None
    a = tssd.get(ch)
    if not a: return None, ch, mid
    i = int(np.searchsorted(POS[ch], mid)); best, bd = None, None
    for j in (i - 1, i):
        if 0 <= j < len(a):
            dd = abs(a[j][0] - mid)
            if bd is None or dd < bd: bd, best = dd, a[j][1]
    return best, ch, mid
uni = pd.DataFrame({'cre_id': rob.cre_id.unique()})
r_ = [near_mid(c) for c in uni.cre_id]
uni['gene'] = [x[0] for x in r_]; uni['chrom'] = [x[1] for x in r_]; uni['mid'] = [x[2] for x in r_]
rob = rob.merge(uni, on='cre_id', how='left').dropna(subset=['gene'])
rob['dir'] = np.where(rob.spearman_rho > 0, 'up', 'down')

gg = (rob.groupby(['gene', 'cell_type'])
      .apply(lambda x: pd.Series({
          'dist_up': int(((x.proximity == 'distal') & (x.dir == 'up')).any()),
          'dist_dn': int(((x.proximity == 'distal') & (x.dir == 'down')).any()),
          'prox_up': int(((x.proximity == 'proximal') & (x.dir == 'up')).any()),
          'prox_dn': int(((x.proximity == 'proximal') & (x.dir == 'down')).any())}),
             include_groups=False).reset_index())
n_all = len(gg)
a_ = int(((gg.dist_up == 1) & (gg.prox_dn == 1)).sum())
b_ = int(((gg.dist_up == 1) & (gg.prox_dn == 0)).sum())
c_ = int(((gg.dist_up == 0) & (gg.prox_dn == 1)).sum())
dd_ = n_all - a_ - b_ - c_
orr, pv = fisher_exact([[a_, b_], [c_, dd_]])
w(f'\n(a) 同一遺伝子 x 同一細胞種 (n={n_all:,} 遺伝子x細胞種):')
w(f'    遠位↑ かつ 近位↓ の共起: {a_} / 遠位↑のみ {b_} / 近位↓のみ {c_}')
w(f'    Fisher OR = {orr:.3f}, p = {pv:.3g}  (文書値 OR 0.043 と比較)')
both_g = gg[(gg.dist_up == 1) & (gg.prox_up == 1)]
w(f'    参考: 遠位↑ かつ 近位↑ (両方上昇): {len(both_g)} '
  f'({100*len(both_g)/max(int((gg.dist_up==1).sum()),1):.1f}% of 遠位↑遺伝子)')
gg.to_csv(f'{OUT}/cooccur_gene_celltype.csv', index=False)

du = rob[(rob.proximity == 'distal') & (rob.dir == 'up')]
def frac_near(target_df, source_df, win=100_000):
    hits = 0
    src = {ch: np.sort(g.mid.values) for ch, g in source_df.groupby('chrom')}
    for _, x in target_df.iterrows():
        a = src.get(x.chrom)
        if a is None or not len(a): continue
        i = np.searchsorted(a, x.mid - win)
        if i < len(a) and a[i] <= x.mid + win: hits += 1
    return hits, len(target_df)
pd_dn = rob[(rob.proximity == 'proximal') & (rob.dir == 'down')]
pd_up = rob[(rob.proximity == 'proximal') & (rob.dir == 'up')]
res_b = []
for lab, tgt in [('近位↓ (主対象)', pd_dn), ('近位↑ (対照)', pd_up)]:
    out = {}
    for ct in tgt.cell_type.unique():
        t_ = tgt[tgt.cell_type == ct]; s_ = du[du.cell_type == ct]
        h, n = frac_near(t_, s_)
        out[ct] = (h, n)
    H = sum(h for h, n in out.values()); N = sum(n for h, n in out.values())
    w(f'\n(b) {lab}: ±100kb 以内に同細胞種の 遠位↑ がある割合 = {H}/{N} ({100*H/max(N,1):.1f}%)')
    res_b.append(dict(group=lab, hits=H, n=N, frac=100 * H / max(N, 1)))
pd.DataFrame(res_b).to_csv(f'{OUT}/cooccur_100kb.csv', index=False)
w('\n  ★判定: OR が 1 を大きく下回り、±100kb でも近位↓ が対照より濃くなければ、')
w('    「同じ座位内の受け渡し」は棄却され「別々の座位で走る 2 つのプログラム」が保存される。')
w('    今回はじめて出力ファイル (cooccur_*.csv) として保存した。')

open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'\n-> {OUT}')


import pickle, os, gzip
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, fisher_exact

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
TS = '_v4'
OUT = f'{D}/results/blocking_two_canonical'
os.makedirs(OUT, exist_ok=True)
TV = 'yugene_diff_EN_tAge'
TAGE = f'{D}/results/tage_v110_check/tage_all_combined_v4.csv'
TV = 'yugene_diff_EN_tAge'
GTF = '${DATA_DIR}/adachi/reference/GRCh38.113.gtf.gz'
rng = np.random.default_rng(0)
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

ann = pd.read_csv(f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv',
                  usecols=['cre_id', 'proximity']).drop_duplicates('cre_id')
prox_map = dict(zip(ann.cre_id, ann.proximity))

w('=' * 100)
w('[1] Synapse 単独 (単一センター)・粗指標 (遠位読み割合) の検定')
w('    バッチ固定効果 + 細胞種別 log10(Coverage) + log10(n_cells) を全部入れる')
w('=' * 100)
tg = pd.read_csv(TAGE)
tg = tg[(tg.model_type == 'Chronoage') & (tg['mode'] == 'EN')].copy()
tg = tg[['Donor_id', 'Batch', 'CellType', 'Age', 'Sex']]
_cp = pd.read_csv(f'{D}/results/tage_canonical/pace_synapse_canonical.csv').rename(
    columns={'donor': 'Donor_id', 'subset': 'CellType', 'pace': 'tage'})[['Donor_id', 'CellType', 'tage']]
_cov = pd.read_csv(f'{D}/results/gene_mc_canonical/Metadata_metacells_canonical.csv',
                   usecols=['Sample', 'Subcluster', 'Coverage', 'cohort'])
_cov = _cov[_cov.cohort == 'Synapse'].rename(columns={'Sample': 'Donor_id', 'Subcluster': 'CellType'})[['Donor_id', 'CellType', 'Coverage']]
tg = tg.merge(_cp, on=['Donor_id', 'CellType'], how='left').merge(_cov, on=['Donor_id', 'CellType'], how='left')
base = (tg[['Donor_id', 'Batch', 'Age']].drop_duplicates()
        .sort_values(['Donor_id', 'Age']).groupby('Donor_id').first().reset_index())
base_key = set(zip(base.Donor_id, base.Batch))
w(f'Synapse: ドナー {tg.Donor_id.nunique()} / baseline 訪問キー {len(base_key)}')

d = pickle.load(open(f'{OUR}/aggregation_cache_synapse_subcluster.pkl', 'rb'))
cres = np.array(d['cres']); keys = list(d['expr_sum'].keys())
prox_arr = np.array([prox_map.get(c, '') for c in cres])
is_dist = prox_arr == 'distal'
w(f'tCRE {len(cres):,} / 遠位 {int(is_dist.sum()):,}')

rows = []
for k in keys:
    coh, don, bat, ct = k
    if (don, bat) not in base_key: continue
    v = np.asarray(d['expr_sum'][k], dtype=np.float64)
    tot = v.sum()
    if tot < 1: continue
    rows.append(dict(Donor_id=don, Batch=bat, CellType=ct,
                     dist_frac=v[is_dist].sum() / tot,
                     umi=float(d['umi_sum'][k]), n_cells=int(d['n_cells'][k])))
mc = pd.DataFrame(rows)
mc = mc[mc.n_cells >= 10]
mc = mc.merge(tg, on=['Donor_id', 'Batch', 'CellType'], how='inner').dropna(subset=['tage'])
w(f'baseline メタセル (n_cells>=10, tAge あり): {len(mc):,}')

def partial_spear(df, xcol, ycol, covs, batch_col='Batch'):
    x = rankdata(df[xcol]); y = rankdata(df[ycol])
    Z = [np.ones(len(df))]
    for c in covs: Z.append(rankdata(df[c]))
    for b_ in sorted(df[batch_col].unique())[1:]:
        Z.append((df[batch_col] == b_).astype(float).values)
    Z = np.column_stack(Z)
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]; by = np.linalg.lstsq(Z, y, rcond=None)[0]
    r = spearmanr(x - Z @ bx, y - Z @ by)
    return r.statistic, r.pvalue

res1 = []
w(f'\n{"subcluster":14s}{"n":>5s}{"生rho":>8s}{"p":>9s}{"偏rho(深度+細胞数+バッチ)":>24s}{"p":>9s}')
for ct, g in mc.groupby('CellType'):
    if ct == 'Other' or len(g) < 60: continue
    g = g.copy()
    g['lcov'] = np.log10(g.Coverage.astype(float) + 1)
    g['lnc'] = np.log10(g.n_cells.astype(float) + 1)
    r0 = spearmanr(g.dist_frac, g.tage)
    rp, pp = partial_spear(g, 'dist_frac', 'tage', ['lcov', 'lnc'])
    w(f'{ct:14s}{len(g):5d}{r0.statistic:+8.3f}{r0.pvalue:9.4f}{rp:+24.3f}{pp:9.4f}')
    res1.append(dict(CellType=ct, n=len(g), rho_raw=r0.statistic, p_raw=r0.pvalue,
                     rho_partial=rp, p_partial=pp))
R1 = pd.DataFrame(res1)
R1.to_csv(f'{OUT}/synapse_only_crude_distal.csv', index=False)
npos_raw = int((R1.rho_raw > 0).sum()); npos = int((R1.rho_partial > 0).sum())
S_obs = float(R1.rho_partial.sum())
w(f'\n  正の細胞種: 生 {npos_raw}/{len(R1)} / 偏相関後 {npos}/{len(R1)}  S(偏rho 総和) = {S_obs:+.3f}')

donors = sorted(mc.Donor_id.unique())
tage_don = {}
for ct, g in mc.groupby('CellType'):
    tage_don[ct] = dict(zip(g.Donor_id, g.tage))
NPERM = 2000
null_S = np.zeros(NPERM)
sub = {ct: mc[mc.CellType == ct].copy() for ct in R1.CellType}
for it in range(NPERM):
    perm = dict(zip(donors, rng.permutation(donors)))
    S = 0.0
    for ct in R1.CellType:
        g = sub[ct]
        t2 = g.Donor_id.map(lambda x: tage_don[ct].get(perm[x], np.nan)).values
        m_ = np.isfinite(t2)
        if m_.sum() < 30: continue
        gg = g.loc[m_].copy(); gg['tage'] = t2[m_]
        gg['lcov'] = np.log10(gg.Coverage.astype(float) + 1)
        gg['lnc'] = np.log10(gg.n_cells.astype(float) + 1)
        rp, _ = partial_spear(gg, 'dist_frac', 'tage', ['lcov', 'lnc'])
        S += rp
    null_S[it] = S
p_one = float((null_S >= S_obs).mean()); p_two = float((np.abs(null_S) >= abs(S_obs)).mean())
w(f'  大域置換 ({NPERM} 回): 帰無 平均 {null_S.mean():+.3f} / SD {null_S.std():.3f} / 最大 {null_S.max():+.3f}')
w(f'  片側 p = {max(p_one, 1/NPERM):.4g} / 両側 p = {max(p_two, 1/NPERM):.4g}')
np.save(f'{OUT}/synapse_only_null_S.npy', null_S)
w('\n  ★判定: 偏相関後も正が優勢で置換検定が通れば、攻撃1 (深度・品質交絡) は Synapse で封じられる。')

res1b = []
for ct, g in mc.groupby('CellType'):
    if ct == 'Other' or len(g) < 60: continue
    g = g.copy()
    g['lcov'] = np.log10(g.Coverage.astype(float) + 1)
    g['lnc'] = np.log10(g.n_cells.astype(float) + 1)
    rp, pp = partial_spear(g, 'dist_frac', 'Age', ['lcov', 'lnc'])
    res1b.append(dict(CellType=ct, n=len(g), rho_partial_age=rp, p=pp))
R1b = pd.DataFrame(res1b)
R1b.to_csv(f'{OUT}/synapse_only_crude_distal_agecontrol.csv', index=False)
w(f'  対照 (暦年齢軸・同じ偏相関): 正 {int((R1b.rho_partial_age>0).sum())}/{len(R1b)}  '
  f'S = {R1b.rho_partial_age.sum():+.3f}')

open(f'{OUT}/report.txt', 'w').write('\n'.join(L))


import pickle, numpy as np, pandas as pd
from pathlib import Path

P = Path('${PROJECT_DIR}/results')
OUT = P / 'PAPER_CRE_MECHANISM/results/subcluster_marker_check_canonical'
OUT.mkdir(parents=True, exist_ok=True)
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

MARK = {
    'ナイーブT': ['CCR7', 'SELL', 'LEF1', 'TCF7'],
    '記憶/実行T': ['IL7R', 'S100A4', 'GZMK', 'GZMB', 'NKG7', 'PRF1'],
    'CD4': ['CD4', 'CD40LG'], 'CD8': ['CD8A', 'CD8B'],
    'NK': ['KLRD1', 'GNLY', 'NCAM1'],
    'B系': ['MS4A1', 'CD79A'], 'ナイーブB': ['TCL1A', 'IGHD'], '記憶B': ['CD27', 'TNFRSF13B'],
    '単球': ['CD14', 'LYZ', 'FCGR3A'],
}
GENES = sorted({g for v in MARK.values() for g in v})

cls = pd.read_csv(P / 'intragenic_initiation/cre_classification.csv.gz',
                  usecols=['cre_id', 'is_prom', 'body_gene'])
pm = cls[(cls.is_prom) & (cls.body_gene.isin(GENES))]
g2c = pm.groupby('body_gene').cre_id.apply(list).to_dict()
w(f'マーカー {len(GENES)} 個中、プロモーター tCRE が見つかった遺伝子: {len(g2c)}')
w(f'  見つからない: {sorted(set(GENES) - set(g2c))}')

res = {}
for coh, cache in [('AIDA', 'aggregation_cache_aida_subcluster.pkl'),
                   ('Synapse', 'aggregation_cache_synapse_subcluster.pkl')]:
    with open(P / 'phase3b_ourcd4' / cache, 'rb') as f:
        C = pickle.load(f)
    idx = {c: i for i, c in enumerate(C['cres'])}
    byk = {}
    for k in C['expr_sum']:
        byk.setdefault(k[3], {}).setdefault(k[1], []).append(k)
    for sc, dons in byk.items():
        vals = {g: [] for g in g2c}
        ncell = []
        for d, ks in dons.items():
            e = np.sum([C['expr_sum'][k] for k in ks], axis=0)
            u = sum(C['umi_sum'][k] for k in ks)
            ncell.append(sum(C['n_cells'][k] for k in ks))
            if u <= 0: continue
            for g, cs in g2c.items():
                ii = [idx[c] for c in cs if c in idx]
                if ii: vals[g].append(e[ii].sum() / u * 1e6)
        res[(coh, sc)] = {g: np.median(v) if v else np.nan for g, v in vals.items()}
        res[(coh, sc)]['_ncell'] = np.median(ncell)
    del C
    w(f'{coh} 完了')

df = pd.DataFrame(res).T
df.index.names = ['cohort', 'sub']
df.to_csv(OUT / 'marker_cpm.csv')

w('\n' + '=' * 80)
w('マーカー CPM 中央値 (プロモーター tCRE 由来)')
w('=' * 80)
for grp, gs in MARK.items():
    gs = [g for g in gs if g in df.columns]
    if not gs: continue
    w(f'\n--- {grp}: {", ".join(gs)} ---')
    sub = df[gs].copy()
    sub['計'] = sub.sum(axis=1)
    pv = sub['計'].unstack(0)
    w(pv.to_string(float_format=lambda v: f'{v:8.0f}'))

w('\n' + '=' * 80)
w('★ラベルの整合性判定: ナイーブ度 = ナイーブT計 / (ナイーブT計 + 記憶実行T計)')
w('=' * 80)
nv = [g for g in MARK['ナイーブT'] if g in df.columns]
mem = [g for g in MARK['記憶/実行T'] if g in df.columns]
score = (df[nv].sum(axis=1) / (df[nv].sum(axis=1) + df[mem].sum(axis=1))).unstack(0)
score['差'] = (score.get('AIDA', np.nan) - score.get('Synapse', np.nan)).abs()
w(score.to_string(float_format=lambda v: f'{v:.3f}'))
w('\n  → 同じラベルなら 2 コホートで近い値になるはず。差が大きい集団はラベル不整合の疑い')

nc = df['_ncell'].unstack(0)
prop = nc / nc.sum()
w('\n=== 細胞数の割合 (ドナー中央値ベース) ===')
w(pd.concat([nc.round(0), (100 * prop).round(1).add_suffix('_%')], axis=1).to_string())
(OUT / 'report.txt').write_text('\n'.join(L))
w(f'\n→ {OUT}')


import pickle, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata, norm, fisher_exact
from statsmodels.stats.multitest import multipletests

R = Path('${HOME}/projects/integrated_aida_synapse/results')
OUT = R / 'PAPER_CRE_MECHANISM/results/depth_gate_robustset_v2'
OUT.mkdir(parents=True, exist_ok=True)
CTS = ['B_cell', 'CD4_T_cell', 'CD8_T_cell', 'Monocyte', 'NK_cell']
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

w('=' * 78); w('主メッセージ 深さ交絡 決着検定 (頑健セットを選抜からやり直す)'); w('=' * 78)

cls = pd.read_csv(R / 'intragenic_initiation/cre_classification.csv.gz',
                  usecols=['cre_id', 'is_prom'])
meta = pd.read_csv('${SCRATCH}'
                   'scratchpad/cd_tss/mc_meta.csv').set_index(['cohort', 'Donor_id', 'CellType'])

def zp(r, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 2 * norm.sf(np.abs(np.arctanh(np.clip(r, -0.999999, 0.999999)) * np.sqrt(n - 3)))

def corr_variants(X, y, depth, country):
    n = len(y)
    Rx = np.apply_along_axis(rankdata, 0, X)
    Rx = (Rx - Rx.mean(0)) / np.where(Rx.std(0) == 0, 1, Rx.std(0))
    Ry = rankdata(y); Ry = (Ry - Ry.mean()) / Ry.std()
    covs = {'未調整': None,
            '深さ調整': [rankdata(depth)],
            '深さ+国調整': [rankdata(depth)] + ([ (country == 'JP').astype(float) ]
                                              if country is not None and len(set(country)) > 1 else [])}
    out = {}
    for lab, cv in covs.items():
        if cv is None:
            out[lab] = (Rx.T @ Ry / n, n)
            continue
        Z = np.column_stack([(c - np.mean(c)) / (np.std(c) if np.std(c) else 1) for c in cv])

        Q, _ = np.linalg.qr(Z)
        Ry_r = Ry - Q @ (Q.T @ Ry)
        Xr = Rx - Q @ (Q.T @ Rx)
        num = Xr.T @ Ry_r
        den = np.sqrt((Xr ** 2).sum(0)) * np.sqrt((Ry_r ** 2).sum())
        out[lab] = (np.where(den > 0, num / np.where(den == 0, 1, den), 0.0), n - Q.shape[1])
    return out

res = {}
for coh, cache in [('AIDA', 'aggregation_cache_aida_celltype.pkl'),
                   ('Synapse', 'aggregation_cache_synapse_celltype.pkl')]:
    w(f'\n{coh} キャッシュ読み込み中...')
    with open(R / 'phase3b_cre_corr' / cache, 'rb') as f:
        C = pickle.load(f)
    cres = np.array(C['cres'])
    if 'cre_order' not in res:
        res['cre_order'] = cres
        cl = cls.set_index('cre_id').reindex(cres)
        res['is_prom'] = cl.is_prom.values.astype(bool)
        w(f'  CRE {len(cres):,} / 分類付き {int(cl.is_prom.notna().sum()):,}')
    keys = list(C['expr_sum'].keys())
    for ct in CTS:
        sel = [k for k in keys if k[3] == ct]

        bydon = {}
        for k in sel:
            bydon.setdefault(k[1], []).append(k)
        sel = sorted(bydon)
        idx = [(coh, d_, ct) for d_ in sel]
        md = meta.reindex(idx)
        ok = md.yugene_diff_EN_tAge.notna().values
        sel = [k for k, o in zip(sel, ok) if o]
        y = md.yugene_diff_EN_tAge.values[ok]
        donors = list(sel)
        country = np.array(['JP' if d.startswith('JP') else 'KR' for d in donors]) if coh == 'AIDA' else None
        M = np.vstack([np.sum([C['expr_sum'][k] for k in bydon[d_]], axis=0)
                       for d_ in sel]).astype(np.float32)
        umi = np.array([sum(C['umi_sum'][k] for k in bydon[d_]) for d_ in sel], float)
        depth = umi / np.array([sum(C['n_cells'][k] for k in bydon[d_]) for d_ in sel], float)
        cpm = M / umi[:, None] * 1e6
        keep = (cpm > 0).mean(0) >= 0.2
        res[(coh, ct, 'keep')] = keep
        res[(coh, ct, 'corr')] = corr_variants(cpm[:, keep], y, depth, country)
        del M, cpm
        w(f'  {ct}: ドナー {len(y)} (キー {sum(len(v) for v in bydon.values())}) / 検定 CRE {int(keep.sum()):,}')
    del C

is_prom = res['is_prom']
rows = []
for variant in ['未調整', '深さ調整', '深さ+国調整']:
    for ct in CTS:
        ka, ks = res[('AIDA', ct, 'keep')], res[('Synapse', ct, 'keep')]
        both = ka & ks
        ra, na = res[('AIDA', ct, 'corr')][variant]
        rs, ns = res[('Synapse', ct, 'corr')][variant]

        RA = np.full(len(is_prom), np.nan); RA[ka] = ra
        RS = np.full(len(is_prom), np.nan); RS[ks] = rs
        m = both & np.isfinite(RA) & np.isfinite(RS)
        fa = np.full(len(is_prom), 1.0); fs = np.full(len(is_prom), 1.0)
        fa[m] = multipletests(zp(RA[m], na), method='fdr_bh')[1]
        fs[m] = multipletests(zp(RS[m], ns), method='fdr_bh')[1]
        rob = m & (fa < 0.05) & (fs < 0.05) & (np.sign(RA) == np.sign(RS))
        up = rob & (RA > 0); dist = ~is_prom
        a, b = int((up & rob & dist).sum()), int((rob & ~up & dist).sum())
        c, dd = int((up & rob & ~dist).sum()), int((rob & ~up & ~dist).sum())
        orr, pf = fisher_exact([[a, b], [c, dd]]) if min(a + b, c + dd) > 0 else (np.nan, np.nan)
        rows.append(dict(variant=variant, celltype=ct, n_robust=int(rob.sum()),
                         up_distal_pct=a / max(a + b, 1) * 100, up_prox_pct=c / max(c + dd, 1) * 100,
                         OR=orr, p=pf))

    sub = [r for r in rows if r['variant'] == variant]
    rows.append(dict(variant=variant, celltype='ALL', n_robust=sum(r['n_robust'] for r in sub),
                     up_distal_pct=np.nan, up_prox_pct=np.nan, OR=np.nan, p=np.nan))

df = pd.DataFrame(rows); df.to_csv(OUT / 'results.csv', index=False)
w('')
w('=== 結果 ===')
w(f'{"調整":12s}{"細胞種":13s}{"頑健CRE":>8s}{"遠位上昇%":>9s}{"近位上昇%":>9s}{"OR":>9s}{"p":>10s}')
for _, x in df.iterrows():
    if x.celltype == 'ALL':
        w(f'{x.variant:12s}{"(合計)":13s}{x.n_robust:8d}'); continue
    w(f'{x.variant:12s}{x.celltype:13s}{x.n_robust:8d}{x.up_distal_pct:8.1f}%{x.up_prox_pct:8.1f}%'
      f'{x.OR:9.2f}{x.p:10.1e}')
w('')
w('=== ゲート判定 (未調整で既報 遠位88.8% / 近位34.8% / OR14.83 / n=7,508 を再現するか) ===')
g = df[(df.variant == '未調整') & (df.celltype != 'ALL')]
w(f'  頑健 CRE 合計 {int(g.n_robust.sum()):,} (既報 7,508)')
w(f'  細胞種別 OR: ' + ', '.join(f'{r.celltype} {r.OR:.1f}' for _, r in g.iterrows()))
w(f'  既報の細胞種別 OR: B 21.2, CD4 8.4, CD8 7.9, Monocyte 23.0, NK 53.3')
(OUT / 'report.txt').write_text('\n'.join(L))
w(f'\n→ {OUT}')


import pandas as pd, numpy as np, gzip, os, sys
from scipy.stats import fisher_exact, mannwhitneyu

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
TS = '_v4'
OUT = f'{D}/results/gate3_recalc'
os.makedirs(OUT, exist_ok=True)
TV = 'yugene_diff_EN_tAge'
rng = np.random.default_rng(0)
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

w('=' * 100)
w('本文執筆前の必須再計算 3 件')
w('=' * 100)

ann = pd.read_csv(f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv',
                  usecols=['cre_id', 'proximity', 'enhancer_class'])
w(f'注釈: {len(ann):,} 素子')

COLS = ['cre_id', 'cell_type', 'target_var', 'spearman_rho', 'fdr', 'mean_expression']
w('AIDA (有意分) 読み込み...')
A = pd.read_csv(f'{OUR}/cre_corr_aida_subcluster_significant_{TS}.csv', usecols=COLS)
w(f'  {len(A):,} 行')
w('Synapse (全件) 読み込み...')
S = pd.read_csv(f'{OUR}/cre_corr_synapse_subcluster_full_{TS}.csv',
                usecols=['cre_id', 'cell_type', 'target_var', 'spearman_rho', 'fdr'])
w(f'  {len(S):,} 行')

tssd = {}
with gzip.open('${DATA_DIR}/adachi/reference/GRCh38.113.gtf.gz', 'rt') as f:
    for line in f:
        if line[0] == '#': continue
        p = line.split('\t')
        if p[2] != 'gene' or 'gene_biotype "protein_coding"' not in p[8]: continue
        i = p[8].find('gene_name "')
        if i < 0: continue
        gn = p[8][i + 11:p[8].find('"', i + 11)]
        ch = p[0] if p[0].startswith('chr') else 'chr' + p[0]
        tssd.setdefault(ch, []).append((int(p[3]) if p[6] == '+' else int(p[4]), gn))
for ch in tssd: tssd[ch].sort()
POS = {ch: np.array([x[0] for x in v]) for ch, v in tssd.items()}
def nearest(cid):
    try:
        ch, s, e, st = cid.rsplit('_', 3); mid = (int(s) + int(e)) // 2
    except Exception:
        return None
    a = tssd.get(ch)
    if not a: return None
    i = int(np.searchsorted(POS[ch], mid)); best, bd = None, None
    for j in (i - 1, i):
        if 0 <= j < len(a):
            d = abs(a[j][0] - mid)
            if bd is None or d < bd: bd, best = d, a[j][1]
    return best

clock = set(pd.read_csv(f'{D}/results/clock_contribution/clock_coefficients_named.csv')
            .human_sym.astype(str).str.upper())
w(f'\n時計の非ゼロ係数遺伝子: {len(clock):,}')

e2g = pd.read_csv(f'{B}/re2g_targets_clean/targets_clean.csv',
                  usecols=['cre_id', 'target_gene'])
e2g['target_gene'] = e2g.target_gene.astype(str).str.upper()
e2g_clock = set(e2g[e2g.target_gene.isin(clock)].cre_id.unique())
w(f'E2G 標的に時計遺伝子を含む素子: {len(e2g_clock):,} (E2G 標的つき素子 {e2g.cre_id.nunique():,} 中)')

def build(axis):
    a = A[(A.target_var == axis) & (A.fdr < 0.05)].copy()
    s = S[S.target_var == axis].rename(columns={'spearman_rho': 'rho_s', 'fdr': 'fdr_s'})
    m = a.merge(s[['cre_id', 'cell_type', 'rho_s', 'fdr_s']], on=['cre_id', 'cell_type'],
                how='inner').merge(ann, on='cre_id', how='left')
    m['same_sign'] = np.sign(m.spearman_rho) == np.sign(m.rho_s)
    m['robust'] = m.same_sign & (m.fdr_s < 0.05)
    return m

def program_stats(df, tag):
    r = df[df.robust]
    out = {}
    for prox in ['distal', 'proximal']:
        x = r[r.proximity == prox]
        out[f'{prox}_n'] = len(x)
        out[f'{prox}_up_pct'] = 100 * (x.spearman_rho > 0).mean() if len(x) else np.nan
    out['heldout_n'] = len(df)
    out['heldout_concord_pct'] = 100 * df.same_sign.mean() if len(df) else np.nan
    out['tag'] = tag
    return out

w('\n' + '=' * 100)
w('(1) 時計近傍素子を落としても二つのプログラムが残るか')
w('=' * 100)
m = build(TV)
w(f'AIDA 有意 x Synapse 対応: {len(m):,} 行 / 両コホート頑健 {int(m.robust.sum()):,} 行')

uni = pd.DataFrame({'cre_id': m.cre_id.unique()})
uni['gene'] = [nearest(c) for c in uni.cre_id]
uni['gene'] = uni.gene.astype(str).str.upper()
m = m.merge(uni, on='cre_id', how='left')
m['clock_near'] = m.gene.isin(clock)
m['clock_e2g'] = m.cre_id.isin(e2g_clock)
m['clock_any'] = m.clock_near | m.clock_e2g

r = m[m.robust]
w(f'\n頑健素子のうち時計遺伝子に紐づくもの:')
w(f'  最近接が時計遺伝子     : {int(r.clock_near.sum()):,} / {len(r):,} ({100*r.clock_near.mean():.1f}%)')
w(f'  E2G 標的に時計遺伝子   : {int(r.clock_e2g.sum()):,} / {len(r):,} ({100*r.clock_e2g.mean():.1f}%)')
w(f'  ★どちらか (落とす対象): {int(r.clock_any.sum()):,} / {len(r):,} ({100*r.clock_any.mean():.1f}%)')

rows = [program_stats(m, '全素子'),
        program_stats(m[~m.clock_near], '最近接が時計遺伝子の素子を除外'),
        program_stats(m[~m.clock_any], '★時計に紐づく素子を全部除外 (最厳)')]
tab = pd.DataFrame(rows)[['tag', 'distal_n', 'distal_up_pct', 'proximal_n', 'proximal_up_pct',
                          'heldout_n', 'heldout_concord_pct']]
w('')
w(f'{"条件":42s}{"遠位n":>8s}{"遠位上昇%":>10s}{"近位n":>8s}{"近位上昇%":>10s}{"held-out n":>12s}{"方向一致%":>10s}')
for _, x in tab.iterrows():
    w(f'{x.tag:42s}{x.distal_n:8,d}{x.distal_up_pct:10.1f}{x.proximal_n:8,d}'
      f'{x.proximal_up_pct:10.1f}{x.heldout_n:12,d}{x.heldout_concord_pct:10.1f}')
tab.to_csv(f'{OUT}/clock_sensitivity.csv', index=False)

cf = m[(~m.clock_any) & m.robust]
a_ = int(((cf.proximity == 'distal') & (cf.spearman_rho > 0)).sum())
b_ = int(((cf.proximity == 'distal') & (cf.spearman_rho < 0)).sum())
c_ = int(((cf.proximity == 'proximal') & (cf.spearman_rho > 0)).sum())
d_ = int(((cf.proximity == 'proximal') & (cf.spearman_rho < 0)).sum())
orr, pv = fisher_exact([[a_, b_], [c_, d_]])
w(f'\n  最厳条件での 遠位上昇 vs 近位上昇: OR {orr:.2f}, p={pv:.3g}  (遠位 {a_}/{a_+b_}, 近位 {c_}/{c_+d_})')

w('\n  細胞種ごと (最厳条件、遠位の上昇%):')
w(f'  {"細胞種":14s}{"全素子":>16s}{"時計除外後":>18s}')
for ct in sorted(m.cell_type.unique()):
    g0 = m[(m.cell_type == ct) & m.robust & (m.proximity == 'distal')]
    g1 = g0[~g0.clock_any]
    if len(g0) < 20: continue
    w(f'  {ct:14s}{100*(g0.spearman_rho>0).mean():9.1f}% (n={len(g0):5,d}){100*(g1.spearman_rho>0).mean():9.1f}% (n={len(g1):5,d})')

w('\n' + '=' * 100)
w('(2) CD4 記憶の陰性は検出力で説明されるか')
w('=' * 100)
up_d = m[(m.proximity == 'distal') & (m.spearman_rho > 0)]
w('\n  各細胞種の素材と、上昇x遠位の Synapse 再現率:')
w(f'  {"細胞種":14s}{"AIDA有意(上昇遠位)":>20s}{"発現中央":>10s}{"再現率":>9s}')
base = []
for ct, g in up_d.groupby('cell_type'):
    base.append(dict(cell_type=ct, n=len(g), expr=g.mean_expression.median(),
                     rep=100 * g.same_sign.mean()))
base = pd.DataFrame(base).sort_values('rep', ascending=False)
for _, x in base.iterrows():
    w(f'  {x.cell_type:14s}{int(x.n):20,d}{x.expr:10.0f}{x.rep:8.1f}%')
base.to_csv(f'{OUT}/updistal_by_celltype_raw.csv', index=False)

w('\n  ★検出数と発現量を揃えた対比 (CD4_Memory を基準に、各細胞種から発現量最近傍を 1:1 抽出)')
ref = up_d[up_d.cell_type == 'CD4_Memory']
w(f'    基準 CD4_Memory: n={len(ref):,}, 発現中央 {ref.mean_expression.median():.0f}, 再現率 {100*ref.same_sign.mean():.1f}%')
res2 = [dict(cell_type='CD4_Memory', n=len(ref), expr_med=ref.mean_expression.median(),
             rep=100 * ref.same_sign.mean(), n_boot=np.nan, rep_lo=np.nan, rep_hi=np.nan)]
tgt = np.sort(np.log10(ref.mean_expression.values + 1))
w(f'    {"細胞種":14s}{"抽出n":>7s}{"発現中央":>10s}{"再現率":>9s}{"95%CI":>18s}')
for ct, g in up_d.groupby('cell_type'):
    if ct == 'CD4_Memory' or len(g) < len(ref): continue
    pool = g.copy()
    pl = np.log10(pool.mean_expression.values + 1)
    order = np.argsort(pl); pls = pl[order]
    used, idx = set(), []
    for t in tgt:
        j = int(np.searchsorted(pls, t))
        cand = [k for k in range(max(0, j - 60), min(len(pls), j + 60)) if k not in used]
        if not cand: continue
        k = min(cand, key=lambda k: abs(pls[k] - t))
        used.add(k); idx.append(order[k])
    sub = pool.iloc[idx]
    boot = [100 * sub.same_sign.sample(len(sub), replace=True, random_state=int(s)).mean()
            for s in range(400)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    w(f'    {ct:14s}{len(sub):7,d}{sub.mean_expression.median():10.0f}'
      f'{100*sub.same_sign.mean():8.1f}%{f"[{lo:.1f}, {hi:.1f}]":>18s}')
    res2.append(dict(cell_type=ct, n=len(sub), expr_med=sub.mean_expression.median(),
                     rep=100 * sub.same_sign.mean(), n_boot=400, rep_lo=lo, rep_hi=hi))
pd.DataFrame(res2).to_csv(f'{OUT}/updistal_matched.csv', index=False)

bm = [100 * ref.same_sign.sample(len(ref), replace=True, random_state=int(s)).mean() for s in range(400)]
w(f'    CD4_Memory の 95%CI: [{np.percentile(bm,2.5):.1f}, {np.percentile(bm,97.5):.1f}]')

dn_p = m[(m.proximity == 'proximal') & (m.spearman_rho < 0)]
w('\n  陽性対照 — 低下x近位の再現率 (低下プログラムは全系統で走るはず):')
for ct, g in dn_p.groupby('cell_type'):
    if len(g) < 20: continue
    w(f'    {ct:14s}{len(g):8,d}{100*g.same_sign.mean():8.1f}%')

w('\n' + '=' * 100)
w('(3) 暦年齢軸に載せ替えると Stevens 2026 の方向が出るか')
w('    Stevens 2026: ヒト HSPC / 暦年齢 / ATAC -> 遠位は閉じ、近位プロモーターは保存または増幅')
w('=' * 100)
ma = build('age')
ma = ma.merge(uni, on='cre_id', how='left')
w(f'暦年齢軸: AIDA 有意 x Synapse 対応 {len(ma):,} 行 / 両コホート頑健 {int(ma.robust.sum()):,} 行')
cmp_rows = []
for lab, df in [('加速軸 (yugene_diff_EN_tAge)', m), ('暦年齢軸 (age)', ma)]:
    r_ = df[df.robust]
    d_ = r_[r_.proximity == 'distal']; p_ = r_[r_.proximity == 'proximal']
    cmp_rows.append(dict(axis=lab, distal_n=len(d_), distal_up=100 * (d_.spearman_rho > 0).mean() if len(d_) else np.nan,
                         proximal_n=len(p_), proximal_up=100 * (p_.spearman_rho > 0).mean() if len(p_) else np.nan,
                         heldout=100 * df.same_sign.mean()))
cmp = pd.DataFrame(cmp_rows)
w('')
w(f'{"軸":32s}{"遠位n":>8s}{"遠位上昇%":>10s}{"近位n":>8s}{"近位上昇%":>10s}{"方向一致%":>10s}')
for _, x in cmp.iterrows():
    w(f'{x.axis:32s}{x.distal_n:8,d}{x.distal_up:10.1f}{x.proximal_n:8,d}{x.proximal_up:10.1f}{x.heldout:10.1f}')
cmp.to_csv(f'{OUT}/axis_comparison.csv', index=False)

key = ['cre_id', 'cell_type']
both = (m[m.robust][key + ['spearman_rho', 'proximity']]
        .merge(ma[key + ['spearman_rho']].rename(columns={'spearman_rho': 'rho_age'}), on=key, how='left'))
w(f'\n  加速軸で頑健な素子 {len(both):,} のうち、暦年齢軸の AIDA 有意リストにも載るもの: '
  f'{int(both.rho_age.notna().sum()):,} ({100*both.rho_age.notna().mean():.1f}%)')
w('  -> 載らない素子は暦年齢では検出されない = 「暦年齢では見えない」の直接の数字')
both.to_csv(f'{OUT}/robust_on_age_axis.csv', index=False)

open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'\n-> {OUT}')
