#!/usr/bin/env python3
import os, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

D = '${PAPER_DIR}'
OUT = f'{D}/results/cima_part4b'
os.makedirs(OUT, exist_ok=True)
TV = 'yugene_diff_EN_tAge'
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

def partial(x, y, z):
    rx, ry = rankdata(x), rankdata(y)
    Z = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in z])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return spearmanr(ex, ey)

cand = sorted(glob.glob(f'{D}/results/cima_tage/tage_all_combined_*.csv'))
if not cand:
    cand = sorted(glob.glob(f'{D}/results/cima_tage/tage_Chronoage_EN_*.csv'))
f = cand[-1]
ta = pd.read_csv(f)
w(f'tAge: {os.path.basename(f)}  {len(ta):,} 行')
if 'model_type' in ta.columns:
    w(f'  model_type: {ta.model_type.unique().tolist()} / mode: {ta["mode"].unique().tolist()}')
    ta = ta[(ta.model_type == 'Chronoage') & (ta['mode'] == 'EN')]
w(f'  Chronoage/EN に絞る {len(ta):,} 行 / {TV} 欠損 {int(ta[TV].isna().sum())}')

pace = ta.groupby('Sample').agg(pace=(TV, 'median'), age=('age', 'first')).reset_index()
pace.columns = ['donor', 'pace', 'age_t']
w(f'  ドナー {len(pace)} / pace 中央 {pace.pace.median():+.3f} / '
  f'範囲 {pace.pace.min():+.2f}〜{pace.pace.max():+.2f}')
r = spearmanr(pace.age_t, pace.pace)
w(f'  参考: pace × 暦年齢 rho {r.statistic:+.3f}（tAge は年齢相関で評価しない。傾き診断のみ）')

g = pd.read_csv(f'{D}/results/cima_part4a/groups_with_share.csv')
g = g[g.n_cells >= 50]

dn = g.groupby('donor').agg(distal_frag=('distal_frag', 'sum'), prox_frag=('prox_frag', 'sum'),
                            total_frag=('total_frag', 'sum'), age=('age', 'first')).reset_index()
dn['distal_share'] = dn.distal_frag / (dn.distal_frag + dn.prox_frag)
dn['log_depth'] = np.log10(dn.total_frag + 1)
m = dn.merge(pace, on='donor', how='inner')
w(f'\nATAC と RNA の両方があるドナー {len(m)}（ATAC {dn.donor.nunique()} / RNA {pace.donor.nunique()}）')

w('\n[ドナー単位] 遠位割合 × 老化速度')
r1 = spearmanr(m.distal_share, m.pace)
ra = partial(m.distal_share, m.pace, [m.age])
rd = partial(m.distal_share, m.pace, [m.log_depth])
r0 = spearmanr(m.distal_share, m.age)
w(f'  遠位割合 × 老化速度            rho {r1.statistic:+.3f} (p={r1.pvalue:.2g})   '
  f'[AIDA 転写では +0.711]')
w(f'    暦年齢を除いた偏相関         rho {ra.statistic:+.3f} (p={ra.pvalue:.2g})   [AIDA +0.720]')
w(f'    深度を除いた偏相関           rho {rd.statistic:+.3f} (p={rd.pvalue:.2g})   [AIDA +0.713]')
w(f'  参考 遠位割合 × 暦年齢         rho {r0.statistic:+.3f} (p={r0.pvalue:.2g})   [AIDA +0.228]')

z = np.polyfit(m.age, m.pace, 1)
m['pace_resid'] = m.pace - np.polyval(z, m.age)
r2 = spearmanr(m.distal_share, m.pace_resid)
w(f'  （参考）手順書式 pace = resid(tAge~age) では rho {r2.statistic:+.3f} (p={r2.pvalue:.2g})')

w('\n[細胞種単位] その細胞種の遠位割合 × その細胞種の tAge')
ct = ta[['Sample', 'CellType', TV]].rename(columns={'Sample': 'donor', 'CellType': 'subset',
                                                    TV: 'pace_ct'})
gg = g.merge(ct, on=['donor', 'subset'], how='inner')
rows = []
for s, q in gg.groupby('subset'):
    if len(q) < 30: continue
    rr = spearmanr(q.distal_share, q.pace_ct)
    rp = partial(q.distal_share.values, q.pace_ct.values, [q.age.values, q.log_depth.values])
    rows.append(dict(subset=s, n=len(q), rho=rr.statistic, p=rr.pvalue,
                     rho_partial=rp.statistic, p_partial=rp.pvalue))
    w(f'  {s:14s} n={len(q):4d}  rho {rr.statistic:+.3f} (p={rr.pvalue:.2g})  '
      f'年齢と深度を除くと {rp.statistic:+.3f} (p={rp.pvalue:.2g})')
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/per_subset.csv', index=False)
if len(R):
    w(f'  正の subset {int((R.rho>0).sum())}/{len(R)} / '
      f'除いても正 {int((R.rho_partial>0).sum())}/{len(R)}')
m.to_csv(f'{OUT}/donor_level.csv', index=False)
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')


import os, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

D = '${PAPER_DIR}'
P3 = f'{D}/results/cima_part3'
CW = f'{D}/results/cima_crosswalk'
OUT = f'{D}/results/cima_setscore'
os.makedirs(OUT, exist_ok=True)
TV = 'yugene_diff_EN_tAge'
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

def partial(x, y, z):
    rx, ry = rankdata(x), rankdata(y)
    Z = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in z])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return spearmanr(ex, ey)

gm = pd.read_csv(f'{P3}/groups.csv')
pk = pd.read_csv(f'{P3}/peaks.csv')
PB = np.load(f'{P3}/pseudobulk.npy', mmap_mode='r')
PIDX = {p: i for i, p in enumerate(pk.peak)}

fx = pd.read_csv(f'{CW}/frozen_x_cima.csv')
bg = pd.read_csv(f'{CW}/background_x_cima_v2.csv')
sets = {}
sets['up'] = sorted(set(fx[(fx.program == 'up_distal') & fx.cima_peak.notna()].cima_peak))
sets['down'] = sorted(set(fx[(fx.program == 'down_proximal') & fx.cima_peak.notna()].cima_peak))

by = {}
for c, g in pk.groupby('chrom'):
    g = g.sort_values('start')
    by[c] = (g.start.values, g.end.values, g.peak.values)
def peak_of(chrom, sm):
    d = by.get(chrom)
    if d is None: return None
    i = np.searchsorted(d[0], sm, side='right') - 1
    for j in (i, i - 1):
        if 0 <= j < len(d[0]) and d[0][j] <= sm < d[1][j]: return d[2][j]
    return None
bg = bg[bg.cima_hit]
bg['peak'] = [peak_of(c, s) for c, s in zip(bg.chrom, bg.summit)]
sets['bg_distal'] = sorted(set(bg[bg.proximity == 'distal'].peak.dropna()))
sets['bg_proximal'] = sorted(set(bg[bg.proximity == 'proximal'].peak.dropna()))
for k, v in sets.items():
    w(f'{k:12s} ピーク {len(v):,}')

IDX = {k: np.array([PIDX[p] for p in v if p in PIDX]) for k, v in sets.items()}
tot = np.maximum(gm.total_frag.values, 1)
S = {}
for k, idx in IDX.items():
    S[k] = np.asarray(PB[:, idx], np.float64).mean(1) / tot * 1e6
gm['score_up'] = S['up'] - S['bg_distal']
gm['score_down'] = S['down'] - S['bg_proximal']
gm['diff'] = gm.score_up - gm.score_down
gm['log_depth'] = np.log10(gm.total_frag + 1)

f = sorted(glob.glob(f'{D}/results/cima_tage/tage_all_combined_*.csv'))[-1]
ta = pd.read_csv(f)
ta = ta[(ta.model_type == 'Chronoage') & (ta['mode'] == 'EN')]
ct = ta[['Sample', 'CellType', TV]].rename(columns={'Sample': 'donor', 'CellType': 'subset',
                                                    TV: 'pace_ct'})
g = gm[gm.n_cells >= 50].merge(ct, on=['donor', 'subset'], how='inner')
w(f'\nグループ {len(g):,}（n_cells>=50 かつ tAge あり）')

w('\n[上昇腕の獲得 score_up × pace]  ← Fig.4h の直接対応物')
rows = []
for s, q in g.groupby('subset'):
    if len(q) < 30: continue
    r_up = spearmanr(q.score_up, q.pace_ct)
    rp_up = partial(q.score_up.values, q.pace_ct.values, [q.age.values, q.log_depth.values])
    r_dn = spearmanr(q.score_down, q.pace_ct)
    r_df = spearmanr(q['diff'], q.pace_ct)
    ra_up = spearmanr(q.score_up, q.age)
    rows.append(dict(subset=s, n=len(q),
                     rho_up=r_up.statistic, p_up=r_up.pvalue,
                     rho_up_partial=rp_up.statistic, p_up_partial=rp_up.pvalue,
                     rho_down=r_dn.statistic, p_down=r_dn.pvalue,
                     rho_diff=r_df.statistic, p_diff=r_df.pvalue,
                     rho_up_age=ra_up.statistic))
    w(f'  {s:14s} n={len(q):4d}  上昇 {r_up.statistic:+.3f} (p={r_up.pvalue:.2g}, '
      f'年齢+深度偏 {rp_up.statistic:+.3f})  低下 {r_dn.statistic:+.3f}  '
      f'差分 {r_df.statistic:+.3f}  | 上昇×暦年齢 {ra_up.statistic:+.3f}')
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/per_subset.csv', index=False)
w(f'\n  上昇腕が正: {int((R.rho_up>0).sum())}/{len(R)} / 低下腕が負: {int((R.rho_down<0).sum())}/{len(R)}')
w('  AIDA 参照（Fig.4h の趣旨）: CD4_Memory は上昇プログラムをほとんど獲得しない')
gm.to_csv(f'{OUT}/groups_with_scores.csv', index=False)
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')


import os, gzip
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

D = '${PAPER_DIR}'
P3 = f'{D}/results/cima_part3'
OUT = f'{D}/results/cima_poscontrol'
os.makedirs(OUT, exist_ok=True)
GTF = '${DATA_DIR}/adachi/reference/GRCh38.113.gtf.gz'
GENES = ['CR2', 'CD248']
WIN = 5000
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

def partial(x, y, z):
    rx, ry = rankdata(x), rankdata(y)
    Z = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in z])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return spearmanr(ex, ey)

gm = pd.read_csv(f'{P3}/groups.csv')
pk = pd.read_csv(f'{P3}/peaks.csv')
PB = np.load(f'{P3}/pseudobulk.npy', mmap_mode='r')
w(f'擬似バルク {PB.shape} / グループ {len(gm):,} / ピーク {len(pk):,}')
pace = pd.read_csv(f'{D}/results/cima_part4b/donor_level.csv')[['donor', 'pace']]

tss = {}
with gzip.open(GTF, 'rt') as f:
    for line in f:
        if line[0] == '#': continue
        p = line.split('\t')
        if p[2] != 'gene': continue
        for g in GENES:
            if f'gene_name "{g}"' in p[8]:
                c = p[0] if p[0].startswith('chr') else 'chr' + p[0]
                tss[g] = (c, int(p[3]) if p[6] == '+' else int(p[4]))
w('TSS: ' + ', '.join(f'{g} {tss[g][0]}:{tss[g][1]:,}' for g in tss))

targets = {}

fx = pd.read_csv(f'{D}/results/cima_crosswalk/frozen_x_cima.csv')
jo = pd.read_csv(f'{D}/results/gene_to_cre/cre_gene_joined.csv',
                 usecols=['cre_id', 'near_gene']).drop_duplicates('cre_id')
fx = fx.merge(jo, on='cre_id', how='left')
for g in GENES:
    s = fx[(fx.near_gene == g) & fx.cima_peak.notna()]
    if len(s): targets[f'{g}_our_element'] = s.cima_peak.tolist()

for g, (c, t) in tss.items():
    s = pk[(pk.chrom == c) & (pk.start >= t - WIN) & (pk.end <= t + WIN)]
    if len(s): targets[f'{g}_TSS±5kb'] = s.peak.tolist()

PIDX = {p: i for i, p in enumerate(pk.peak)}
w('')
rows = []
for name, plist in targets.items():
    idx = [PIDX[p] for p in plist if p in PIDX]
    if not idx: w(f'{name}: 該当ピーク無し'); continue
    sig = np.asarray(PB[:, idx], np.float64).sum(1)
    cpm = sig / np.maximum(gm.total_frag.values, 1) * 1e6
    q = gm.assign(cpm=cpm)
    q = q[q.n_cells >= 50]
    w(f'=== {name}  ピーク {len(idx)} 個 ===')

    dn = q.groupby('donor').apply(
        lambda t: pd.Series(dict(cpm=(t.cpm * t.total_frag).sum() / t.total_frag.sum(),
                                 age=t.age.iloc[0], depth=np.log10(t.total_frag.sum()))),
        include_groups=False).reset_index()
    dn = dn.merge(pace, on='donor', how='left')
    r = spearmanr(dn.age, dn.cpm)
    rp = partial(dn.age.values, dn.cpm.values, [dn.depth.values])
    rpa = spearmanr(dn.pace.dropna(), dn.cpm[dn.pace.notna()])
    w(f'  ドナー単位 n={len(dn)}  × 暦年齢 rho {r.statistic:+.3f} (p={r.pvalue:.2g})  '
      f'深度を除くと {rp.statistic:+.3f}')
    w(f'                          × 老化速度 rho {rpa.statistic:+.3f} (p={rpa.pvalue:.2g})')
    rows.append(dict(target=name, n_peaks=len(idx), level='donor', n=len(dn),
                     rho_age=r.statistic, p_age=r.pvalue,
                     rho_age_depthpartial=rp.statistic, rho_pace=rpa.statistic))
    for s, t in q.groupby('subset'):
        if len(t) < 30: continue
        rr = spearmanr(t.age, t.cpm)
        w(f'    {s:14s} n={len(t):4d}  rho {rr.statistic:+.3f} (p={rr.pvalue:.2g})  '
          f'CPM 中央 {t.cpm.median():7.1f}')
        rows.append(dict(target=name, n_peaks=len(idx), level=s, n=len(t),
                         rho_age=rr.statistic, p_age=rr.pvalue))
    w('')
pd.DataFrame(rows).to_csv(f'{OUT}/cr2_cd248.csv', index=False)
w('参照: Morandini 2024 (GSE193142, 159 名) CR2 r=-0.59 / '
  '我々の Fig.5a CR2 -0.32→-0.70, CD248 -0.41→-0.60')
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')


import os, pickle
import numpy as np, pandas as pd

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
OUT = f'{D}/results/figures_data/figures_data'
TV = 'yugene_diff_EN_tAge'
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

rob = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv')
rob = rob[rob.program.isin(['up_distal', 'down_proximal'])].drop_duplicates('cre_id')
UP = sorted(set(rob[rob.program == 'up_distal'].cre_id))
DN = sorted(set(rob[rob.program == 'down_proximal'].cre_id))
bg = pd.read_csv(f'{D}/results/cima_crosswalk/background_summits_v2.csv')
BGD = sorted(set(bg[bg.proximity == 'distal'].cre_id))
BGP = sorted(set(bg[bg.proximity == 'proximal'].cre_id))
w(f'上昇遠位 {len(UP)} / 低下近位 {len(DN)} / 背景 遠位 {len(BGD):,} 近位 {len(BGP):,}')

def score_cohort(tag, pkl, key_cols, meta_fill):
    w(f'\n{tag}: キャッシュを開く')
    d = pickle.load(open(pkl, 'rb'))
    cres = np.array(d['cres']); keys = list(d['expr_sum'].keys())
    idx = {c: i for i, c in enumerate(cres)}
    I = {n: np.array([idx[c] for c in ids if c in idx])
         for n, ids in [('up', UP), ('dn', DN), ('bgd', BGD), ('bgp', BGP)]}
    w('  ' + ' / '.join(f'{n} {len(v)}' for n, v in I.items()))
    rows = []
    for k in keys:
        v = d['expr_sum'][k].astype(np.float64)
        tot = v.sum()
        if tot <= 0: continue
        cpm = 1e6 / tot
        rows.append(list(k) + [int(d['n_cells'][k]), float(tot),
                               float(v[I['up']].mean() * cpm), float(v[I['dn']].mean() * cpm),
                               float(v[I['bgd']].mean() * cpm), float(v[I['bgp']].mean() * cpm)])
    del d
    m = pd.DataFrame(rows, columns=key_cols + ['n_cells', 'total_counts',
                                               'up_raw', 'dn_raw', 'bgd_raw', 'bgp_raw'])
    m['score_up'] = m.up_raw - m.bgd_raw
    m['score_down'] = m.dn_raw - m.bgp_raw
    m['diff'] = m.score_up - m.score_down
    return meta_fill(m)

a = pd.read_csv('${DATA_DIR}/lnc_project_AIDA_dataset/by_subcluster/tage/'
                'tage_all_combined_v1.csv')
a = a[(a.model_type == 'Chronoage') & (a['mode'] == 'EN')]
A1 = a.drop_duplicates('Sample').set_index('Sample')[['age', 'sex', 'country']]
AT = a.set_index(['Sample', 'Subcluster'])[TV].to_dict()
AP = a.groupby('Sample')[TV].median().to_dict()
def fill_aida(m):
    m = m.rename(columns={'donor_': 'donor'})
    m['visit'] = 1
    m = m.join(A1, on='donor')
    m['site'] = m.country
    m['tAge'] = [AT.get((dn, sb), np.nan) for dn, sb in zip(m.donor, m.subset)]
    m['pace'] = m.donor.map(AP)
    return m
MA = score_cohort('AIDA', f'{OUR}/aggregation_cache_aida_subcluster.pkl',
                  ['cohort_k', 'donor_', 'lib', 'subset'], fill_aida)
MA = MA[MA.n_cells >= 10]
cols = ['donor', 'visit', 'subset', 'n_cells', 'total_counts', 'age', 'sex', 'site',
        'tAge', 'pace', 'score_up', 'score_down', 'diff']
MA[cols].to_csv(f'{OUT}/aida_setscores.csv', index=False)
w(f'AIDA: {len(MA):,} 行 / ドナー {MA.donor.nunique()} を書き出し')

s = pd.read_csv('${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/'
                'subcluster/tage_all_combined_v2.csv')
s = s[(s.model_type == 'Chronoage') & (s['mode'] == 'EN')]
vis = s[['Donor_id', 'Batch', 'Age']].drop_duplicates().sort_values(['Donor_id', 'Age'])
vis['visit'] = vis.groupby('Donor_id').cumcount() + 1
VIS = {(r.Donor_id, r.Batch): (int(r.visit), int(r.Age)) for _, r in vis.iterrows()}
SEX = dict(zip(s.drop_duplicates('Donor_id').Donor_id, s.drop_duplicates('Donor_id').Sex))
ST = s.set_index(['Donor_id', 'Batch', 'CellType'])[TV].to_dict()
SP = s.groupby(['Donor_id', 'Batch'])[TV].median().to_dict()
def fill_syn(m):
    m = m.rename(columns={'donor_': 'donor'})
    vv = [VIS.get((dn, lb), (np.nan, np.nan)) for dn, lb in zip(m.donor, m.lib)]
    m['visit'] = [x[0] for x in vv]; m['age'] = [x[1] for x in vv]
    m['sex'] = m.donor.map(SEX); m['site'] = ''
    m['tAge'] = [ST.get((dn, lb, sb), np.nan) for dn, lb, sb in zip(m.donor, m.lib, m.subset)]
    m['pace'] = [SP.get((dn, lb), np.nan) for dn, lb in zip(m.donor, m.lib)]
    return m
MS = score_cohort('Terekhova', f'{OUR}/aggregation_cache_synapse_subcluster.pkl',
                  ['cohort_k', 'donor_', 'lib', 'subset'], fill_syn)
MS = MS[(MS.n_cells >= 10) & MS.visit.notna()]
MS[cols].to_csv(f'{OUT}/terekhova_setscores.csv', index=False)
w(f'Terekhova: {len(MS):,} 行 / ドナー {MS.donor.nunique()}（全訪問。初回= visit 1）を書き出し')

for tag, m in [('AIDA', MA), ('Terekhova', MS[MS.visit == 1])]:
    m = m[m.pace.notna()]
    t1, t2 = m.drop_duplicates('donor').pace.quantile([1/3, 2/3])
    f = m[m.pace >= t2]; sl = m[m.pace <= t1]
    w(f'  {tag}: score_up 中央 速 {f.score_up.median():+.3f} 対 遅 {sl.score_up.median():+.3f} / '
      f'score_down 速 {f.score_down.median():+.3f} 対 遅 {sl.score_down.median():+.3f}')
open(f'{OUT}/../extract_setscores.log', 'w').write('\n'.join(L))
print('-> done')


import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

D = '${PAPER_DIR}'
W = f'{D}/results/figures_data'
NW = f'{D}/results/fig3g_canonical_tertile'
SPLIT = os.environ['SPLIT']; assert SPLIT in ('v4_raw', 'canonical')
OD = f'{NW}/{SPLIT}'; os.makedirs(OD, exist_ok=True)
F3I = pd.read_csv(f'{D}/results/srcdata_fig2f_fig3i_canonical/Fig3i.csv',
                  usecols=['cohort', 'donor', 'pace', 'pace_adj', 'pace_tertile']).drop_duplicates(['cohort', 'donor'])
CANON = {c: g.set_index('donor') for c, g in F3I.groupby('cohort')}
LOCI = {'chr1': ('CR2', 207449328, 207459329), 'chr11': ('CD248', 66312043, 66322044)}
WANT = {'CR2': ['CD4_Naive', 'B_Naive'], 'CD248': ['CD4_Naive', 'CD14_Mono']}
EL = {'CR2': ('chr1', 207453600, 207454881, '+'), 'CD248': ('chr11', 66316879, 66318362, '-')}
NREP = 20
FLOOR_PCT = 10
rng = np.random.default_rng(0)
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

bm = pd.read_parquet(f'{D}/results/canonical_labels/barcode_subset_canonical.parquet',
                     columns=['cohort', 'sample', 'barcode_full', 'Donor_id', 'subset_canonical'])
bm = bm.rename(columns={'subset_canonical': 'subcluster'})
P = pd.read_csv(f'{D}/results/band_pace/donor_pace.csv')
ssA = pd.read_csv(f'{W}/figures_data/aida_setscores.csv')
ssT = pd.read_csv(f'{W}/figures_data/terekhova_setscores.csv'); ssT = ssT[ssT.visit == 1]

out, dono, qc = [], [], []

def thin(counts, tot, floor):
    p = min(1.0, floor / tot)
    if p >= 1.0: return counts.astype(float)
    return np.vstack([rng.binomial(counts.astype(int), p) for _ in range(NREP)]).mean(0)

def run(tag, winfile, ss):
    tw = pd.read_csv(winfile, sep='\t', header=None,
                     names=['sample', 'chr', 'pos', 'barcode', 'count', 'strand'])
    b = bm[bm.cohort == ('AIDA' if tag == 'AIDA' else 'Synapse')].set_index('barcode_full')[['Donor_id', 'subcluster']]
    tw['barcode_full'] = tw['sample'] + '_' + tw.barcode
    tw = tw.join(b, on='barcode_full', how='inner')
    pc = P[P.cohort == tag].set_index('donor').pace if SPLIT == 'v4_raw' else CANON[tag].pace
    TOT = ss.set_index(['donor', 'subset']).total_counts.to_dict()
    for ch, (locus, lo, hi) in LOCI.items():
        for sub in WANT[locus]:
            alld = sorted({d for (d, s2) in TOT if s2 == sub and d in pc.index})
            tots = np.array([TOT[(d, sub)] for d in alld])
            fmin = np.percentile(tots, FLOOR_PCT)
            keep = [d for d, t in zip(alld, tots) if t >= fmin]
            floor = int(min(TOT[(d, sub)] for d in keep))
            w(f'\n{tag} {locus} {sub}: ドナー {len(alld)} → 床 {floor:,} で {len(keep)} 名採用'
              f'（除外 {len(alld)-len(keep)}）')
            q = tw[(tw.chr == ch) & (tw.subcluster == sub) & (tw.Donor_id.isin(keep))]
            per = q.groupby(['Donor_id', 'pos'])['count'].sum().reset_index()

            recs = []
            for dn, g in per.groupby('Donor_id'):
                v = thin(g['count'].values, TOT[(dn, sub)], floor)
                recs.append(pd.DataFrame(dict(Donor_id=dn, pos=g.pos.values, ds=v)))
            per2 = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=['Donor_id','pos','ds'])
            pcv = pd.Series({d: pc[d] for d in keep})
            if SPLIT == 'v4_raw':
                t1, t2 = pcv.quantile([1/3, 2/3])
                grp = {d: ('slow' if v <= t1 else 'fast' if v >= t2 else 'mid') for d, v in pcv.items()}
            else:
                grp = CANON[tag].pace_tertile.reindex(keep).to_dict()
            per2['group'] = per2.Donor_id.map(grp)

            ech, est, een, estr = EL[locus]
            dtot = per2.groupby('Donor_id').ds.sum().reindex(keep).fillna(0)
            rho = spearmanr(pcv.reindex(dtot.index), dtot).statistic
            rho_adj = spearmanr(CANON[tag].pace_adj.reindex(dtot.index), dtot).statistic
            for gname in ['slow', 'fast']:
                dn_g = [d for d in keep if grp[d] == gname]
                if not dn_g: continue
                sel = per2[per2.group == gname]
                agg = sel.groupby('pos').ds.sum() / len(dn_g)
                for pos, v in agg.items():
                    out.append(dict(cohort=tag, locus=locus, subset=sub, group=gname,
                                    position=int(pos), value=float(v)))
                for d in dn_g:
                    dono.append(dict(cohort=tag, locus=locus, subset=sub, group=gname,
                                     donor=d, donor_downsampled_reads=float(dtot.get(d, 0))))
            f_ = dtot[[d for d in dtot.index if grp[d]=='fast']].mean()
            s_ = dtot[[d for d in dtot.index if grp[d]=='slow']].mean()
            w(f'  窓内 平均読み数 slow {s_:.2f} / fast {f_:.2f} → 比 {f_/s_ if s_ else np.nan:.2f}'
              f'  |  rho(pace) {rho:+.3f}')
            qc.append(dict(cohort=tag, locus=locus, subset=sub, floor=floor,
                           n_donors=len(keep), n_excluded=len(alld)-len(keep),
                           mean_slow=s_, mean_fast=f_, ratio=f_/s_ if s_ else np.nan, rho_pace=rho, rho_pace_adj=rho_adj))

run('AIDA', f'{NW}/trackwin/aida_windows.tsv', ssA)
run('Terekhova', f'{NW}/trackwin/terekhova_windows.tsv', ssT)

P3 = f'{D}/results/cima_part3'
gm = pd.read_csv(f'{P3}/groups.csv'); pk = pd.read_csv(f'{P3}/peaks.csv')
PB = np.load(f'{P3}/pseudobulk.npy', mmap_mode='r')
cl = pd.read_csv(f'{D}/results/cima_part4b/donor_level.csv')
pcC = cl.set_index('donor').pace if SPLIT == 'v4_raw' else CANON['CIMA'].pace
BIN = 150
for ch, (locus, lo, hi) in LOCI.items():
    sel = pk[(pk.chrom == ch) & (pk.end > lo) & (pk.start < hi)]
    for sub in WANT[locus]:
        rows = gm[(gm.subset == sub) & (gm.n_cells >= 50) & (gm.donor.isin(pcC.index))]
        if not len(rows): continue
        tots = rows.total_frag.values
        fmin = np.percentile(tots, FLOOR_PCT)
        k = rows[rows.total_frag >= fmin]
        floor = int(k.total_frag.min())
        w(f'\nCIMA {locus} {sub}: ドナー {len(rows)} → 床 {floor:,} で {len(k)} 名採用'
          f'（除外 {len(rows)-len(k)}）')
        M = np.asarray(PB[np.ix_(k.index.values, sel.index.values)], np.float64)
        DS = np.vstack([thin(M[i], k.total_frag.values[i], floor) for i in range(len(k))])
        pcv = pcC.reindex(k.donor.values)
        if SPLIT == 'v4_raw':
            t1, t2 = pcv.quantile([1/3, 2/3])
            gl = np.where(pcv.values <= t1, 'slow', np.where(pcv.values >= t2, 'fast', 'mid'))
        else:
            gl = CANON['CIMA'].pace_tertile.reindex(k.donor.values).values
        dtot = DS.sum(1)
        rho = spearmanr(pcv.values, dtot).statistic
        rho_adj = spearmanr(CANON['CIMA'].pace_adj.reindex(k.donor.values).values, dtot).statistic
        for gname in ['slow', 'fast']:
            m = gl == gname
            if not m.any(): continue
            mean_sig = DS[m].mean(0)
            for (_, prow), v in zip(sel.iterrows(), mean_sig):
                for pos in range(int(prow.start)//BIN*BIN, int(prow.end), BIN):
                    if lo <= pos <= hi:
                        out.append(dict(cohort='CIMA', locus=locus, subset=sub, group=gname,
                                        position=pos, value=float(v)))
            for dn, tv in zip(k.donor.values[m], dtot[m]):
                dono.append(dict(cohort='CIMA', locus=locus, subset=sub, group=gname,
                                 donor=dn, donor_downsampled_reads=float(tv)))
        f_ = dtot[gl=='fast'].mean(); s_ = dtot[gl=='slow'].mean()
        w(f'  窓内 平均断片数 slow {s_:.2f} / fast {f_:.2f} → 比 {f_/s_:.2f}  |  rho(pace) {rho:+.3f}')
        qc.append(dict(cohort='CIMA', locus=locus, subset=sub, floor=floor, n_donors=len(k),
                       n_excluded=len(rows)-len(k), mean_slow=s_, mean_fast=f_,
                       ratio=f_/s_, rho_pace=rho, rho_pace_adj=rho_adj))

O = pd.DataFrame(out); O.to_csv(f'{OD}/locus_tracks.csv', index=False)
pd.DataFrame(dono).to_csv(f'{OD}/locus_tracks_donor.csv', index=False)
Q = pd.DataFrame(qc); Q.to_csv(f'{OD}/locus_tracks_qc.csv', index=False)
w('\n=== 深度そろえ後の fast/slow 比（1 未満 = 速い人で低い）===')
w(f'{"座位 / 細胞種":26s}{"AIDA":>9s}{"Terekhova":>11s}{"CIMA":>8s}')
for locus in ['CR2','CD248']:
    for sub in WANT[locus]:
        line=f'{locus+" / "+sub:26s}'
        for co,wd in [('AIDA',9),('Terekhova',11),('CIMA',8)]:
            r=Q[(Q.cohort==co)&(Q.locus==locus)&(Q['subset']==sub)]
            line+=f'{(r.ratio.iloc[0] if len(r) else np.nan):>{wd}.2f}'
        w(line)
w(f'\n期待どおり（比<1）の組み合わせ: {int((Q.ratio<1).sum())}/{len(Q)}')
w(f'rho(pace) が負: {int((Q.rho_pace<0).sum())}/{len(Q)}')
open(f'{OD}/extract_tracks_v5.log','w').write('\n'.join(L))
print('-> done')


import os, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
OUT = f'{D}/results/fig7a_distal_share'
os.makedirs(OUT, exist_ok=True)
ann = pd.read_csv(f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv',
                  usecols=['cre_id', 'proximity']).drop_duplicates('cre_id')
prox = dict(zip(ann.cre_id, ann.proximity))
f3 = pd.read_csv(f'{D}/results/srcdata_fig2f_fig3i_canonical/Fig3i.csv',
                 usecols=['cohort', 'donor', 'visit', 'subset', 'age', 'pace', 'pace_adj']).drop_duplicates(['cohort', 'donor', 'subset'])
ts = pd.read_csv(f'{D}/results/tage_v110_check/tage_all_combined_v4.csv',
                 usecols=['Donor_id', 'Batch', 'Age'])
base = ts.drop_duplicates().sort_values(['Donor_id', 'Age']).groupby('Donor_id').first().reset_index()
BK = set(zip(base.Donor_id, base.Batch))

def share(tag, pkl, first_only):
    d = pickle.load(open(pkl, 'rb'))
    cres = np.array(d['cres']); isd = np.array([prox.get(c, '') == 'distal' for c in cres])
    rows = []
    for k in d['expr_sum']:
        coh, don, bat, ct = k
        if first_only and (don, bat) not in BK: continue
        if d['n_cells'][k] < 10: continue
        v = np.asarray(d['expr_sum'][k], np.float64); tot = v.sum()
        if tot < 1: continue
        rows.append(dict(cohort=tag, donor=don, subset=ct, distal_share=v[isd].sum() / tot, total_counts=tot))
    del d
    m = pd.DataFrame(rows)
    return m.groupby(['cohort', 'donor', 'subset'], as_index=False).agg(distal_share=('distal_share', 'mean'), total_counts=('total_counts', 'sum'))

A = share('AIDA', f'{OUR}/aggregation_cache_aida_subcluster.pkl', False)
T = share('Terekhova', f'{OUR}/aggregation_cache_synapse_subcluster.pkl', True)
cg = pd.read_csv(f'{D}/results/cima_part4a/groups_with_share.csv')
cg = cg[cg.n_cells >= 50]
Cm = pd.DataFrame(dict(cohort='CIMA', donor=cg.donor, subset=cg['subset'], distal_share=cg.distal_share, total_counts=cg.total_frag))
L = pd.concat([A, T, Cm], ignore_index=True).merge(f3, on=['cohort', 'donor', 'subset'], how='inner')
L.to_csv(f'{OUT}/distal_share_long.csv', index=False)
rows = []
for (c, s), g in L.groupby(['cohort', 'subset']):
    rows.append(dict(cohort=c, subset=s, n=len(g), rho_pace_adj=spearmanr(g.distal_share, g.pace_adj).statistic,
                     p_pace_adj=spearmanr(g.distal_share, g.pace_adj).pvalue,
                     rho_age=spearmanr(g.distal_share, g.age).statistic))
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/fig7a_distal_share_stats.csv', index=False)
for c, g in R.groupby('cohort'):
    g = g[g.subset != 'Other']
    print(c, 'n_ct', len(g), 'rho_pace_adj>0', int((g.rho_pace_adj > 0).sum()), 'range', round(g.rho_pace_adj.min(), 3), round(g.rho_pace_adj.max(), 3),
          '> age', int((g.rho_pace_adj > g.rho_age).sum()))


import h5py, numpy as np, pandas as pd
from pathlib import Path

D = Path.home() / 'aida_tcre/deliver'
M = Path('${DATA_DIR}/cima/meta')
H = '${DATA_DIR}/cima/atac/CIMA_ATAC_3762242cells_338036peaks_compressed.h5ad'

mp = pd.read_csv(M / 'celltype_map.csv')
ex = pd.read_csv(M / 'celltype_excluded.csv')
with h5py.File(H, 'r') as f:
    g = f['obs/cell_type_l4']
    cats = [c.decode() if isinstance(c, bytes) else c for c in g['categories'][:]]
    codes = g['codes'][:]
n = pd.Series(np.bincount(codes[codes >= 0], minlength=len(cats)), index=cats, name='n_cells_atac')
tot = int((codes >= 0).sum())

t = pd.DataFrame({'cima_L4': cats}).merge(n.rename_axis('cima_L4').reset_index(), on='cima_L4')
t = t.merge(mp, on='cima_L4', how='left').merge(ex, on='cima_L4', how='left')
t['status'] = np.where(t.your_subset.notna(), 'mapped', np.where(t.reason.notna(), 'excluded', 'not in map'))
t = t.rename(columns={'your_subset': 'subset'}).sort_values(['status', 'subset', 'n_cells_atac'], ascending=[True, True, False])
t.to_csv(D / '04_methods/cima_L4_to_subset_map.csv', index=False)

mapped = t[t.status == 'mapped']
L = [f'ATAC の L4 ラベル数 {len(cats)}、うち対応表で 11 サブセットに写すもの {len(mapped)}。',
     f'写した細胞 {int(mapped.n_cells_atac.sum()):,} / {tot:,} = {mapped.n_cells_atac.sum() / tot * 100:.1f}%。',
     f'対応表の行数 {len(mp)}（ATAC に無いラベル: {sorted(set(mp.cima_L4) - set(cats))}）。',
     f'対応表にも除外表にも無い ATAC ラベル: {sorted(t[t.status == "not in map"].cima_L4)}。',
     f'対応表と除外表の両方に載るラベル: {sorted(set(mp.cima_L4) & set(ex.cima_L4))}。', '',
     mapped.groupby('subset').agg(n_labels=('cima_L4', 'size'), n_cells=('n_cells_atac', 'sum')).to_string()]
md = ['| CIMA L4 | 11 サブセット | ATAC 細胞数 | 状態 |', '|---|---|---:|---|']
md += [f"| {r.cima_L4} | {r.subset if isinstance(r.subset, str) else '—'} | {r.n_cells_atac:,} | {r.status}{'（' + r.reason + '）' if isinstance(r.reason, str) else ''} |" for r in t.itertuples()]
(D / '04_methods/cima_L4_to_subset_map.md').write_text('\n'.join(L) + '\n\n' + '\n'.join(md) + '\n')
(D / 'logs/02_cima_label_map.txt').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
