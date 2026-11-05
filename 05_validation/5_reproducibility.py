#!/usr/bin/env python3
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, binomtest, fisher_exact

B   = '${PROJECT_DIR}/results'
P   = f'{B}/phase3b_ourcd4'
D   = f'{B}/PAPER_CRE_MECHANISM'
OUT = f'{D}/results/fixes/age_adjusted'
TS  = '_v4'
TV  = 'yugene_diff_EN_tAge'
NE  = f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv'
CD4 = f'{D}/results/cd4_redo/cd4_our_tage.csv'
SYN_TAGE_SC = '${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/subcluster/tage_all_combined_v2.csv'
AIDA_TAGE_SC = '${DATA_DIR}/lnc_project_AIDA_dataset/by_subcluster/tage/tage_all_combined_v1.csv'

L = []
def w(s=''):
    print(s, flush=True); L.append(str(s))

rng = np.random.default_rng(_v5)

w('='*100)
w('課題 B2 段階1 — 交絡の実測、選抜段階別 OR、ブートストラップ CI、暦年齢ベースライン')
w('='*100)

w('\n' + '='*100)
w('(1) 加齢加速軸 (yugene_diff_EN_tAge) と 暦年齢 の Spearman')
w('='*100)

def load_syn_sub(baseline=True):
    s = pd.read_csv(SYN_TAGE_SC)
    s = s[(s.model_type=='Chronoage') & (s['mode']=='EN')].copy()
    s = s.rename(columns={'Age':'age'})
    s['cohort'] = 'Synapse'
    if baseline:
        pa = s[['Donor_id','Batch','age']].drop_duplicates()
        keep = pa.sort_values(['Donor_id','age']).groupby('Donor_id').head(1)
        s = s.merge(keep[['Donor_id','Batch']], on=['Donor_id','Batch'])
    return s[['cohort','Donor_id','Batch','CellType','age','Sex','yugene_diff_EN_tAge','n_cells','Coverage']]

def load_aida_sub():
    a = pd.read_csv(AIDA_TAGE_SC)
    a = a[(a.model_type=='Chronoage') & (a['mode']=='EN')].copy()
    a = a.rename(columns={'Sample':'Donor_id','Subcluster':'CellType','sex':'Sex'})
    a['Batch'] = a['Donor_id']; a['cohort'] = 'AIDA'
    return a[['cohort','Donor_id','Batch','CellType','age','Sex','yugene_diff_EN_tAge','n_cells','Coverage']]

syn_b = load_syn_sub(baseline=True)
aida  = load_aida_sub()
orig  = pd.concat([aida, syn_b], ignore_index=True)
w(f'元 tAge 表 (baseline filter 後): AIDA {len(aida)} 行 / Synapse {len(syn_b)} 行')

our = pd.read_csv(CD4)
o = pd.DataFrame({'cohort':our.cohort,'Donor_id':our.Donor_id,'Batch':our.Batch,
                  'CellType':our.Group,'age':our.age,'Sex':our.Sex,
                  'yugene_diff_EN_tAge':our.yugene_diff_EN_tAge,
                  'n_cells':100,'Coverage':np.nan})
canon = pd.concat([orig[~orig.CellType.isin(['CD4_Naive','CD4_Memory'])], o], ignore_index=True)
w(f'正典 (CD4 差し替え後): {len(canon)} 行')
w(f'  CD4 差し替え分 = {len(o)} 行 (AIDA {int((o.cohort=="AIDA").sum())} / Synapse {int((o.cohort=="Synapse").sum())})')

def conf_table(df, tag):
    rows=[]
    for (co,ct), g in df.groupby(['cohort','CellType']):
        gg = g.dropna(subset=['yugene_diff_EN_tAge','age'])
        if len(gg) < 10: continue
        r,p = spearmanr(gg.yugene_diff_EN_tAge, gg.age)
        rows.append(dict(table=tag, cohort=co, cell_type=ct, n=len(gg),
                         n_donor=gg.Donor_id.nunique(),
                         rho_axis_vs_age=r, pval=p, r2_shared=r**2))
    return pd.DataFrame(rows)

t_orig  = conf_table(orig,  '元 tAge 表')
t_canon = conf_table(canon, '正典 (CD4 差し替え後)')
conf = pd.concat([t_orig, t_canon], ignore_index=True)
conf.to_csv(f'{OUT}/tables/01_confound_axis_vs_age.csv', index=False)

w('\n--- 正典 (CD4 差し替え後 = 実際に CRE 相関で使われた変数) ---')
w(f'{"コホート":>9s} {"細胞種":14s} {"n":>5s} {"rho(軸,暦年齢)":>14s} {"順位分散共有":>10s}')
for _, x in t_canon.sort_values(['cohort','rho_axis_vs_age'], ascending=[True,False]).iterrows():
    w(f'{x.cohort:>9s} {x.cell_type:14s} {int(x.n):5d} {x.rho_axis_vs_age:+14.3f} {100*x.r2_shared:9.1f}%')

w('\n--- 元 tAge 表 (差し替え前) ---')
for _, x in t_orig.sort_values(['cohort','rho_axis_vs_age'], ascending=[True,False]).iterrows():
    w(f'{x.cohort:>9s} {x.cell_type:14s} {int(x.n):5d} {x.rho_axis_vs_age:+14.3f} {100*x.r2_shared:9.1f}%')

w('\n--- 差し替えで交絡は強まったか弱まったか (CD4 の 2 群のみ) ---')
mg = t_orig.merge(t_canon, on=['cohort','cell_type'], suffixes=('_orig','_canon'))
cd4 = mg[mg.cell_type.isin(['CD4_Memory','CD4_Naive'])]
for _, x in cd4.iterrows():
    d = x.rho_axis_vs_age_canon - x.rho_axis_vs_age_orig
    w(f'  {x.cohort:8s} {x.cell_type:12s} 元 {x.rho_axis_vs_age_orig:+.3f} (n={int(x.n_orig)}) -> '
      f'正典 {x.rho_axis_vs_age_canon:+.3f} (n={int(x.n_canon)})  差 {d:+.3f}  '
      f'{"★強まった" if d>0 else "弱まった"}')
w('\n  全細胞種の |rho| 中央値: 元 %.3f / 正典 %.3f' %
  (t_orig.rho_axis_vs_age.abs().median(), t_canon.rho_axis_vs_age.abs().median()))

syn_donor = canon[(canon.cohort=='Synapse')].copy()
w(f'\n  参考: 正典 Synapse の行数 = 細胞種ごとに '
  f'{syn_donor.groupby("CellType").size().to_dict()}')

w('\n' + '='*100)
w('(4) 選抜段階別 OR 表')
w('='*100)
ne = pd.read_csv(NE, usecols=['cre_id','proximity']).dropna()
ne = ne[ne.proximity.isin(['proximal','distal'])].drop_duplicates('cre_id')
prox = dict(zip(ne.cre_id, ne.proximity))
w(f'注釈: {len(ne):,} tCRE (proximal {int((ne.proximity=="proximal").sum()):,} / distal {int((ne.proximity=="distal").sum()):,})')

a = pd.read_csv(f'{P}/cre_corr_aida_subcluster_significant_{TS}.csv',
                usecols=['cre_id','cell_type','target_var','n_samples','spearman_rho','spearman_pval','fdr'])
a = a[(a.target_var==TV) & (a.fdr<0.05)].copy()
a['prox'] = a.cre_id.map(prox); a = a.dropna(subset=['prox'])
a['dir_aida'] = np.where(a.spearman_rho>0,'up','down')
w(f'AIDA 選抜 (加速軸 FDR<0.05, 近位/遠位注釈あり): {len(a):,} 行 / 異なり tCRE {a.cre_id.nunique():,}')

s_full = pd.read_csv(f'{P}/cre_corr_synapse_subcluster_full_{TS}.csv',
                     usecols=['cre_id','cell_type','target_var','spearman_rho','fdr'])
s_full = s_full[s_full.target_var==TV].rename(columns={'spearman_rho':'rho_syn','fdr':'fdr_syn'})
m = a.merge(s_full[['cre_id','cell_type','rho_syn','fdr_syn']], on=['cre_id','cell_type'], how='left')
m['dir_syn'] = np.where(m.rho_syn>0,'up',np.where(m.rho_syn<0,'down',None))
w(f'Synapse 側 rho が付いた行: {int(m.rho_syn.notna().sum()):,} (欠損 {int(m.rho_syn.isna().sum()):,})')

def or2x2(df, dircol='dir_aida'):
    du = int(((df.prox=='distal') & (df[dircol]=='up')).sum())
    dd = int(((df.prox=='distal') & (df[dircol]=='down')).sum())
    pu = int(((df.prox=='proximal') & (df[dircol]=='up')).sum())
    pd_ = int(((df.prox=='proximal') & (df[dircol]=='down')).sum())
    orv, p = fisher_exact([[du,dd],[pu,pd_]])
    return dict(n_distal=du+dd, pct_distal_up=100*du/max(du+dd,1),
                n_prox=pu+pd_, pct_prox_up=100*pu/max(pu+pd_,1),
                OR=orv, fisher_p=p, du=du, dd=dd, pu=pu, pd=pd_)

stages = {}
stages['AIDA 選抜のみ']   = or2x2(m)
same = m.dropna(subset=['rho_syn'])
same = same[same.dir_aida==same.dir_syn]
stages['符号一致のみ']     = or2x2(same)
rob = m.dropna(subset=['rho_syn'])
rob = rob[(rob.fdr_syn<0.05) & (rob.dir_aida==rob.dir_syn)]
stages['両コホート頑健']   = or2x2(rob)

st = pd.DataFrame(stages).T.reset_index().rename(columns={'index':'選抜段階'})
st.to_csv(f'{OUT}/tables/04_or_by_selection_stage.csv', index=False)
w(f'\n{"選抜段階":16s}{"遠位 n":>10s}{"遠位上昇率":>10s}{"近位 n":>10s}{"近位上昇率":>10s}{"OR":>9s}')
for _, x in st.iterrows():
    w(f'{x["選抜段階"]:16s}{int(x.n_distal):10,d}{x.pct_distal_up:9.1f}%{int(x.n_prox):10,d}'
      f'{x.pct_prox_up:9.1f}%{x.OR:9.2f}')
w(f'\n頑健セット: 延べ {len(rob):,} ペア / 異なり tCRE {rob.cre_id.nunique():,}')

w('\n--- 方向別 生存率: P(頑健まで生き残る | 近位×上昇) 対 P(... | 近位×低下) ---')
mm = m.dropna(subset=['rho_syn']).copy()
robkey = set(zip(rob.cre_id, rob.cell_type))
mm['survived'] = [ (c,t) in robkey for c,t in zip(mm.cre_id, mm.cell_type) ]
surv = mm.groupby(['prox','dir_aida']).agg(n=('survived','size'), n_surv=('survived','sum')).reset_index()
surv['pct_survive'] = 100*surv.n_surv/surv.n
surv.to_csv(f'{OUT}/tables/04b_survival_by_direction.csv', index=False)
for _, x in surv.iterrows():
    w(f'  {x.prox:9s} × {x.dir_aida:5s}: {int(x.n_surv):6,d} / {int(x.n):8,d} = {x.pct_survive:6.3f}%')
pv = surv.set_index(['prox','dir_aida']).pct_survive
w(f'\n  近位: 低下 / 上昇 = {pv[("proximal","down")]/pv[("proximal","up")]:.2f} 倍')
w(f'  遠位: 上昇 / 低下 = {pv[("distal","up")]/pv[("distal","down")]:.2f} 倍')
_or, _p = fisher_exact([[int(surv.query("prox=='proximal' and dir_aida=='down'").n_surv.iloc[0]),
                         int(surv.query("prox=='proximal' and dir_aida=='down'").n.iloc[0]-surv.query("prox=='proximal' and dir_aida=='down'").n_surv.iloc[0])],
                        [int(surv.query("prox=='proximal' and dir_aida=='up'").n_surv.iloc[0]),
                         int(surv.query("prox=='proximal' and dir_aida=='up'").n.iloc[0]-surv.query("prox=='proximal' and dir_aida=='up'").n_surv.iloc[0])]])
w(f'  近位 低下 vs 上昇 の生存 OR = {_or:.2f} (p={_p:.3e})')

w('\n' + '='*100)
w('(5) プール一致率の細胞種ブートストラップ CI')
w('='*100)
mo = m.dropna(subset=['rho_syn']).copy()
per = []
for ct, g in mo.groupby('cell_type'):
    ud = g[(g.dir_aida=='up') & (g.prox=='distal')]
    dp = g[(g.dir_aida=='down') & (g.prox=='proximal')]
    per.append(dict(cell_type=ct, n_updist=len(ud), pct_updist=100*(ud.dir_syn=='up').mean() if len(ud) else np.nan,
                    k_updist=int((ud.dir_syn=='up').sum()),
                    n_downprox=len(dp), pct_downprox=100*(dp.dir_syn=='down').mean() if len(dp) else np.nan,
                    k_downprox=int((dp.dir_syn=='down').sum())))
per = pd.DataFrame(per)
per.to_csv(f'{OUT}/tables/05_percelltype_concordance.csv', index=False)
w(f'{"細胞種":14s}{"上昇×遠位 n":>13s}{"一致%":>8s}{"低下×近位 n":>13s}{"一致%":>8s}')
for _, x in per.iterrows():
    w(f'{x.cell_type:14s}{int(x.n_updist):13,d}{x.pct_updist:7.1f}%{int(x.n_downprox):13,d}'
      f'{(x.pct_downprox if not np.isnan(x.pct_downprox) else float("nan")):7.1f}%')

def boot(sub, kcol, ncol, B=10000):
    d = sub.dropna(subset=[kcol]); d = d[d[ncol]>0]
    k = d[kcol].values.astype(float); n = d[ncol].values.astype(float)
    pooled = 100*k.sum()/n.sum()
    idx = rng.integers(0, len(d), size=(B, len(d)))
    bs = 100*k[idx].sum(axis=1)/n[idx].sum(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pos = int((100*k/n > 50).sum()); tot = len(d)
    bt = binomtest(pos, tot, 0.5).pvalue
    return pooled, lo, hi, pos, tot, bt

for lab, kc, nc in [('上昇×遠位','k_updist','n_updist'), ('低下×近位','k_downprox','n_downprox')]:
    pooled, lo, hi, pos, tot, bt = boot(per, kc, nc)
    w(f'\n  {lab}: プール {pooled:.1f}% [{lo:.1f}, {hi:.1f}] '
      f'({"50% を含む" if lo<=50<=hi else "50% を除外"}), 符号検定 {pos}/{tot} p={bt:.3f}')
    L.append('')
pd.DataFrame([dict(metric=lab, pooled=boot(per,kc,nc)[0], ci_lo=boot(per,kc,nc)[1], ci_hi=boot(per,kc,nc)[2],
                   sign_pos=boot(per,kc,nc)[3], sign_tot=boot(per,kc,nc)[4], sign_p=boot(per,kc,nc)[5])
              for lab,kc,nc in [('上昇×遠位','k_updist','n_updist'),('低下×近位','k_downprox','n_downprox')]]
            ).to_csv(f'{OUT}/tables/05b_bootstrap_ci.csv', index=False)

w('\n' + '='*100)
w('(6) 「暦年齢では 25.7% のみ」のベースライン')
w('='*100)
aa = pd.read_csv(f'{P}/cre_corr_aida_subcluster_significant_{TS}.csv',
                 usecols=['cre_id','cell_type','target_var','spearman_rho','spearman_pval','fdr'])
age_sig = aa[(aa.target_var=='age') & (aa.fdr<0.05)].copy()
acc_sig = aa[(aa.target_var==TV) & (aa.fdr<0.05)].copy()
w(f'AIDA 暦年齢軸 FDR<0.05: {len(age_sig):,} 行 / 異なり tCRE {age_sig.cre_id.nunique():,}')
w(f'AIDA 加速軸  FDR<0.05: {len(acc_sig):,} 行 / 異なり tCRE {acc_sig.cre_id.nunique():,}')
w(f'  検出数の比 = {len(acc_sig)/len(age_sig):.1f} 倍')

ageset = set(zip(age_sig.cre_id, age_sig.cell_type))
robk = set(zip(rob.cre_id, rob.cell_type))
n_rob_in_age = sum(1 for k in robk if k in ageset)
w(f'\n  頑健 {len(robk):,} ペアのうち暦年齢軸でも AIDA 有意: {n_rob_in_age:,} ({100*n_rob_in_age/len(robk):.1f}%)')
accset = set(zip(acc_sig.cre_id, acc_sig.cell_type))
n_acc_in_age = sum(1 for k in accset if k in ageset)
w(f'  ★ベースライン: AIDA 加速軸有意 {len(accset):,} のうち暦年齢軸でも有意 = {n_acc_in_age:,} '
  f'({100*n_acc_in_age/len(accset):.1f}%)')
w(f'  倍率 = {(100*n_rob_in_age/len(robk)) / (100*n_acc_in_age/len(accset)):.1f} 倍')

N = min(len(age_sig), len(acc_sig))
w(f'\n--- 検出数を揃えた比較 (両軸とも p 値上位 {N:,} 行に切り詰め) ---')
age_top = aa[aa.target_var=='age'].nsmallest(N, 'spearman_pval')
acc_top = aa[aa.target_var==TV].nsmallest(N, 'spearman_pval')
A = set(zip(age_top.cre_id, age_top.cell_type)); C = set(zip(acc_top.cre_id, acc_top.cell_type))
w(f'  重なり: {len(A&C):,}  (加速軸側から見て {100*len(A&C)/len(C):.1f}%, 暦年齢側から見て {100*len(A&C)/len(A):.1f}%)')
w(f'  頑健 {len(robk):,} ペアのうち暦年齢 top{N:,} に入る: '
  f'{sum(1 for k in robk if k in A):,} ({100*sum(1 for k in robk if k in A)/len(robk):.1f}%)')

pd.DataFrame([dict(item='AIDA 暦年齢軸 有意行', value=len(age_sig)),
              dict(item='AIDA 加速軸 有意行', value=len(acc_sig)),
              dict(item='頑健ペア数', value=len(robk)),
              dict(item='頑健のうち暦年齢有意', value=n_rob_in_age),
              dict(item='頑健のうち暦年齢有意 %', value=100*n_rob_in_age/len(robk)),
              dict(item='ベースライン: 加速有意のうち暦年齢有意', value=n_acc_in_age),
              dict(item='ベースライン %', value=100*n_acc_in_age/len(accset)),
              dict(item='検出数を揃えた重なり (n=%d)'%N, value=len(A&C)),
              ]).to_csv(f'{OUT}/tables/06_age_axis_baseline.csv', index=False)

open(f'{OUT}/logs/stage1.log','w').write('\n'.join(L))
print('\n-> stage1 done')


import numpy as np, pandas as pd
B='${PROJECT_DIR}/results'
D=f'{B}/PAPER_CRE_MECHANISM'
OUT=f'{D}/results/fixes/age_adjusted'
SYN='${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/subcluster/tage_all_combined_v2.csv'
CD4=f'{D}/results/cd4_redo/cd4_our_tage.csv'
L=[]
def w(s=''):
    print(s,flush=True); L.append(str(s))

def icc1(vals, groups):
    d = pd.DataFrame({'y':vals,'g':groups}).dropna()
    N = len(d); m = d.g.nunique()
    if m < 2 or N <= m: return np.nan, N, m, np.nan
    gm = d.y.mean()
    ni = d.groupby('g').y.size().values
    mi = d.groupby('g').y.mean().values
    SSB = float((ni*(mi-gm)**2).sum()); dfB = m-1
    SSW = float(((d.y - d.g.map(d.groupby('g').y.mean()))**2).sum()); dfW = N-m
    MSB, MSW = SSB/dfB, SSW/dfW
    k0 = (N - (ni**2).sum()/N)/(m-1)
    icc = (MSB-MSW)/(MSB+(k0-1)*MSW)
    return icc, N, m, k0

def resid_on_age(g):
    d = g.dropna(subset=['y','age'])
    X = np.column_stack([np.ones(len(d)), d.age.values.astype(float)])
    b,*_ = np.linalg.lstsq(X, d.y.values.astype(float), rcond=None)
    out = pd.Series(np.nan, index=g.index)
    out.loc[d.index] = d.y.values - X@b
    return out

w('='*100); w('課題 B2 段階3 — ICC を年齢調整前後で'); w('='*100)

s = pd.read_csv(SYN)
s = s[(s.model_type=='Chronoage')&(s['mode']=='EN')].copy().rename(columns={'Age':'age'})
base = s[['Donor_id','Batch','CellType','age','yugene_diff_EN_tAge']].rename(columns={'yugene_diff_EN_tAge':'y'})
w(f'元 tAge 表 (全訪問): {len(base)} 行 / {base.Donor_id.nunique()} ドナー')
vis = base[['Donor_id','Batch']].drop_duplicates()
w(f'  訪問キー: {len(vis)}  ドナーあたり訪問数 {vis.groupby("Donor_id").size().value_counts().sort_index().to_dict()}')

our = pd.read_csv(CD4)
ours = our[our.cohort=='Synapse'][['Donor_id','Batch','Group','age','yugene_diff_EN_tAge']].rename(
    columns={'Group':'CellType','yugene_diff_EN_tAge':'y'})
w(f'差し替え用 CD4 (Synapse): {len(ours)} 行 / {ours.Donor_id.nunique()} ドナー')
swapped = pd.concat([base[~base.CellType.isin(['CD4_Naive','CD4_Memory'])], ours], ignore_index=True)

nvis = base.groupby('Donor_id').Batch.nunique()
multi = set(nvis[nvis > 1].index)
w(f'複数時点ドナー: {len(multi)} 名 (単一時点 {int((nvis==1).sum())} 名)')

for g_, gg in ours.groupby('CellType'):
    nu = gg.groupby('Donor_id').y.nunique()
    w(f'  差し替え CD4 {g_}: {len(gg)} 行 / {gg.Donor_id.nunique()} ドナー / '
      f'異なり値 {gg.y.nunique()} / ドナー内で値が1種類 {int((nu==1).sum())}/{len(nu)}')

rows=[]
for tag, tbl in [('差し替え前 (元 tAge 表)', base), ('★差し替え後 (横断で実際に使った変数)', swapped)]:
    for scope, sel in [('複数時点ドナーのみ', lambda d: d[d.Donor_id.isin(multi)]),
                       ('全ドナー', lambda d: d)]:
        for ct, g in sel(tbl).groupby('CellType'):
            g = g.copy()
            i_raw, N, m, k0 = icc1(g.y.values, g.Donor_id.values)
            g['y_res'] = resid_on_age(g)
            i_adj, *_ = icc1(g['y_res'].values, g.Donor_id.values)
            within = g.groupby('Donor_id').y.nunique()
            rows.append(dict(table=tag, scope=scope, cell_type=ct, n_obs=N, n_donor=m, k0=k0,
                             ICC_raw=i_raw, ICC_age_adj=i_adj, delta=i_adj-i_raw,
                             donors_with_constant_value=int((within==1).sum())))
icc = pd.DataFrame(rows)
icc.to_csv(f'{OUT}/tables/03_icc.csv', index=False)

for tag, scope in [(t_,s_) for t_ in icc.table.unique() for s_ in ['複数時点ドナーのみ','全ドナー']]:
    t = icc[(icc.table==tag)&(icc.scope==scope)].sort_values('ICC_raw', ascending=False).reset_index(drop=True)
    tag = f'{tag} / {scope}'
    t['rank_raw'] = t.ICC_raw.rank(ascending=False).astype(int)
    t['rank_adj'] = t.ICC_age_adj.rank(ascending=False).astype(int)
    w(f'\n--- {tag} ---')
    w(f'{"細胞種":14s}{"観測":>6s}{"ドナー":>7s}{"ICC 生":>9s}{"ICC 年齢調整":>13s}{"差":>8s}{"順位 生->調整":>14s}')
    for _, x in t.iterrows():
        mv = '' if x.rank_raw==x.rank_adj else '  ★入替'
        w(f'{x.cell_type:14s}{int(x.n_obs):6d}{int(x.n_donor):7d}{x.ICC_raw:9.3f}{x.ICC_age_adj:13.3f}'
          f'{x.delta:+8.3f}{int(x.rank_raw):7d}->{int(x.rank_adj):<5d}{mv}')

w('\n--- 監査報告値との照合 (差し替え前) ---')
chk = icc[(icc.table=='差し替え前 (元 tAge 表)')&(icc.scope=='複数時点ドナーのみ')].set_index('cell_type')
for ct, ref_raw, ref_adj in [('CD4_Memory',0.783,0.528),('CD4_Naive',0.681,0.397),('CD14_Mono',0.579,0.494)]:
    r = chk.loc[ct]
    w(f'  {ct:12s} 生 {r.ICC_raw:.4f} (監査 {ref_raw})  年齢調整 {r.ICC_age_adj:.4f} (監査 {ref_adj})')

w('\n--- 差し替えで ICC がどう動くか (CD4 の 2 群) ---')
a = icc[(icc.table=='差し替え前 (元 tAge 表)')&(icc.scope=='複数時点ドナーのみ')].set_index('cell_type')
b = icc[(icc.table=='★差し替え後 (横断で実際に使った変数)')&(icc.scope=='複数時点ドナーのみ')].set_index('cell_type')
for ct in ['CD4_Memory','CD4_Naive']:
    w(f'  {ct:12s} 生 {a.loc[ct].ICC_raw:.3f} -> {b.loc[ct].ICC_raw:.3f}   '
      f'年齢調整 {a.loc[ct].ICC_age_adj:.3f} -> {b.loc[ct].ICC_age_adj:.3f}')

open(f'{OUT}/logs/stage3.log','w').write('\n'.join(L))
print('\n-> stage3 done')


import numpy as np, pandas as pd
from scipy.stats import fisher_exact, binomtest, t as tdist
B='${PROJECT_DIR}/results'
P=f'{B}/phase3b_ourcd4'; D=f'{B}/PAPER_CRE_MECHANISM'
OUT=f'{D}/results/fixes/age_adjusted'
TS='_v4'; TV='yugene_diff_EN_tAge'
NE=f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv'
L=[]; rng=np.random.default_rng(_v5)
def w(s=''):
    print(s,flush=True); L.append(str(s))

ne=pd.read_csv(NE,usecols=['cre_id','proximity']).dropna()
ne=ne[ne.proximity.isin(['proximal','distal'])].drop_duplicates('cre_id')
prox=dict(zip(ne.cre_id,ne.proximity))

A=pd.read_parquet(f'{OUT}/tables/recorr_aida_subcluster.parquet')
S=pd.read_parquet(f'{OUT}/tables/recorr_synapse_subcluster.parquet')
A['prox']=A.cre_id.map(prox); S=S[['cre_id','cell_type','rho_plain','fdr_plain','rho_partial','fdr_partial','n_samples']]
S.columns=['cre_id','cell_type','s_rho_plain','s_fdr_plain','s_rho_partial','s_fdr_partial','s_n']
M=A.merge(S,on=['cre_id','cell_type'],how='left')
M=M.dropna(subset=['prox'])
w('='*100); w('課題 B2 段階4 — 年齢調整後に柱が残るか'); w('='*100)
w(f'突合表: {len(M):,} 行 (AIDA 全 tCRE × 全 subcluster, 近位/遠位注釈あり)')

def rates(df, acol, prox_col='prox'):
    du=int(((df[prox_col]=='distal')&(df[acol]>0)).sum()); dd=int(((df[prox_col]=='distal')&(df[acol]<0)).sum())
    pu=int(((df[prox_col]=='proximal')&(df[acol]>0)).sum()); pdn=int(((df[prox_col]=='proximal')&(df[acol]<0)).sum())
    orv,p=fisher_exact([[du,dd],[pu,pdn]])
    return dict(n_distal=du+dd,pct_distal_up=100*du/max(du+dd,1),n_prox=pu+pdn,
                pct_prox_up=100*pu/max(pu+pdn,1),OR=orv,fisher_p=p)

w('\n'+'='*100)
w('A. ★決定的検定 — 素子集合を正典の 4,598 ペアに固定したまま偏相関に置き換える')
w('   (集合を固定するので検出力の交絡がない。唯一の検出力フリーな検定)')
w('='*100)
rob = M[(M.fdr_plain<0.05)&(M.s_fdr_plain<0.05)&(np.sign(M.rho_plain)==np.sign(M.s_rho_plain))].copy()
w(f'正典 頑健セット: {len(rob):,} ペア (遠位 {int((rob.prox=="distal").sum()):,} / 近位 {int((rob.prox=="proximal").sum()):,})')

rob['sign_keep_aida']=np.sign(rob.rho_plain)==np.sign(rob.rho_partial)
rob['sign_keep_syn'] =np.sign(rob.s_rho_plain)==np.sign(rob.s_rho_partial)
rob['still_same_sign']=np.sign(rob.rho_partial)==np.sign(rob.s_rho_partial)
w(f'\n  AIDA 側 符号保存: {100*rob.sign_keep_aida.mean():.1f}% ({int(rob.sign_keep_aida.sum()):,}/{len(rob):,})')
w(f'  Synapse 側 符号保存: {100*rob.sign_keep_syn.mean():.1f}% ({int(rob.sign_keep_syn.sum()):,}/{len(rob):,})')
w(f'  偏相関でも両コホート同符号: {100*rob.still_same_sign.mean():.1f}%')
w(f'\n  |rho| 中央値  AIDA 素 {rob.rho_plain.abs().median():.3f} -> 偏 {rob.rho_partial.abs().median():.3f} '
  f'({100*(rob.rho_partial.abs().median()/rob.rho_plain.abs().median()-1):+.1f}%)')
w(f'  |rho| 中央値  Syn  素 {rob.s_rho_plain.abs().median():.3f} -> 偏 {rob.s_rho_partial.abs().median():.3f} '
  f'({100*(rob.s_rho_partial.abs().median()/rob.s_rho_plain.abs().median()-1):+.1f}%)')

w('\n  --- 分割: 遠位×上昇 / 近位×低下 ---')
sub=[]
for pxy,dirn,lab in [('distal',1,'遠位×上昇'),('proximal',-1,'近位×低下'),
                     ('distal',-1,'遠位×低下'),('proximal',1,'近位×上昇')]:
    g=rob[(rob.prox==pxy)&(np.sign(rob.rho_plain)==dirn)]
    if not len(g): continue
    w(f'  {lab}: n={len(g):,}  AIDA 符号保存 {100*g.sign_keep_aida.mean():.1f}%  '
      f'Syn 符号保存 {100*g.sign_keep_syn.mean():.1f}%  両コホート同符号 {100*g.still_same_sign.mean():.1f}%')
    sub.append(dict(group=lab,n=len(g),aida_sign_keep=100*g.sign_keep_aida.mean(),
                    syn_sign_keep=100*g.sign_keep_syn.mean(),both_same_sign=100*g.still_same_sign.mean(),
                    med_abs_rho_plain=g.rho_plain.abs().median(),med_abs_rho_partial=g.rho_partial.abs().median()))
pd.DataFrame(sub).to_csv(f'{OUT}/tables/07a_robust4598_partial_signkeep.csv',index=False)

w('\n  --- 集合固定のまま 2x2 を引き直す ---')
r0=rates(rob,'rho_plain'); r1=rates(rob,'rho_partial')
w(f'  {"":10s}{"遠位 n":>9s}{"遠位上昇率":>10s}{"近位 n":>9s}{"近位上昇率":>10s}{"OR":>9s}')
w(f'  {"素相関":10s}{r0["n_distal"]:9,d}{r0["pct_distal_up"]:9.1f}%{r0["n_prox"]:9,d}{r0["pct_prox_up"]:9.1f}%{r0["OR"]:9.2f}')
w(f'  {"★偏相関":10s}{r1["n_distal"]:9,d}{r1["pct_distal_up"]:9.1f}%{r1["n_prox"]:9,d}{r1["pct_prox_up"]:9.1f}%{r1["OR"]:9.2f}')
pd.DataFrame([dict(axis='素相関',**r0),dict(axis='偏相関(年齢調整)',**r1)]).to_csv(
    f'{OUT}/tables/07b_robust4598_fixed_2x2.csv',index=False)

w('\n  --- 集合固定のまま FDR も引き直す (検出力に敏感な量) ---')
keep=rob[(rob.fdr_partial<0.05)&(rob.s_fdr_partial<0.05)&(np.sign(rob.rho_partial)==np.sign(rob.s_rho_partial))]
w(f'  偏相関でも両コホート FDR<0.05 かつ同符号: {len(keep):,} / {len(rob):,} = {100*len(keep)/len(rob):.1f}%')
rk=rates(keep,'rho_partial')
w(f'  そのうち: 遠位 {rk["n_distal"]:,} / {rk["pct_distal_up"]:.1f}% 上昇, '
  f'近位 {rk["n_prox"]:,} / {rk["pct_prox_up"]:.1f}% 上昇, OR {rk["OR"]:.2f}')

w('\n'+'='*100)
w('B. 年齢調整版で AIDA 選抜も Synapse 検証もやり直す')
w('='*100)
def pipeline(acol_r,acol_f,scol_r,scol_f,tag,sel_mask=None):
    a = M[sel_mask] if sel_mask is not None else M[M[acol_f]<0.05]
    a = a.copy(); a['dir_a']=np.where(a[acol_r]>0,'up','down')
    m = a.dropna(subset=[scol_r]).copy(); m['dir_s']=np.where(m[scol_r]>0,'up','down')
    stages={}
    stages['AIDA 選抜のみ']=rates(a,acol_r)
    same=m[m.dir_a==m.dir_s]; stages['符号一致のみ']=rates(same,acol_r)
    rb=m[(m[scol_f]<0.05)&(m.dir_a==m.dir_s)]; stages['両コホート頑健']=rates(rb,acol_r)
    st=pd.DataFrame(stages).T.reset_index().rename(columns={'index':'選抜段階'}); st['axis']=tag
    per=[]
    for ct,g in m.groupby('cell_type'):
        ud=g[(g.dir_a=='up')&(g.prox=='distal')]; dp=g[(g.dir_a=='down')&(g.prox=='proximal')]
        per.append(dict(axis=tag,cell_type=ct,n_updist=len(ud),k_updist=int((ud.dir_s=='up').sum()),
                        pct_updist=100*(ud.dir_s=='up').mean() if len(ud) else np.nan,
                        n_downprox=len(dp),k_downprox=int((dp.dir_s=='down').sum()),
                        pct_downprox=100*(dp.dir_s=='down').mean() if len(dp) else np.nan))
    return st,pd.DataFrame(per),len(a),len(rb),rb

res={}
res['素 (正典)']=pipeline('rho_plain','fdr_plain','s_rho_plain','s_fdr_plain','素 (正典)')
res['★偏相関 (年齢調整)']=pipeline('rho_partial','fdr_partial','s_rho_partial','s_fdr_partial','★偏相関 (年齢調整)')

n_match=res['素 (正典)'][2]
thr=M.nsmallest(n_match,'p_partial').p_partial.max()
res['偏相関 matched-n']=pipeline('rho_partial','fdr_partial','s_rho_partial','s_fdr_partial',
                                 '偏相関 matched-n',sel_mask=(M.p_partial<=thr))

allst=pd.concat([v[0] for v in res.values()],ignore_index=True)
allst.to_csv(f'{OUT}/tables/07c_or_by_stage_ageadj.csv',index=False)
for tag,(st,per,na,nr,rb) in res.items():
    w(f'\n--- {tag} (AIDA 選抜 {na:,} 行 / 頑健 {nr:,} ペア, 異なり tCRE {rb.cre_id.nunique():,}) ---')
    w(f'{"選抜段階":16s}{"遠位 n":>10s}{"遠位上昇率":>10s}{"近位 n":>10s}{"近位上昇率":>10s}{"OR":>9s}')
    for _,x in st.iterrows():
        w(f'{x["選抜段階"]:16s}{int(x.n_distal):10,d}{x.pct_distal_up:9.1f}%{int(x.n_prox):10,d}'
          f'{x.pct_prox_up:9.1f}%{x.OR:9.2f}')

allper=pd.concat([v[1] for v in res.values()],ignore_index=True)
allper.to_csv(f'{OUT}/tables/07d_percelltype_ageadj.csv',index=False)
w('\n--- 細胞種別 held-out 方向一致率 (上昇×遠位 / 低下×近位) ---')
w(f'{"細胞種":14s}' + ''.join(f'{t:>26s}' for t in res))
for ct in sorted(allper.cell_type.unique()):
    line=f'{ct:14s}'
    for t in res:
        r=allper[(allper.axis==t)&(allper.cell_type==ct)]
        if len(r)==0: line+=f'{"-":>26s}'; continue
        r=r.iloc[0]
        u=f'{r.pct_updist:.1f}' if not np.isnan(r.pct_updist) else 'na'
        d=f'{r.pct_downprox:.1f}' if not np.isnan(r.pct_downprox) else 'na'
        line+=f'{u+"% / "+d+"%":>26s}'
    w(line)
w('\n  (n は 07d_percelltype_ageadj.csv 参照)')

w('\n--- ★CD4_Memory の 37.2% と 76.3% ---')
for t in res:
    r=allper[(allper.axis==t)&(allper.cell_type=='CD4_Memory')]
    if len(r)==0: w(f'  {t}: 該当なし'); continue
    r=r.iloc[0]
    w(f'  {t:22s} 上昇×遠位 {r.pct_updist:5.1f}% (n={int(r.n_updist):,})   '
      f'低下×近位 {r.pct_downprox:5.1f}% (n={int(r.n_downprox):,})')

def boot(sub,B=10000,minn=10):
    d=sub.dropna(subset=['k']); d=d[d.n>=minn]
    if len(d)<2: return (np.nan,)*6
    k=d.k.values.astype(float); n=d.n.values.astype(float)
    pooled=100*k.sum()/n.sum()
    idx=rng.integers(0,len(d),size=(B,len(d)))
    bs=100*k[idx].sum(axis=1)/n[idx].sum(axis=1)
    lo,hi=np.percentile(bs,[2.5,97.5])
    pos=int((100*k/n>50).sum()); tot=len(d)
    return pooled,lo,hi,pos,tot,binomtest(pos,tot,0.5).pvalue

w('\n--- プール率 + 細胞種ブートストラップ CI (n>=10 の細胞種のみ) ---')
brows=[]
for t in res:
    p=allper[allper.axis==t]
    for lab,kc,nc in [('上昇×遠位','k_updist','n_updist'),('低下×近位','k_downprox','n_downprox')]:
        sub=p[[ 'cell_type',kc,nc]].rename(columns={kc:'k',nc:'n'})
        pooled,lo,hi,pos,tot,bp=boot(sub)
        w(f'  {t:22s} {lab}: {pooled:.1f}% [{lo:.1f}, {hi:.1f}] '
          f'{"(50% を含む)" if lo<=50<=hi else "(50% を除外)"}  符号検定 {pos}/{tot} p={bp:.3f}')
        brows.append(dict(axis=t,metric=lab,pooled=pooled,ci_lo=lo,ci_hi=hi,sign_pos=pos,sign_tot=tot,sign_p=bp))
pd.DataFrame(brows).to_csv(f'{OUT}/tables/07e_bootstrap_ageadj.csv',index=False)

w('\n--- 年齢調整前後で頑健セットはどれだけ重なるか ---')
k0=set(zip(res['素 (正典)'][4].cre_id,res['素 (正典)'][4].cell_type))
for t in ['★偏相関 (年齢調整)','偏相関 matched-n']:
    k1=set(zip(res[t][4].cre_id,res[t][4].cell_type))
    w(f'  {t:22s}: {len(k1):,} ペア, 正典 4,598 との重なり {len(k0&k1):,} '
      f'(正典側 {100*len(k0&k1)/len(k0):.1f}%)')

open(f'{OUT}/logs/stage4.log','w').write('\n'.join(L))
print('\n-> stage4 done')
