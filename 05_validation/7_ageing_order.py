#!/usr/bin/env python3
import pickle, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
OUT = f'{D}/results/ageing_order'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

rob = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv',
                  usecols=['cre_id', 'cell_type', 'program'])
rob = rob[rob.program.isin(['up_distal', 'down_proximal'])]
PROG = rob.drop_duplicates('cre_id').set_index('cre_id').program.to_dict()
w(f'頑健 tCRE {len(PROG):,}')
jo = pd.read_csv(f'{D}/results/gene_to_cre/cre_gene_joined.csv',
                 usecols=['cre_id', 'near_gene']).drop_duplicates('cre_id')
GMAP = dict(zip(jo.cre_id, jo.near_gene))

ta = pd.read_csv('${DATA_DIR}/lnc_project_AIDA_dataset/by_subcluster/tage/'
                 'tage_all_combined_v1.csv')
ta = ta[(ta.model_type == 'Chronoage') & (ta['mode'] == 'EN')]
AGE = ta.drop_duplicates('Sample').set_index('Sample').age.to_dict()

w('AIDA キャッシュを開く（tmux + nice、空きメモリ確認済み）')
d = pickle.load(open(f'{OUR}/aggregation_cache_aida_subcluster.pkl', 'rb'))
cres = np.array(d['cres']); keys = list(d['expr_sum'].keys())
w(f'tCRE {len(cres):,} × メタセル {len(keys):,}')
idx = {c: i for i, c in enumerate(cres)}
rob_idx = np.array([idx[c] for c in PROG if c in idx])
rob_ids = np.array([c for c in PROG if c in idx])
w(f'キャッシュに存在する頑健 tCRE {len(rob_idx):,}')

meta = pd.DataFrame(keys, columns=['cohort_k', 'Donor_id', 'Batch', 'CellType'])
meta['n_cells'] = [d['n_cells'][k] for k in keys]
meta['age'] = meta.Donor_id.map(AGE)
keep = (meta.n_cells >= 10) & meta.age.notna()
w(f'採用メタセル {int(keep.sum()):,}（細胞 10 個以上、年齢あり）')
kk = [k for k, ok in zip(keys, keep) if ok]
meta = meta[keep].reset_index(drop=True)
w(f'年齢 {meta.age.min():.0f}〜{meta.age.max():.0f} 歳 / 細胞種 {meta.CellType.nunique()}')

sub = kk[::max(1, len(kk) // 300)]
det = np.zeros(len(cres))
for k in sub:
    det += (d['expr_sum'][k] > 0)
det /= len(sub)
ref = np.nonzero(det >= 0.8)[0]
if len(ref) > 20000:
    ref = rng.choice(ref, 20000, replace=False)
w(f'CLR の基準に使う tCRE {len(ref):,}（検出率 80% 以上から抽出）')

M = np.zeros((len(kk), len(rob_idx)), np.float32)
G = np.zeros(len(kk), np.float64)
for i, k in enumerate(kk):
    v = d['expr_sum'][k]
    G[i] = np.log(v[ref] + 0.5).mean()
    M[i] = np.log(v[rob_idx] + 0.5) - G[i]
del d
w('CLR 行列を構築（大域スケール除去済み）')

pg = np.array([PROG[c] for c in rob_ids])
S = meta.copy()
S['up_distal'] = M[:, pg == 'up_distal'].mean(1)
S['down_proximal'] = M[:, pg == 'down_proximal'].mean(1)
S['balance'] = S.up_distal - S.down_proximal
S.to_csv(f'{OUT}/metacell_program_scores.csv', index=False)

w('\n[1] 生涯のどこで動くか — 年齢との関係（CLR 得点、細胞種ごと）')
w(f'  {"細胞種":13s} {"n":>5s} {"上昇×年齢":>10s} {"低下×年齢":>10s} {"差分×年齢":>10s}')
rows = []
for ct, g in S.groupby('CellType'):
    if len(g) < 80: continue
    ru = spearmanr(g.up_distal, g.age); rd = spearmanr(g.down_proximal, g.age)
    rb = spearmanr(g.balance, g.age)
    rows.append(dict(cell_type=ct, n=len(g), rho_up=ru.statistic, p_up=ru.pvalue,
                     rho_dn=rd.statistic, p_dn=rd.pvalue, rho_bal=rb.statistic, p_bal=rb.pvalue))
    w(f'  {ct:13s} {len(g):5,} {ru.statistic:+10.3f} {rd.statistic:+10.3f} {rb.statistic:+10.3f}')
R = pd.DataFrame(rows).sort_values('rho_bal', ascending=False)
R.to_csv(f'{OUT}/celltype_age_trajectory.csv', index=False)
w('  ★年齢に沿って最も動く細胞種: ' +
  ', '.join(f'{r.cell_type} {r.rho_bal:+.2f}' for _, r in R.head(3).iterrows()))

w('\n[2] 離陸年齢 — 何歳台から最若年帯と差がつくか')
S['bin'] = (S.age // 5 * 5).astype(int)
dep = []
for ct, g in S.groupby('CellType'):
    if len(g) < 80: continue
    bins = sorted(g.bin.unique())
    base = g[g.bin <= bins[1]]
    if len(base) < 15: continue
    got = False
    for b in bins[2:]:
        cur = g[g.bin == b]
        if len(cur) < 15: continue
        u = mannwhitneyu(cur.balance, base.balance)
        if u.pvalue < 0.05 and cur.balance.median() > base.balance.median():
            dep.append(dict(cell_type=ct, departure_age=b, n_base=len(base), n_bin=len(cur),
                            p=u.pvalue, delta=cur.balance.median() - base.balance.median()))
            got = True; break
    if not got:
        dep.append(dict(cell_type=ct, departure_age=np.nan, n_base=len(base), n_bin=np.nan,
                        p=np.nan, delta=np.nan))
DP = pd.DataFrame(dep).sort_values('departure_age')
DP.to_csv(f'{OUT}/departure_age.csv', index=False)
for _, r in DP.iterrows():
    if np.isnan(r.departure_age):
        w(f'  {r.cell_type:13s} 生涯を通じて基準から外れない')
    else:
        w(f'  {r.cell_type:13s} {int(r.departure_age)} 歳台から離陸（p={r.p:.2g}, Δ={r.delta:+.3f}）')

w('\n[3] 先導する tCRE — 若年層だけで既に年齢と動いているもの')
med = S.age.median()
w(f'  若年 = {med:.0f} 歳未満、高齢 = {med:.0f} 歳以上')
ct_of = rob.drop_duplicates('cre_id').set_index('cre_id').cell_type.to_dict()
lead = []
for ct, g in S.groupby('CellType'):
    if len(g) < 80: continue
    sel = np.array([ct_of.get(c) == ct for c in rob_ids])
    if sel.sum() < 30: continue
    yi = (g.age < med).values; oi = ~yi
    if yi.sum() < 40 or oi.sum() < 40: continue
    sub_m = M[g.index.values][:, sel]
    ay = g.age.values[yi]; ao = g.age.values[oi]
    for j, c in enumerate(rob_ids[sel]):
        ry = spearmanr(sub_m[yi, j], ay); ro = spearmanr(sub_m[oi, j], ao)
        lead.append(dict(cre_id=c, cell_type=ct, program=PROG[c], near_gene=GMAP.get(c),
                         rho_young=ry.statistic, p_young=ry.pvalue,
                         rho_old=ro.statistic, p_old=ro.pvalue))
LD = pd.DataFrame(lead)
LD.to_csv(f'{OUT}/leading_tcres.csv', index=False)
w(f'  検定 {len(LD):,} 件（tCRE × 細胞種）')
for prg, sign in [('up_distal', 1), ('down_proximal', -1)]:
    sub2 = LD[LD.program == prg]
    early = sub2[(sub2.p_young < 0.01) & (np.sign(sub2.rho_young) == sign)]
    cont = early[(np.sign(early.rho_old) == sign) & (early.p_old < 0.05)]
    w(f'\n  【{prg}】若年層で既に動く {len(early)} / {len(sub2)}、うち高齢でも継続 {len(cont)}')
    top = cont.reindex(cont.rho_young.abs().sort_values(ascending=False).index).head(20)
    for _, r in top.iterrows():
        w(f'    {str(r.near_gene):14s} {r.cell_type:13s} 若年 rho {r.rho_young:+.3f} / '
          f'高齢 rho {r.rho_old:+.3f}')
w('\n  ★若年層で既に動き高齢でも同じ向きに続く tCRE = 生涯を通じて進む変化 = 予防の標的候補。')
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')


import pickle, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
OUT = f'{D}/results/ageing_order/synapse'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
SPLIT = 50
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

rob = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv',
                  usecols=['cre_id', 'cell_type', 'program'])
rob = rob[rob.program.isin(['up_distal', 'down_proximal'])]
PROG = rob.drop_duplicates('cre_id').set_index('cre_id').program.to_dict()
CT_OF = rob.drop_duplicates('cre_id').set_index('cre_id').cell_type.to_dict()
jo = pd.read_csv(f'{D}/results/gene_to_cre/cre_gene_joined.csv',
                 usecols=['cre_id', 'near_gene']).drop_duplicates('cre_id')
GMAP = dict(zip(jo.cre_id, jo.near_gene))

ts = pd.read_csv('${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/'
                 'subcluster/tage_all_combined_v2.csv')
ts = ts[(ts.model_type == 'Chronoage') & (ts['mode'] == 'EN')]
vis = ts[['Donor_id', 'Batch', 'Age']].drop_duplicates().sort_values(['Donor_id', 'Age'])
vis['visit'] = vis.groupby('Donor_id').cumcount() + 1
AGEKEY = {(r.Donor_id, r.Batch): r.Age for _, r in vis.iterrows()}
VISKEY = {(r.Donor_id, r.Batch): r.visit for _, r in vis.iterrows()}

w('Synapse キャッシュを開く')
d = pickle.load(open(f'{OUR}/aggregation_cache_synapse_subcluster.pkl', 'rb'))
cres = np.array(d['cres']); keys = list(d['expr_sum'].keys())
idx = {c: i for i, c in enumerate(cres)}
rob_ids = np.array([c for c in PROG if c in idx])
rob_idx = np.array([idx[c] for c in rob_ids])
w(f'tCRE {len(cres):,} × メタセル {len(keys):,} / 頑健 tCRE {len(rob_ids):,}')

meta = pd.DataFrame(keys, columns=['cohort_k', 'Donor_id', 'Batch', 'CellType'])
meta['n_cells'] = [d['n_cells'][k] for k in keys]
meta['age'] = [AGEKEY.get((a, b), np.nan) for a, b in zip(meta.Donor_id, meta.Batch)]
meta['visit'] = [VISKEY.get((a, b), np.nan) for a, b in zip(meta.Donor_id, meta.Batch)]
keep = (meta.n_cells >= 10) & meta.age.notna()
kk = [k for k, ok in zip(keys, keep) if ok]
meta = meta[keep].reset_index(drop=True)
w(f'採用メタセル {len(meta):,} / 年齢 {meta.age.min():.0f}〜{meta.age.max():.0f} 歳 '
  f'（50 歳以上 {int((meta.age>=SPLIT).sum()):,}）')

sub = kk[::max(1, len(kk) // 300)]
det = np.zeros(len(cres))
for k in sub:
    det += (d['expr_sum'][k] > 0)
det /= len(sub)
ref = np.nonzero(det >= 0.8)[0]
if len(ref) > 20000:
    ref = rng.choice(ref, 20000, replace=False)
w(f'CLR 基準 tCRE {len(ref):,}')
M = np.zeros((len(kk), len(rob_idx)), np.float32)
for i, k in enumerate(kk):
    v = d['expr_sum'][k]
    M[i] = np.log(v[rob_idx] + 0.5) - np.log(v[ref] + 0.5).mean()
del d

pg = np.array([PROG[c] for c in rob_ids])
S = meta.copy()
S['up_distal'] = M[:, pg == 'up_distal'].mean(1)
S['down_proximal'] = M[:, pg == 'down_proximal'].mean(1)
S['balance'] = S.up_distal - S.down_proximal
S.to_csv(f'{OUT}/metacell_program_scores.csv', index=False)
base = S[S.visit == 1]
w(f'初回訪問のみ {len(base):,} メタセル（主解析）')

w('\n[Q1] AIDA の細胞種順位は Synapse で再現するか（初回訪問のみ、CLR 得点 × 暦年齢）')
rows = []
for ct, g in base.groupby('CellType'):
    if len(g) < 60: continue
    ru = spearmanr(g.up_distal, g.age); rd = spearmanr(g.down_proximal, g.age)
    rb = spearmanr(g.balance, g.age)
    rows.append(dict(cell_type=ct, n=len(g), rho_up=ru.statistic, rho_dn=rd.statistic,
                     rho_bal=rb.statistic, p_bal=rb.pvalue))
Rs = pd.DataFrame(rows).sort_values('rho_bal', ascending=False)
Ra = pd.read_csv(f'{D}/results/ageing_order/celltype_age_trajectory.csv')
mg = Rs.merge(Ra[['cell_type', 'rho_bal', 'rho_up', 'rho_dn']], on='cell_type',
              suffixes=('_syn', '_aida'))
w(f'  {"細胞種":13s} {"AIDA":>8s} {"Synapse":>9s}   上昇 A/S        低下 A/S')
for _, r in mg.sort_values('rho_bal_aida', ascending=False).iterrows():
    w(f'  {r.cell_type:13s} {r.rho_bal_aida:+8.3f} {r.rho_bal_syn:+9.3f}   '
      f'{r.rho_up_aida:+.2f}/{r.rho_up_syn:+.2f}   {r.rho_dn_aida:+.2f}/{r.rho_dn_syn:+.2f}')
rr = spearmanr(mg.rho_bal_aida, mg.rho_bal_syn)
w(f'  ★細胞種順位の一致: Spearman {rr.statistic:+.3f} (p={rr.pvalue:.3g}, n={len(mg)})')
w(f'  ★符号一致: 差分 {int((np.sign(mg.rho_bal_aida)==np.sign(mg.rho_bal_syn)).sum())}/{len(mg)}、'
  f'低下 {int((np.sign(mg.rho_dn_aida)==np.sign(mg.rho_dn_syn)).sum())}/{len(mg)}、'
  f'上昇 {int((np.sign(mg.rho_up_aida)==np.sign(mg.rho_up_syn)).sum())}/{len(mg)}')
mg.to_csv(f'{OUT}/celltype_replication.csv', index=False)

w('\n[Q1b] AIDA の先導 tCRE は Synapse でも同じ向きか')
LDa = pd.read_csv(f'{D}/results/ageing_order/leading_tcres.csv')
lead = LDa[(LDa.p_young < 0.01) &
           (np.sign(LDa.rho_young) == np.where(LDa.program == 'up_distal', 1, -1))]
res = []
for ct, g in base.groupby('CellType'):
    if len(g) < 60: continue
    sel = np.array([CT_OF.get(c) == ct for c in rob_ids])
    if sel.sum() < 20: continue
    sm = M[g.index.values][:, sel]
    ids = rob_ids[sel]
    for j, c in enumerate(ids):
        r = spearmanr(sm[:, j], g.age.values)
        res.append(dict(cre_id=c, cell_type=ct, rho_syn=r.statistic, p_syn=r.pvalue))
Rt = pd.DataFrame(res)
mm = lead.merge(Rt, on=['cre_id', 'cell_type'], how='inner')
mm.to_csv(f'{OUT}/leading_tcre_replication.csv', index=False)
if len(mm):
    conc = int((np.sign(mm.rho_young) == np.sign(mm.rho_syn)).sum())
    w(f'  照合できた先導 tCRE {len(mm)} / 符号一致 {conc} ({conc/len(mm)*100:.0f}%)')
    for prg in ['up_distal', 'down_proximal']:
        s = mm[mm.program == prg]
        if len(s) < 5: continue
        c2 = int((np.sign(s.rho_young) == np.sign(s.rho_syn)).sum())
        w(f'    {prg}: {c2}/{len(s)} ({c2/len(s)*100:.0f}%) 中央 rho_syn {s.rho_syn.median():+.3f}')
    top = mm.reindex(mm.rho_syn.abs().sort_values(ascending=False).index).head(15)
    w('  両コホートで強い先導 tCRE:')
    for _, r in top.iterrows():
        ok = '一致' if np.sign(r.rho_young) == np.sign(r.rho_syn) else '不一致'
        w(f'    {str(r.near_gene):14s} {r.cell_type:12s} {r.program:14s} '
          f'AIDA若年 {r.rho_young:+.3f} / Synapse {r.rho_syn:+.3f}  ({ok})')

w(f'\n[Q2] 中年以降（{SPLIT} 歳）で別の過程が始まるか')
w(f'  Synapse 初回訪問: {SPLIT} 歳未満 {int((base.age<SPLIT).sum()):,} / 以上 {int((base.age>=SPLIT).sum()):,} メタセル')
rows2 = []
for ct, g in base.groupby('CellType'):
    y = g[g.age < SPLIT]; o = g[g.age >= SPLIT]
    if len(y) < 40 or len(o) < 40: continue
    ry = spearmanr(y.balance, y.age); ro = spearmanr(o.balance, o.age)
    rows2.append(dict(cell_type=ct, n_young=len(y), n_old=len(o),
                      rho_young=ry.statistic, p_young=ry.pvalue,
                      rho_old=ro.statistic, p_old=ro.pvalue))
    w(f'  {ct:13s} 前半 rho {ry.statistic:+.3f} (p={ry.pvalue:.2g}) → '
      f'後半 rho {ro.statistic:+.3f} (p={ro.pvalue:.2g})')
pd.DataFrame(rows2).to_csv(f'{OUT}/piecewise_celltype.csv', index=False)

w(f'\n[Q2b] tCRE 単位: 生涯型（前半から動く）と 後発型（{SPLIT} 歳以降だけ動く）')
pw = []
for ct, g in base.groupby('CellType'):
    y = g[g.age < SPLIT]; o = g[g.age >= SPLIT]
    if len(y) < 40 or len(o) < 40: continue
    sel = np.array([CT_OF.get(c) == ct for c in rob_ids])
    if sel.sum() < 20: continue
    my = M[y.index.values][:, sel]; mo = M[o.index.values][:, sel]
    ids = rob_ids[sel]
    for j, c in enumerate(ids):
        ry = spearmanr(my[:, j], y.age.values); ro = spearmanr(mo[:, j], o.age.values)
        pw.append(dict(cre_id=c, cell_type=ct, program=PROG[c], near_gene=GMAP.get(c),
                       rho_young=ry.statistic, p_young=ry.pvalue,
                       rho_old=ro.statistic, p_old=ro.pvalue))
PW = pd.DataFrame(pw)
PW.to_csv(f'{OUT}/piecewise_tcre.csv', index=False)
sign = np.where(PW.program == 'up_distal', 1, -1)
PW['lifelong'] = (PW.p_young < 0.05) & (np.sign(PW.rho_young) == sign) & \
                 (PW.p_old < 0.05) & (np.sign(PW.rho_old) == sign)
PW['late_only'] = (PW.p_young > 0.3) & (PW.p_old < 0.05) & (np.sign(PW.rho_old) == sign)
PW['early_only'] = (PW.p_young < 0.05) & (np.sign(PW.rho_young) == sign) & (PW.p_old > 0.3)
w(f'  検定 {len(PW):,} 件（tCRE × 細胞種）')
w(f'  生涯型 {int(PW.lifelong.sum())} / 後発型 {int(PW.late_only.sum())} / 前半のみ {int(PW.early_only.sum())}')
for lab, col in [('後発型（中年以降にだけ動く）', 'late_only'), ('生涯型', 'lifelong')]:
    s = PW[PW[col]]
    if not len(s): continue
    w(f'\n  【{lab}】{len(s)} 件、細胞種 {s.cell_type.value_counts().to_dict()}')
    w(f'    プログラム内訳 {s.program.value_counts().to_dict()}')
    top = s.reindex(s.rho_old.abs().sort_values(ascending=False).index).head(20)
    for _, r in top.iterrows():
        w(f'    {str(r.near_gene):14s} {r.cell_type:12s} {r.program:14s} '
          f'前半 {r.rho_young:+.3f} → 後半 {r.rho_old:+.3f}')

gl = set(PW[PW.lifelong].near_gene.dropna()); gt = set(PW[PW.late_only].near_gene.dropna())
inter = gl & gt
w(f'\n  生涯型 遺伝子 {len(gl)} / 後発型 遺伝子 {len(gt)} / 共有 {len(inter)}')
if gl and gt:
    allg = set(PW.near_gene.dropna())
    tab = [[len(inter), len(gt - gl)], [len(gl - gt), len(allg - gl - gt)]]
    orr, pp = fisher_exact(tab)
    w(f'  重なりの OR {orr:.2f} (p={pp:.3g}) — 1 に近ければ「別の遺伝子群」')
    w(f'  共有: {", ".join(sorted(inter)[:20])}')
w('\n★読み方: 後半だけ動く tCRE が別の遺伝子群を作るなら、中年以降の老化は前半の延長ではない。')
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')
