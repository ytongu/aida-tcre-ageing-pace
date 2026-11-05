#!/usr/bin/env python3
from __future__ import annotations
import pickle, os, gzip, time
import pandas as pd, numpy as np
from scipy.stats import spearmanr, rankdata, binomtest

B   = '${PROJECT_DIR}/results'
D   = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
TS  = '_v4'
OUT = f'{D}/results/fixes/longitudinal_fix'
os.makedirs(OUT, exist_ok=True)
TV  = 'yugene_diff_EN_tAge'
TAGE = ('${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/'
        'subcluster/tage_all_combined_v2.csv')
GTF = '${DATA_DIR}/adachi/reference/GRCh38.113.gtf.gz'
T0 = time.time()
L = []
def w(s=''):
    print(s, flush=True); L.append(str(s))
def flush():
    open(f'{OUT}/report.txt', 'w').write('\n'.join(L))

w('=' * 110)
w('縦断解析の再実装 — 細胞種別 Coverage による深度統制の引き直しと置換検定')
w('=' * 110)

tg = pd.read_csv(TAGE)
tg = tg[(tg.model_type == 'Chronoage') & (tg['mode'] == 'EN')].copy()
w(f'[0] tAge 表 (Chronoage/EN): {len(tg):,} 行 / 細胞種 {tg.CellType.nunique()} / '
  f'ドナー {tg.Donor_id.nunique()} / donor-batch {tg.groupby(["Donor_id","Batch"]).ngroups}')

db = tg[['Donor_id', 'Batch', 'Age', 'Sex']].drop_duplicates(['Donor_id', 'Batch'])
db = db.sort_values(['Donor_id', 'Age'], kind='stable')
db['visit'] = db.groupby('Donor_id').cumcount() + 1
nv = db.groupby('Donor_id').size(); multi = set(nv[nv > 1].index)
w(f'[0] ドナー {db.Donor_id.nunique()} / donor-batch {len(db)} / 複数時点ドナー {len(multi)}')

cov_new = tg[['Donor_id', 'Batch', 'CellType', 'Coverage']].drop_duplicates(
    ['Donor_id', 'Batch', 'CellType']).rename(columns={'Coverage': 'cov_new'})

cov_old = tg[['Donor_id', 'Batch', 'Coverage']].drop_duplicates(
    ['Donor_id', 'Batch']).rename(columns={'Coverage': 'cov_old'})
chk = cov_new.merge(cov_old, on=['Donor_id', 'Batch'])
same = (chk.cov_new == chk.cov_old).mean()
g = tg.groupby(['Donor_id', 'Batch']).Coverage.nunique()
w(f'[0] Coverage が細胞種間で異なる donor-batch: {int((g>1).sum())}/{len(g)}')
w(f'[0] 旧 Coverage が正しい値と一致する (donor-batch x 細胞種) の割合: {100*same:.1f}%')
r_ = tg.groupby(['Donor_id','Batch']).Coverage.agg(['min','max'])
w(f'[0] donor-batch 内の Coverage の幅 (max/min) 中央値: {(r_["max"]/r_["min"]).median():.1f} 倍')
flush()

tage = tg[['Donor_id', 'Batch', 'CellType', TV]].rename(columns={TV: 'tage_res'})

ann = pd.read_csv(f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv',
                  usecols=['cre_id', 'proximity'])
w(f'[1] 注釈 {len(ann):,} 行 / cre_id 一意 {ann.cre_id.is_unique}')

clock_df = pd.read_csv(f'{D}/results/clock_contribution/clock_coefficients_named.csv')
w(f'[1] 時計係数ファイル {len(clock_df):,} 行 / coef!=0 {int((clock_df.coef!=0).sum()):,} '
  f'-> ファイルは非ゼロ係数のみを保持しており、非ゼロ絞り込みは実質適用済み')
clock = set(clock_df.loc[clock_df.coef != 0, 'human_sym'].astype(str).str.upper())
e2g = pd.read_csv(f'{B}/re2g_targets_clean/targets_clean.csv',
                  usecols=['cre_id', 'target_gene'])
e2g_clock = set(e2g[e2g.target_gene.astype(str).str.upper().isin(clock)].cre_id.unique())
w(f'[1] rE2G 標的が時計遺伝子である素子: {len(e2g_clock):,}')

tssd, gb = {}, {}
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
        gb.setdefault(ch, []).append((int(p[3]), int(p[4])))
for ch in tssd: tssd[ch].sort()
POS = {ch: np.array([x[0] for x in v]) for ch, v in tssd.items()}
GBS, GBE = {}, {}
for ch, v in gb.items():
    v = sorted(v)
    s = np.array([x[0] for x in v]); e = np.maximum.accumulate(np.array([x[1] for x in v]))
    GBS[ch], GBE[ch] = s, e
w(f'[1] GTF タンパク質コード遺伝子: {sum(len(v) for v in tssd.values()):,} / 染色体 {len(tssd)}')

def parse_cre(cid):
    try:
        ch, s, e, st = cid.rsplit('_', 3); return ch, (int(s) + int(e)) // 2
    except Exception:
        return None, None

def nearest(cid):
    ch, mid = parse_cre(cid)
    arr = tssd.get(ch) if ch else None
    if not arr: return None
    i = int(np.searchsorted(POS[ch], mid)); best, bd = None, None
    for j in (i - 1, i):
        if 0 <= j < len(arr):
            dd = abs(arr[j][0] - mid)
            if bd is None or dd < bd: bd, best = dd, arr[j][1]
    return best

def in_genebody(cid):
    ch, mid = parse_cre(cid)
    if ch is None or ch not in GBS: return False
    i = int(np.searchsorted(GBS[ch], mid, side='right')) - 1
    return bool(i >= 0 and GBE[ch][i] >= mid)
flush()

a = pd.read_csv(f'{OUR}/cre_corr_aida_subcluster_significant_{TS}.csv',
                usecols=['cre_id', 'cell_type', 'target_var', 'spearman_rho', 'fdr'])
a = a[(a.target_var == TV) & (a.fdr < 0.05)].merge(ann, on='cre_id', how='left')
w(f'[2] AIDA 選抜 (Synapse 不使用): {len(a):,} 行 / subcluster {a.cell_type.nunique()}')

uni = pd.DataFrame({'cre_id': a.cre_id.unique()})
uni['gene'] = [nearest(c) for c in uni.cre_id]
uni['gene'] = uni.gene.astype(str).str.upper()
clock_cre = set(uni[uni.gene.isin(clock)].cre_id) | e2g_clock
w(f'[2] 時計に紐づく素子 (最近接 or rE2G 標的が時計遺伝子): {len(clock_cre):,}  '
  f'[監査報告 20,616 と照合]')
flush()

w(f'\n[3] Synapse 集約キャッシュ読み込み中... ({time.time()-T0:.0f}s)')
d = pickle.load(open(f'{OUR}/aggregation_cache_synapse_subcluster.pkl', 'rb'))
cres = np.array(d['cres']); keys = list(d['expr_sum'].keys())
w(f'[3] tCRE {len(cres):,} x メタセル {len(keys):,}  ({time.time()-T0:.0f}s)')
k0 = keys[0]
w(f'[3] キー例 {k0} / expr_sum shape {np.shape(d["expr_sum"][k0])} '
  f'/ umi_sum type {type(d["umi_sum"][k0]).__name__} ndim {np.ndim(d["umi_sum"][k0])} '
  f'/ n_cells {d["n_cells"][k0]}')

cre_idx = {c: i for i, c in enumerate(cres)}
prox = ann.set_index('cre_id').proximity.reindex(cres).values
m_distal = (prox == 'distal'); m_prox = (prox == 'proximal')
w(f'[3] 注釈が付いた tCRE: {int(pd.notna(prox).sum()):,}/{len(cres):,} '
  f'(遠位 {int(m_distal.sum()):,} / 近位 {int(m_prox.sum()):,})')
m_gb = np.array([in_genebody(c) for c in cres])
w(f'[3] 遺伝子体内 (タンパク質コード遺伝子の内側) の tCRE: {int(m_gb.sum()):,} '
  f'({100*m_gb.mean():.1f}%)  遠位のうち {100*m_gb[m_distal].mean():.1f}% / '
  f'近位のうち {100*m_gb[m_prox].mean():.1f}%')

rows = []
for i, k in enumerate(keys):
    ex = d['expr_sum'][k]
    ex = np.asarray(ex, dtype=np.float64)
    s_all = ex.sum(); s_di = ex[m_distal].sum(); s_pr = ex[m_prox].sum(); s_gb = ex[m_gb].sum()
    u = d['umi_sum'][k]; u = float(np.sum(u))
    rows.append((k[0], k[1], k[2], k[3], float(d['n_cells'][k]), u, s_all, s_di, s_pr, s_gb))
mcm = pd.DataFrame(rows, columns=['cohort','Donor_id','Batch','CellType','n_cells','umi_sum',
                                  's_all','s_distal','s_proximal','s_genebody'])
mcm = mcm.merge(db, on=['Donor_id','Batch'], how='left')
mcm = mcm.merge(cov_new, on=['Donor_id','Batch','CellType'], how='left')
mcm = mcm.merge(cov_old, on=['Donor_id','Batch'], how='left')
w(f'[3] メタセル表 {len(mcm):,} 行 / visit 欠損 {int(mcm.visit.isna().sum())} / '
  f'cov_new 欠損 {int(mcm.cov_new.isna().sum())} / cov_old 欠損 {int(mcm.cov_old.isna().sum())}')
mcm.to_csv(f'{OUT}/metacell_table.csv', index=False)
flush()

w('\n' + '=' * 110)
w('[4] 深度変数の分岐チェック — tAge 表の Coverage と 集約キャッシュの umi_sum は同じものか')
w('=' * 110)
q = mcm[mcm.cov_new.notna() & (mcm.umi_sum > 0)]
rr = spearmanr(q.cov_new, q.umi_sum)
w(f'[4] 全メタセル (n={len(q):,}): Spearman(Coverage_正, umi_sum) = {rr.statistic:+.4f}')
rr2 = spearmanr(q.cov_old, q.umi_sum)
w(f'[4] 全メタセル: Spearman(Coverage_旧誤, umi_sum) = {rr2.statistic:+.4f}')
w(f'[4] 全メタセル: Spearman(Coverage_正, s_all[CRE 合計]) = '
  f'{spearmanr(q.cov_new, q.s_all).statistic:+.4f}')
dep_rows = []
w(f'\n{"subcluster":14s}{"n":>6s}{"rho(Cov正,umi)":>16s}{"rho(Cov旧,umi)":>16s}'
  f'{"umi/Cov 中央":>14s}{"CRE合計/umi":>13s}')
for ct, gq in q.groupby('CellType'):
    r1 = spearmanr(gq.cov_new, gq.umi_sum).statistic
    r2 = spearmanr(gq.cov_old, gq.umi_sum).statistic
    w(f'{ct:14s}{len(gq):6d}{r1:+16.4f}{r2:+16.4f}'
      f'{(gq.umi_sum/gq.cov_new).median():14.3f}{(gq.s_all/gq.umi_sum).median():13.3f}')
    dep_rows.append(dict(CellType=ct, n=len(gq), rho_covnew_umi=r1, rho_covold_umi=r2,
                         med_umi_over_cov=(gq.umi_sum/gq.cov_new).median(),
                         med_sall_over_umi=(gq.s_all/gq.umi_sum).median()))
pd.DataFrame(dep_rows).to_csv(f'{OUT}/depth_variable_check.csv', index=False)
w(f'\n[4] umi_sum > CRE 合計 のメタセル: {int((q.umi_sum > q.s_all).sum()):,}/{len(q):,} '
  f'-> umi_sum が CRE 外の読みを含むなら遠位割合と近位割合は独立な量になる')
flush()

w('\n' + '=' * 110)
w('[5] 最も仮定の少ない指標 — 読みが遠位 tCRE に落ちる割合の個人内 Δ')
w('    素子選抜も z 化も背景差引も経ない。対照として近位割合・遺伝子体内割合も出す')
w('=' * 110)
mcm['f_distal_cre'] = mcm.s_distal / mcm.s_all.replace(0, np.nan)
mcm['f_prox_cre']   = mcm.s_proximal / mcm.s_all.replace(0, np.nan)
mcm['f_distal_umi'] = mcm.s_distal / mcm.umi_sum.replace(0, np.nan)
mcm['f_prox_umi']   = mcm.s_proximal / mcm.umi_sum.replace(0, np.nan)
mcm['f_cre_umi']    = mcm.s_all / mcm.umi_sum.replace(0, np.nan)
mcm['f_gb_cre']     = mcm.s_genebody / mcm.s_all.replace(0, np.nan)
w(f'[5] 遠位割合(CRE分母)と近位割合(CRE分母)の和: 中央 '
  f'{(mcm.f_distal_cre + mcm.f_prox_cre).median():.6f} -> 完全な鏡像なので独立な対照ではない')

def pcorr(x, y, Z):
    x = np.asarray(x, float); y = np.asarray(y, float); Z = np.asarray(Z, float)
    if Z.ndim == 1: Z = Z[:, None]
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), 1)
    x, y, Z = x[m], y[m], Z[m]
    if len(x) < 20: return np.nan, np.nan, len(x)
    rx, ry = rankdata(x), rankdata(y)
    RZ = np.c_[np.ones(len(x)), np.apply_along_axis(rankdata, 0, Z)]
    ex = rx - RZ @ np.linalg.lstsq(RZ, rx, rcond=None)[0]
    ey = ry - RZ @ np.linalg.lstsq(RZ, ry, rcond=None)[0]
    r = spearmanr(ex, ey)
    return r.statistic, r.pvalue, len(x)

MEAS = ['f_distal_cre', 'f_distal_umi', 'f_prox_umi', 'f_cre_umi', 'f_gb_cre']
crude_rows = []
base = mcm[mcm.visit.notna() & (mcm.n_cells >= 10)].merge(
    tage, on=['Donor_id','Batch','CellType'], how='left')
for ct, gq in base.groupby('CellType'):
    t = gq.copy()
    t['lcov_new'] = np.log10(t.cov_new.astype(float) + 1)
    t['lcov_old'] = np.log10(t.cov_old.astype(float) + 1)
    t['lnc'] = np.log10(t.n_cells.astype(float) + 1)
    cols = MEAS + ['tage_res', 'lcov_new', 'lcov_old', 'lnc']
    for c in cols:
        t[c + '_r'] = t[c] - t.groupby('Batch')[c].transform('mean')
    p_ = t[t.Donor_id.isin(multi)].sort_values(['Donor_id', 'visit'])
    f_, l_ = p_.groupby('Donor_id').first(), p_.groupby('Donor_id').last()
    keep = (l_.Age - f_.Age) > 0
    if int(keep.sum()) < 30: continue
    dT_raw = (l_.tage_res - f_.tage_res)[keep].values
    dT     = (l_.tage_res_r - f_.tage_res_r)[keep].values
    dCn    = (l_.lcov_new_r - f_.lcov_new_r)[keep].values
    dCo    = (l_.lcov_old_r - f_.lcov_old_r)[keep].values
    dN     = (l_.lnc_r - f_.lnc_r)[keep].values
    for meas in MEAS:
        dRaw = (l_[meas] - f_[meas])[keep].values
        dRes = (l_[meas + '_r'] - f_[meas + '_r'])[keep].values
        r_raw = spearmanr(dRaw, dT_raw, nan_policy='omit')
        r_bat = spearmanr(dRes, dT, nan_policy='omit')
        pn, ppn, n_ = pcorr(dRes, dT, np.c_[dCn, dN])
        po, ppo, _  = pcorr(dRes, dT, np.c_[dCo, dN])
        pc_only, _, _ = pcorr(dRes, dT, dCn)
        crude_rows.append(dict(CellType=ct, measure=meas, n=int(keep.sum()),
                               rho_raw=r_raw.statistic, p_raw=r_raw.pvalue,
                               rho_batchres=r_bat.statistic, p_batchres=r_bat.pvalue,
                               part_newcov=pn, part_p_newcov=ppn,
                               part_oldcov=po, part_p_oldcov=ppo,
                               part_covonly_new=pc_only, n_part=n_))
CR = pd.DataFrame(crude_rows)
CR.to_csv(f'{OUT}/crude_fraction_deltadelta.csv', index=False)
lab = {'f_distal_cre':'遠位割合(分母=CRE合計)', 'f_distal_umi':'遠位割合(分母=umi_sum)',
       'f_prox_umi':'近位割合(分母=umi_sum)', 'f_cre_umi':'CRE 内割合(分母=umi_sum)',
       'f_gb_cre':'遺伝子体内割合(分母=CRE合計)'}
for meas in MEAS:
    x = CR[CR.measure == meas].sort_values('rho_raw', ascending=False)
    if not len(x): continue
    w(f'\n--- {lab[meas]} : Δ(割合) x Δ(加齢残差) ---')
    w(f'{"subcluster":14s}{"n":>5s}{"生Δ":>9s}{"p":>8s}{"バッチ残差":>10s}{"p":>8s}'
      f'{"偏相関(正Cov)":>14s}{"p":>8s}{"偏相関(旧Cov)":>14s}{"Cov のみ":>10s}')
    for _, r in x.iterrows():
        w(f'{r.CellType:14s}{int(r.n):5d}{r.rho_raw:+9.3f}{r.p_raw:8.3f}'
          f'{r.rho_batchres:+10.3f}{r.p_batchres:8.3f}{r.part_newcov:+14.3f}{r.part_p_newcov:8.3f}'
          f'{r.part_oldcov:+14.3f}{r.part_covonly_new:+10.3f}')
    w(f'  正の細胞種: 生 {int((x.rho_raw>0).sum())}/{len(x)} / '
      f'偏相関(正Cov) {int((x.part_newcov>0).sum())}/{len(x)}')
flush()

w('\n' + '=' * 110)
w('[6] プログラム得点 (AIDA 選抜) の個人内 Δ x Δ(加齢残差)')
w('    旧経路 = donor-batch 単位の誤 Coverage / 新経路 = (donor,batch,細胞種) の正 Coverage')
w('=' * 110)

prog_rows, ho_rows = [], []
DELTAS = {}
for ct in sorted(set(mcm.CellType) & set(a.cell_type)):
    sel = mcm[(mcm.CellType == ct) & mcm.visit.notna() & (mcm.n_cells >= 10)]
    if sel.Donor_id.nunique() < 50: continue
    idx = sel.index.tolist()
    kk = [keys[i] for i in idx]
    M = np.vstack([np.asarray(d['expr_sum'][k]) for k in kk]).astype(np.float32)
    tot = M.sum(1, keepdims=True); tot[tot == 0] = 1
    M = np.log2(M / tot * 1e6 + 1)
    det = (M > 0).mean(0) >= 0.20
    M = M - np.median(M[:, det], axis=1, keepdims=True)
    mu = M.mean(0); sd = M.std(0); sd[sd == 0] = 1
    Z = (M - mu) / sd
    sm = sel.reset_index(drop=True)
    prg = a[a.cell_type == ct]

    def ids(sign, prx, drop_clock):
        s = prg[(np.sign(prg.spearman_rho) == sign) & (prg.proximity == prx)].cre_id
        if drop_clock: s = [c for c in s if c not in clock_cre]
        ix = np.array([cre_idx[c] for c in s if c in cre_idx], dtype=int)
        return ix[det[ix]] if len(ix) else ix

    for tag, drop in [('全素子', False), ('時計除外', True)]:
        iu, idn = ids(1, 'distal', drop), ids(-1, 'proximal', drop)
        if len(iu) < 30 or len(idn) < 30: continue
        rng = np.random.default_rng(0)
        _ex = set(iu.tolist()) | set(idn.tolist())
        pool = np.array([i for i in np.where(det)[0] if i not in _ex])
        qs = np.quantile(mu[pool], np.linspace(0, 1, 11))
        bins = np.clip(np.searchsorted(qs, mu[pool], side='right') - 1, 0, 9)
        def bg(ix):
            tb = np.clip(np.searchsorted(qs, mu[ix], side='right') - 1, 0, 9)
            o = []
            for b_ in range(10):
                need = int((tb == b_).sum()); cand = pool[bins == b_]
                if need and len(cand): o.append(rng.choice(cand, min(need, len(cand)), replace=False))
            return np.concatenate(o) if o else np.array([], dtype=int)
        bu, bd = bg(iu), bg(idn)
        t = sm.copy()
        t['up'] = Z[:, iu].mean(1) - (Z[:, bu].mean(1) if len(bu) else 0)
        t['dn'] = Z[:, idn].mean(1) - (Z[:, bd].mean(1) if len(bd) else 0)
        t = t.merge(tage[tage.CellType == ct][['Donor_id','Batch','tage_res']],
                    on=['Donor_id','Batch'], how='left')
        t['lcov_new'] = np.log10(t.cov_new.astype(float) + 1)
        t['lcov_old'] = np.log10(t.cov_old.astype(float) + 1)
        t['lnc'] = np.log10(t.n_cells.astype(float) + 1)
        for c in ['up','dn','tage_res','lcov_new','lcov_old','lnc']:
            t[c + '_r'] = t[c] - t.groupby('Batch')[c].transform('mean')
        p_ = t[t.Donor_id.isin(multi)].sort_values(['Donor_id','visit'])
        f_, l_ = p_.groupby('Donor_id').first(), p_.groupby('Donor_id').last()
        keep = (l_.Age - f_.Age) > 0
        dd = pd.DataFrame({
            'Donor_id': f_.index[keep],
            'dU': (l_.up_r - f_.up_r)[keep].values,
            'dD': (l_.dn_r - f_.dn_r)[keep].values,
            'dT': (l_.tage_res_r - f_.tage_res_r)[keep].values,
            'dCn': (l_.lcov_new_r - f_.lcov_new_r)[keep].values,
            'dCo': (l_.lcov_old_r - f_.lcov_old_r)[keep].values,
            'dN': (l_.lnc_r - f_.lnc_r)[keep].values})
        DELTAS[(ct, tag)] = dd
        ru = spearmanr(dd.dU, dd.dT, nan_policy='omit')
        rd = spearmanr(dd.dD, dd.dT, nan_policy='omit')
        pun, ppun, n_ = pcorr(dd.dU, dd.dT, np.c_[dd.dCn, dd.dN])
        pdn_, ppdn, _ = pcorr(dd.dD, dd.dT, np.c_[dd.dCn, dd.dN])
        puo, ppuo, _  = pcorr(dd.dU, dd.dT, np.c_[dd.dCo, dd.dN])
        pdo, ppdo, _  = pcorr(dd.dD, dd.dT, np.c_[dd.dCo, dd.dN])

        rTC = spearmanr(dd.dT, dd.dCn, nan_policy='omit').statistic
        rTCo = spearmanr(dd.dT, dd.dCo, nan_policy='omit').statistic
        rud = spearmanr(dd.dU, dd.dD, nan_policy='omit').statistic
        prog_rows.append(dict(CellType=ct, cond=tag, n=n_, n_up=len(iu), n_dn=len(idn),
                              rho_up=ru.statistic, p_up=ru.pvalue,
                              rho_dn=rd.statistic, p_dn=rd.pvalue,
                              part_up_new=pun, part_p_up_new=ppun,
                              part_dn_new=pdn_, part_p_dn_new=ppdn,
                              part_up_old=puo, part_p_up_old=ppuo,
                              part_dn_old=pdo, part_p_dn_old=ppdo,
                              rho_dT_dCovnew=rTC, rho_dT_dCovold=rTCo, rho_up_dn=rud))
        if tag == '全素子':

            for labh, q_ in [('訪問1 (使用済)', t[t.visit == 1]),
                             ('訪問2 のみ (未使用)', t[t.visit == 2]),
                             ('訪問2・3 (未使用)', t[t.visit >= 2])]:
                q_ = q_.dropna(subset=['tage_res'])
                if len(q_) < 40: continue
                r1 = spearmanr(q_.up_r, q_.tage_res_r); r2 = spearmanr(q_.dn_r, q_.tage_res_r)
                hu, hpu, nn = pcorr(q_.up_r, q_.tage_res_r, np.c_[q_.lcov_new_r, q_.lnc_r])
                hd, hpd, _  = pcorr(q_.dn_r, q_.tage_res_r, np.c_[q_.lcov_new_r, q_.lnc_r])
                huo, _, _   = pcorr(q_.up_r, q_.tage_res_r, np.c_[q_.lcov_old_r, q_.lnc_r])
                hdo, _, _   = pcorr(q_.dn_r, q_.tage_res_r, np.c_[q_.lcov_old_r, q_.lnc_r])
                ho_rows.append(dict(CellType=ct, subset=labh, n=len(q_), n_donor=q_.Donor_id.nunique(),
                                    rho_up=r1.statistic, p_up=r1.pvalue,
                                    rho_dn=r2.statistic, p_dn=r2.pvalue,
                                    part_up_new=hu, part_p_up_new=hpu,
                                    part_dn_new=hd, part_p_dn_new=hpd,
                                    part_up_old=huo, part_dn_old=hdo))
    del M, Z
    w(f'  [{ct}] 完了 ({time.time()-T0:.0f}s)')

R = pd.DataFrame(prog_rows); R.to_csv(f'{OUT}/deltadelta_coverage_compare.csv', index=False)
HO = pd.DataFrame(ho_rows); HO.to_csv(f'{OUT}/heldout_visits_coverage_fix.csv', index=False)

for tag in ['全素子', '時計除外']:
    x = R[R.cond == tag]
    if not len(x): continue
    w('\n' + '=' * 110)
    w(f'[6] {tag}   Δ(プログラム得点) x Δ(加齢残差)   [偏相関は Δlog10(Coverage) と Δlog10(n_cells) を除去]')
    w('=' * 110)
    w(f'{"subcluster":14s}{"n":>4s}{"上昇n":>7s}{"低下n":>7s}'
      f'{"Δ上昇xΔ齢":>11s}{"偏相関旧Cov":>12s}{"偏相関正Cov":>12s}{"p":>8s}'
      f'{"Δ低下xΔ齢":>11s}{"偏相関旧Cov":>12s}{"偏相関正Cov":>12s}{"p":>8s}'
      f'{"Δ齢xΔCov正":>12s}')
    for _, r in x.iterrows():
        w(f'{r.CellType:14s}{int(r.n):4d}{int(r.n_up):7,d}{int(r.n_dn):7,d}'
          f'{r.rho_up:+11.3f}{r.part_up_old:+12.3f}{r.part_up_new:+12.3f}{r.part_p_up_new:8.3f}'
          f'{r.rho_dn:+11.3f}{r.part_dn_old:+12.3f}{r.part_dn_new:+12.3f}{r.part_p_dn_new:8.3f}'
          f'{r.rho_dT_dCovnew:+12.3f}')
    ok = int((x.rho_up > 0).sum() + (x.rho_dn < 0).sum()); tot = 2 * len(x)
    w(f'\n  [6] {tag}: 予測どおりの符号 {ok}/{tot} (二項検定 p={binomtest(ok,tot,0.5,alternative="greater").pvalue:.2g} '
      f'— 非独立なので過大。置換検定は [7] 参照)')
    okn = int((x.part_up_new > 0).sum() + (x.part_dn_new < 0).sum())
    oko = int((x.part_up_old > 0).sum() + (x.part_dn_old < 0).sum())
    w(f'  [6] {tag}: 正しい Coverage の偏相関で予測どおり {okn}/{tot} / '
      f'旧 (誤) Coverage では {oko}/{tot}')
    dl = (x.part_up_new - x.part_up_old).abs().tolist() + (x.part_dn_new - x.part_dn_old).abs().tolist()
    w(f'  [6] {tag}: 旧 Coverage と正 Coverage の偏相関の差 (絶対値) 中央 {np.median(dl):.3f} / 最大 {np.max(dl):.3f}')
    big = x[( (x.part_up_new - x.part_up_old).abs() > 0.05 ) | ( (x.part_dn_new - x.part_dn_old).abs() > 0.05 )]
    for _, r in big.iterrows():
        w(f'    差が大きい: {r.CellType:12s} 上昇 {r.part_up_old:+.3f} -> {r.part_up_new:+.3f} / '
          f'低下 {r.part_dn_old:+.3f} -> {r.part_dn_new:+.3f}')
flush()

w('\n' + '=' * 110)
w('[8] 訪問2・3 だけでの横断相関 (held-out) — 細胞種別 Coverage で引き直し')
w('    ★ 訪問2 のみがドナー1人1行。訪問2・3 は同一ドナーが2行入る擬似反復あり')
w('=' * 110)
w(f'{"subcluster":14s}{"部分集合":20s}{"n":>5s}{"人":>5s}{"上昇rho":>9s}{"p":>8s}'
  f'{"偏相関正Cov":>12s}{"p":>8s}{"低下rho":>9s}{"p":>8s}{"偏相関正Cov":>12s}{"p":>8s}')
for _, r in HO.iterrows():
    w(f'{r.CellType:14s}{r.subset:20s}{int(r.n):5d}{int(r.n_donor):5d}'
      f'{r.rho_up:+9.3f}{r.p_up:8.3f}{r.part_up_new:+12.3f}{r.part_p_up_new:8.3f}'
      f'{r.rho_dn:+9.3f}{r.p_dn:8.3f}{r.part_dn_new:+12.3f}{r.part_p_dn_new:8.3f}')
for s_ in HO.subset.unique():
    x = HO[HO.subset == s_]
    w(f'  [8] {s_}: 予測どおりの符号 生 {int((x.rho_up>0).sum()+(x.rho_dn<0).sum())}/{2*len(x)} / '
      f'正 Coverage 偏相関 {int((x.part_up_new>0).sum()+(x.part_dn_new<0).sum())}/{2*len(x)}')
flush()

w('\n' + '=' * 110)
w('[7] 大域置換検定 — 1回の並べ替えにつきドナーのラベル 1 通りを 9 subcluster に同時適用')
w('    細胞種ごとに別の並べ替えをしない (9 細胞種は同じドナー集団を共有するため)')
w('=' * 110)

def fast_spear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10: return np.nan
    rx = rankdata(x[m]); ry = rankdata(y[m])
    rx = rx - rx.mean(); ry = ry - ry.mean()
    dnm = np.sqrt((rx*rx).sum() * (ry*ry).sum())
    return float(rx @ ry / dnm) if dnm > 0 else np.nan

def pcorr_fast(x, y, Z):
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), 1)
    if m.sum() < 20: return np.nan
    xx, yy, zz = x[m], y[m], Z[m]
    rx, ry = rankdata(xx), rankdata(yy)
    RZ = np.c_[np.ones(m.sum()), np.apply_along_axis(rankdata, 0, zz)]
    B_, *_ = np.linalg.lstsq(RZ, np.c_[rx, ry], rcond=None)
    e = np.c_[rx, ry] - RZ @ B_
    return fast_spear(e[:, 0], e[:, 1])

for TAGCOND in ['全素子', '時計除外']:
    cts = [ct for (ct, cd) in DELTAS if cd == TAGCOND]
    cts = sorted(cts)
    if len(cts) < 5: continue
    master = sorted(set().union(*[set(DELTAS[(c, TAGCOND)].Donor_id) for c in cts]))
    mi = {dn_: i for i, dn_ in enumerate(master)}
    nM = len(master)
    U = np.full((len(cts), nM), np.nan); Dm = np.full((len(cts), nM), np.nan)
    T = np.full((len(cts), nM), np.nan); Cn = np.full((len(cts), nM), np.nan)
    Nn = np.full((len(cts), nM), np.nan)
    for j, ct in enumerate(cts):
        dd = DELTAS[(ct, TAGCOND)]
        ii = np.array([mi[x] for x in dd.Donor_id])
        U[j, ii] = dd.dU.values; Dm[j, ii] = dd.dD.values; T[j, ii] = dd.dT.values
        Cn[j, ii] = dd.dCn.values; Nn[j, ii] = dd.dN.values
    w(f'\n--- {TAGCOND} --- subcluster {len(cts)} / ドナー(和集合) {nM}')
    w(f'  細胞種ごとのドナー数: ' + ' '.join(f'{c}={int(np.isfinite(T[j]).sum())}'
      for j, c in enumerate(cts)))

    def stats_from(Tmat):
        ru = np.array([fast_spear(U[j], Tmat[j]) for j in range(len(cts))])
        rd = np.array([fast_spear(Dm[j], Tmat[j]) for j in range(len(cts))])
        nsign = int((ru > 0).sum() + (rd < 0).sum())
        S = float(np.nansum(ru) - np.nansum(rd))
        return nsign, S, ru, rd
    def stats_part(Tmat):
        ru = np.array([pcorr_fast(U[j], Tmat[j], np.c_[Cn[j], Nn[j]]) for j in range(len(cts))])
        rd = np.array([pcorr_fast(Dm[j], Tmat[j], np.c_[Cn[j], Nn[j]]) for j in range(len(cts))])
        nsign = int((ru > 0).sum() + (rd < 0).sum())
        S = float(np.nansum(ru) - np.nansum(rd))
        return nsign, S, ru, rd

    obs_n, obs_S, ru0, rd0 = stats_from(T)
    obsP_n, obsP_S, ruP, rdP = stats_part(T)
    w(f'  実測 (生 rho)   : 予測どおりの符号 {obs_n}/{2*len(cts)} / S = Σρ(上昇) − Σρ(低下) = {obs_S:+.4f}')
    w(f'    Σρ(上昇) = {np.nansum(ru0):+.4f}  Σρ(低下) = {np.nansum(rd0):+.4f}')
    w(f'  実測 (正Cov 偏相関): 予測どおりの符号 {obsP_n}/{2*len(cts)} / S = {obsP_S:+.4f}')

    NPERM = 10000
    rng = np.random.default_rng(_v5)
    nn_null = np.empty(NPERM); SS_null = np.empty(NPERM)
    nnP_null = np.empty(NPERM); SSP_null = np.empty(NPERM)
    for b_ in range(NPERM):
        pi = rng.permutation(nM)
        Tp = T[:, pi]
        nn_null[b_], SS_null[b_], _, _ = stats_from(Tp)
        nnP_null[b_], SSP_null[b_], _, _ = stats_part(Tp)
    def rep(name, obs, null):
        ge = int((null >= obs).sum())
        pv = (ge + 1) / (NPERM + 1)
        w(f'  {name}: 実測 {obs:+.4f} / 帰無 平均 {null.mean():+.4f} SD {null.std(ddof=1):.4f} '
          f'最大 {null.max():+.4f} / 帰無を超えなかった回数 {ge}/{NPERM} -> p = {pv:.2e} '
          f'({(obs-null.mean())/null.std(ddof=1):+.1f} SD)')
        return pv, ge
    p1, g1 = rep('符号数 (生 rho)      ', obs_n, nn_null)
    p2, g2 = rep('連続統計量 S (生 rho)', obs_S, SS_null)
    p3, g3 = rep('符号数 (正Cov 偏相関)', obsP_n, nnP_null)
    p4, g4 = rep('連続統計量 S (正Cov) ', obsP_S, SSP_null)
    w(f'  ※ 超過 0 回のときの p は 1/(NPERM+1) が下限であり「これ以下」という意味しか持たない')
    pd.DataFrame(dict(perm=np.arange(NPERM), nsign=nn_null, S=SS_null,
                      nsign_part=nnP_null, S_part=SSP_null)).to_csv(
        f'{OUT}/permutation_null_{"all" if TAGCOND=="全素子" else "noclock"}.csv', index=False)

    CM = np.full((len(cts), len(cts)), np.nan)
    for i_ in range(len(cts)):
        for j_ in range(len(cts)):
            CM[i_, j_] = fast_spear(T[i_], T[j_])
    cmdf = pd.DataFrame(CM, index=cts, columns=cts)
    cmdf.to_csv(f'{OUT}/dtage_celltype_corr_{"all" if TAGCOND=="全素子" else "noclock"}.csv')
    off = CM[np.triu_indices(len(cts), 1)]
    rbar = float(np.nanmean(off)); N_ = len(cts)
    neff = N_ / (1 + (N_ - 1) * rbar)
    w(f'  細胞種間 Δ(加齢残差) 相関: 平均 {rbar:+.3f} / 中央 {np.nanmedian(off):+.3f} / '
      f'範囲 {np.nanmin(off):+.3f}〜{np.nanmax(off):+.3f}')
    w(f'  有効独立細胞種数 Neff = N/(1+(N-1)*r_bar) = {N_}/(1+{N_-1}x{rbar:.3f}) = {neff:.2f}')
    if TAGCOND == '全素子':
        w('  相関行列:')
        w('           ' + ''.join(f'{c[:9]:>10s}' for c in cts))
        for i_, c in enumerate(cts):
            w(f'  {c:9s}' + ''.join(f'{CM[i_,j_]:+10.3f}' for j_ in range(len(cts))))
    pd.DataFrame([dict(cond=TAGCOND, n_ct=N_, n_donor_union=nM,
                       obs_nsign=obs_n, obs_S=obs_S, p_nsign=p1, p_S=p2,
                       null_S_mean=SS_null.mean(), null_S_sd=SS_null.std(ddof=1),
                       null_S_max=SS_null.max(), exceed_nsign=g1, exceed_S=g2,
                       obs_nsign_part=obsP_n, obs_S_part=obsP_S, p_nsign_part=p3, p_S_part=p4,
                       null_Spart_mean=SSP_null.mean(), null_Spart_sd=SSP_null.std(ddof=1),
                       null_Spart_max=SSP_null.max(),
                       mean_intercelltype_r=rbar, Neff=neff)]).to_csv(
        f'{OUT}/permutation_summary_{"all" if TAGCOND=="全素子" else "noclock"}.csv', index=False)
    flush()

w('\n' + '=' * 110)
w('[9] 横断正典で使う CD4 差し替え tAge は縦断に使えるか')
w('=' * 110)
cd4 = pd.read_csv(f'{D}/results/cd4_redo/cd4_our_tage.csv')
cs = cd4[cd4.cohort == 'Synapse']
w(f'[9] Synapse 分 {len(cs)} 行 / ドナー {cs.Donor_id.nunique()} / '
  f'donor-batch {cs.groupby(["Donor_id","Batch"]).ngroups} / Group {sorted(cs.Group.unique())}')
nun = cs.groupby(['Donor_id','Group'])[TV].nunique()
w(f'[9] 同一ドナー内で訪問をまたいで tAge が 1 値しか取らない (Donor, Group) の割合: '
  f'{100*(nun==1).mean():.1f}%  ({int((nun==1).sum())}/{len(nun)})')
w('[9] -> 差し替え版はドナー水準の値を全訪問に複製したものなので、個人内 Δ が定義上 0 になる。'
  '縦断には使用できない。縦断の CD4 は差し替え前の tAge を用いる (この制約を本文に書く)')
flush()

w('\n' + '=' * 110)
w('[10] 判定 — 「技術要因では説明できない」と書けるか')
w('=' * 110)
x = R[R.cond == '全素子']
xn = R[R.cond == '時計除外']
w(f'[10] 全素子: 生の符号一致 {int((x.rho_up>0).sum()+(x.rho_dn<0).sum())}/{2*len(x)} / '
  f'正 Coverage 偏相関 {int((x.part_up_new>0).sum()+(x.part_dn_new<0).sum())}/{2*len(x)}')
w(f'[10] 時計除外: 生の符号一致 {int((xn.rho_up>0).sum()+(xn.rho_dn<0).sum())}/{2*len(xn)} / '
  f'正 Coverage 偏相関 {int((xn.part_up_new>0).sum()+(xn.part_dn_new<0).sum())}/{2*len(xn)}')
cd = CR[CR.measure == 'f_distal_umi']
w(f'[10] 粗い指標(遠位割合 分母=umi_sum): 生で正 {int((cd.rho_raw>0).sum())}/{len(cd)} / '
  f'正 Coverage 偏相関で正 {int((cd.part_newcov>0).sum())}/{len(cd)}')
cd2 = CR[CR.measure == 'f_distal_cre']
w(f'[10] 粗い指標(遠位割合 分母=CRE合計): 生で正 {int((cd2.rho_raw>0).sum())}/{len(cd2)} / '
  f'正 Coverage 偏相関で正 {int((cd2.part_newcov>0).sum())}/{len(cd2)}')
w(f'\n[10] 実行時間 {time.time()-T0:.0f}s')
flush()
print('DONE ->', OUT)


import pickle, os, gzip, time
import pandas as pd, numpy as np
from scipy.stats import spearmanr, rankdata

B   = '${PROJECT_DIR}/results'
D   = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
TS  = '_v4'
OUT = f'{D}/results/fixes/longitudinal_fix'
TV  = 'yugene_diff_EN_tAge'
TAGE = ('${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/'
        'subcluster/tage_all_combined_v2.csv')
GTF = '${DATA_DIR}/adachi/reference/GRCh38.113.gtf.gz'
SUB9 = ['B_Naive','CD16_Mono','CD4_Memory','CD4_Naive','CD8_EM','CD8_Effector',
        'CD8_Naive','NK','NKT']
T0 = time.time(); L = []
def w(s=''):
    print(s, flush=True); L.append(str(s))
def flush(): open(f'{OUT}/report_addenda.txt','w').write('\n'.join(L))

def fspear(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m = np.isfinite(x)&np.isfinite(y)
    if m.sum()<10: return np.nan
    rx=rankdata(x[m]); ry=rankdata(y[m]); rx=rx-rx.mean(); ry=ry-ry.mean()
    dn=np.sqrt((rx*rx).sum()*(ry*ry).sum())
    return float(rx@ry/dn) if dn>0 else np.nan
def pcorr(x, y, Z):
    x=np.asarray(x,float); y=np.asarray(y,float); Z=np.asarray(Z,float)
    if Z.ndim==1: Z=Z[:,None]
    m=np.isfinite(x)&np.isfinite(y)&np.all(np.isfinite(Z),1)
    if m.sum()<20: return np.nan
    rx,ry=rankdata(x[m]),rankdata(y[m])
    RZ=np.c_[np.ones(m.sum()), np.apply_along_axis(rankdata,0,Z[m])]
    Bc,*_=np.linalg.lstsq(RZ, np.c_[rx,ry], rcond=None)
    e=np.c_[rx,ry]-RZ@Bc
    return fspear(e[:,0], e[:,1])

w('='*110); w('課題 A1 追補 — 2x2 分割 / 両側 p / 訪問3 単独 / 退化列フラグ'); w('='*110)

w('\n' + '='*110)
w('(B) プログラム置換検定の両側 p (保存済み帰無分布 10,000 回から算出)')
w('    符号数の帰無中心は 9/18、S の帰無中心は 0。|実測−中心| 以上のずれを両側で数える')
w('='*110)
obs_tab = {'all': (18, 4.9559, 18, 4.7845), 'noclock': (18, 4.7156, 18, 4.4815)}
tw_rows = []
for tag, fn in [('全素子','all'), ('時計除外','noclock')]:
    nl = pd.read_csv(f'{OUT}/permutation_null_{fn}.csv')
    o_n, o_S, o_np_, o_Sp = obs_tab[fn]
    for nm, obs, arr, center in [
        ('符号数(生rho)', o_n, nl.nsign.values, 9.0),
        ('S(生rho)', o_S, nl.S.values, 0.0),
        ('符号数(正Cov偏相関)', o_np_, nl.nsign_part.values, 9.0),
        ('S(正Cov偏相関)', o_Sp, nl.S_part.values, 0.0)]:
        ge1 = int((arr >= obs).sum()); p1 = (ge1+1)/(len(arr)+1)
        ge2 = int((np.abs(arr-center) >= abs(obs-center)).sum()); p2 = (ge2+1)/(len(arr)+1)
        w(f'  [{tag}] {nm:22s} 実測 {obs:+8.4f} / 帰無 平均 {arr.mean():+7.4f} SD {arr.std(ddof=1):6.4f} '
          f'最大 {arr.max():+7.4f} / 片側超過 {ge1} p={p1:.2e} / 両側超過 {ge2} p={p2:.2e}')
        tw_rows.append(dict(cond=tag, stat=nm, obs=obs, null_mean=arr.mean(),
                            null_sd=arr.std(ddof=1), null_max=arr.max(),
                            exceed_one=ge1, p_one=p1, exceed_two=ge2, p_two=p2))
pd.DataFrame(tw_rows).to_csv(f'{OUT}/permutation_twosided.csv', index=False)
flush()

w('\n' + '='*110)
w('(A) 遠位/近位 x 遺伝子体内/外 の 2x2 分割 — 遠位割合の上昇はどの区画で起きているか')
w('='*110)
ann = pd.read_csv(f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv',
                  usecols=['cre_id','proximity'])
gb = {}
with gzip.open(GTF,'rt') as f:
    for line in f:
        if line[0]=='#': continue
        p_=line.split('\t')
        if p_[2]!='gene' or 'gene_biotype "protein_coding"' not in p_[8]: continue
        ch=p_[0] if p_[0].startswith('chr') else 'chr'+p_[0]
        gb.setdefault(ch,[]).append((int(p_[3]),int(p_[4])))
GBS,GBE={},{}
for ch,v in gb.items():
    v=sorted(v); GBS[ch]=np.array([x[0] for x in v]); GBE[ch]=np.maximum.accumulate(np.array([x[1] for x in v]))
def in_gb(cid):
    try: ch,s,e,st=cid.rsplit('_',3); mid=(int(s)+int(e))//2
    except Exception: return False
    if ch not in GBS: return False
    i=int(np.searchsorted(GBS[ch], mid, side='right'))-1
    return bool(i>=0 and GBE[ch][i]>=mid)

d = pickle.load(open(f'{OUR}/aggregation_cache_synapse_subcluster.pkl','rb'))
cres=np.array(d['cres']); keys=list(d['expr_sum'].keys())
prox = ann.set_index('cre_id').proximity.reindex(cres).values
m_di=(prox=='distal'); m_pr=(prox=='proximal')
m_gb=np.array([in_gb(c) for c in cres])
Q = {'dist_gb': m_di&m_gb, 'dist_nogb': m_di&~m_gb, 'prox_gb': m_pr&m_gb, 'prox_nogb': m_pr&~m_gb}
w('  区画ごとの tCRE 数: ' + ' / '.join(f'{k}={int(v.sum()):,}' for k,v in Q.items()))
rows=[]
for k in keys:
    ex=np.asarray(d['expr_sum'][k], dtype=np.float64)
    r=dict(cohort=k[0], Donor_id=k[1], Batch=k[2], CellType=k[3], s_all=ex.sum())
    for nm,mk in Q.items(): r['s_'+nm]=ex[mk].sum()
    rows.append(r)
X2=pd.DataFrame(rows)
del d
X2.to_csv(f'{OUT}/metacell_2x2.csv', index=False)
w(f'  2x2 集計 完了 {len(X2):,} メタセル ({time.time()-T0:.0f}s)')
flush()

mcm = pd.read_csv(f'{OUT}/metacell_table.csv')
tg = pd.read_csv(TAGE); tg=tg[(tg.model_type=='Chronoage')&(tg['mode']=='EN')]
tage = tg[['Donor_id','Batch','CellType',TV]].rename(columns={TV:'tage_res'})
base = mcm.merge(X2.drop(columns=['cohort','s_all']), on=['Donor_id','Batch','CellType'], how='left')
base = base[base.visit.notna() & (base.n_cells>=10)].merge(tage, on=['Donor_id','Batch','CellType'], how='left')
nv = mcm.drop_duplicates(['Donor_id','Batch']).groupby('Donor_id').size()
multi=set(nv[nv>1].index)
MEAS = ['f_distal','f_gb'] + ['f_'+k for k in Q]
base['f_distal']=base.s_distal/base.s_all
base['f_gb']=base.s_genebody/base.s_all
for k in Q: base['f_'+k]=base['s_'+k]/base.s_all
base['lcov_new']=np.log10(base.cov_new.astype(float)+1)
base['lnc']=np.log10(base.n_cells.astype(float)+1)

DEL={}
for ct,g in base.groupby('CellType'):
    t=g.copy()
    for c in MEAS+['tage_res','lcov_new','lnc']:
        t[c+'_r']=t[c]-t.groupby('Batch')[c].transform('mean')
    q=t[t.Donor_id.isin(multi)].sort_values(['Donor_id','visit'])
    f_,l_=q.groupby('Donor_id').first(), q.groupby('Donor_id').last()
    keep=(l_.Age-f_.Age)>0
    if int(keep.sum())<30: continue
    dd=pd.DataFrame({'Donor_id':f_.index[keep],
        'dT':(l_.tage_res_r-f_.tage_res_r)[keep].values,
        'dCn':(l_.lcov_new_r-f_.lcov_new_r)[keep].values,
        'dN':(l_.lnc_r-f_.lnc_r)[keep].values})
    for c in MEAS: dd['d_'+c]=(l_[c+'_r']-f_[c+'_r'])[keep].values
    DEL[ct]=dd

w('\n  2x2 区画ごとの Δ(割合) x Δ(加齢残差)  [バッチ残差化後 / 括弧内は深度・細胞数の偏相関後]')
w(f'{"subcluster":14s}{"n":>5s}' + ''.join(f'{k:>22s}' for k in ['遠位x体内','遠位x体外','近位x体内','近位x体外']))
q2_rows=[]
for ct in sorted(DEL):
    dd=DEL[ct]; line=f'{ct:14s}{len(dd):5d}'; r_={'CellType':ct,'n':len(dd)}
    for k in Q:
        rr=fspear(dd['d_f_'+k].values, dd.dT.values)
        pp=pcorr(dd['d_f_'+k], dd.dT, np.c_[dd.dCn, dd.dN])
        line+=f'{rr:+11.3f}({pp:+.3f})'; r_['rho_'+k]=rr; r_['part_'+k]=pp
    w(line); q2_rows.append(r_)
Q2=pd.DataFrame(q2_rows); Q2.to_csv(f'{OUT}/crude_2x2_percelltype.csv', index=False)
for k in Q:
    w(f'  {k:10s}: 正 {int((Q2["rho_"+k]>0).sum())}/{len(Q2)} (生) / '
      f'{int((Q2["part_"+k]>0).sum())}/{len(Q2)} (偏相関) / Σρ = {Q2["rho_"+k].sum():+.3f}')
flush()

w('\n' + '='*110)
w('(C) 粗い指標の大域置換検定 — 帰無分布を保存し両側 p も出す (10,000 回)')
w('='*110)
def perm(cts, key, part, npm=10000, seed=_v5):
    master=sorted(set().union(*[set(DEL[c].Donor_id) for c in cts]))
    mi={x:i for i,x in enumerate(master)}; nM=len(master)
    X=np.full((len(cts),nM),np.nan); T=np.full((len(cts),nM),np.nan)
    C=np.full((len(cts),nM),np.nan); N=np.full((len(cts),nM),np.nan)
    for j,c in enumerate(cts):
        dd=DEL[c]; ii=np.array([mi[x] for x in dd.Donor_id])
        X[j,ii]=dd['d_'+key].values; T[j,ii]=dd.dT.values
        C[j,ii]=dd.dCn.values; N[j,ii]=dd.dN.values
    def st(Tm):
        if part: r=np.array([pcorr(X[j],Tm[j],np.c_[C[j],N[j]]) for j in range(len(cts))])
        else:    r=np.array([fspear(X[j],Tm[j]) for j in range(len(cts))])
        return int(np.nansum(r>0)), float(np.nansum(r))
    on,oS=st(T)
    rng=np.random.default_rng(seed); nn=np.empty(npm); SS=np.empty(npm)
    for b in range(npm):
        pi=rng.permutation(nM); nn[b],SS[b]=st(T[:,pi])
    ctr=len(cts)/2.0
    def two(obs,arr,center):
        g1=int((arr>=obs).sum()); g2=int((np.abs(arr-center)>=abs(obs-center)).sum())
        return g1,(g1+1)/(npm+1),g2,(g2+1)/(npm+1)
    a1,p1,a2,p2=two(on,nn,ctr); b1,q1,b2,q2=two(oS,SS,0.0)
    return dict(n_ct=len(cts), obs_pos=on, obs_S=oS,
                pos_one=a1,p_pos_one=p1,pos_two=a2,p_pos_two=p2,
                S_one=b1,p_S_one=q1,S_two=b2,p_S_two=q2,
                null_pos_mean=nn.mean(),null_pos_sd=nn.std(ddof=1),null_pos_max=nn.max(),
                null_S_mean=SS.mean(),null_S_sd=SS.std(ddof=1),
                null_S_max=SS.max(),null_S_min=SS.min()), nn, SS
cases=[('遠位割合 12細胞種 生rho', sorted(DEL),'f_distal',False),
       ('遠位割合 12細胞種 偏相関', sorted(DEL),'f_distal',True),
       ('遠位割合 9subcluster 偏相関',[c for c in SUB9 if c in DEL],'f_distal',True),
       ('遺伝子体内割合 12細胞種 生rho', sorted(DEL),'f_gb',False),
       ('遠位x体外割合 12細胞種 生rho', sorted(DEL),'f_dist_nogb',False),
       ('遠位x体内割合 12細胞種 生rho', sorted(DEL),'f_dist_gb',False)]
out=[]
for lab,cts,key,part in cases:
    r,nn,SS=perm(cts,key,part); r['case']=lab; out.append(r)
    pd.DataFrame({'nsign':nn,'S':SS}).to_csv(
        f'{OUT}/crude_null_{key}_{"part" if part else "raw"}_{len(cts)}ct.csv', index=False)
    w(f'\n[{lab}] 細胞種 {r["n_ct"]}')
    w(f'  実測 正 {r["obs_pos"]}/{r["n_ct"]} / S = {r["obs_S"]:+.4f}')
    w(f'  符号数: 帰無 平均 {r["null_pos_mean"]:.3f} SD {r["null_pos_sd"]:.3f} 最大 {r["null_pos_max"]:.0f}'
      f' / 片側 {r["pos_one"]}/10000 p={r["p_pos_one"]:.2e} / 両側 {r["pos_two"]}/10000 p={r["p_pos_two"]:.2e}')
    w(f'  S     : 帰無 平均 {r["null_S_mean"]:+.4f} SD {r["null_S_sd"]:.4f} '
      f'最小 {r["null_S_min"]:+.4f} 最大 {r["null_S_max"]:+.4f}'
      f' / 片側 {r["S_one"]}/10000 p={r["p_S_one"]:.2e} / 両側 {r["S_two"]}/10000 p={r["p_S_two"]:.2e}'
      f' ({(r["obs_S"]-r["null_S_mean"])/r["null_S_sd"]:+.1f} SD)')
pd.DataFrame(out).to_csv(f'{OUT}/crude_permutation_twosided.csv', index=False)
flush()

cf=pd.read_csv(f'{OUT}/crude_fraction_deltadelta.csv')
note={'f_distal_cre':'primary','f_distal_umi':'DUPLICATE_of_f_distal_cre (umi_sum == CRE 合計)',
      'f_prox_umi':'MIRROR_of_f_distal (1 - 遠位割合、独立な対照ではない)',
      'f_cre_umi':'DEGENERATE (恒等的に 1、Δ は常に 0。数値は無意味)',
      'f_gb_cre':'independent_partition'}
cf['status']=cf.measure.map(note)
cf.to_csv(f'{OUT}/crude_fraction_deltadelta.csv', index=False)
w('\n(E) crude_fraction_deltadelta.csv に status 列を追加:')
for k,v in note.items(): w(f'    {k:15s} -> {v}')
w(f'\n完了 {time.time()-T0:.0f}s')
flush()
print('DONE ->', OUT)


import os, time
import pandas as pd, numpy as np
from scipy.stats import spearmanr, rankdata

B   = '${PROJECT_DIR}/results'
D   = f'{B}/PAPER_CRE_MECHANISM'
OUT = f'{D}/results/fixes/longitudinal_fix'
TV  = 'yugene_diff_EN_tAge'
TAGE = ('${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/'
        'subcluster/tage_all_combined_v2.csv')
SUB9 = ['B_Naive','CD16_Mono','CD4_Memory','CD4_Naive','CD8_EM','CD8_Effector',
        'CD8_Naive','NK','NKT']
L = []
def w(s=''):
    print(s, flush=True); L.append(str(s))
def flush():
    open(f'{OUT}/report_crude.txt', 'w').write('\n'.join(L))

mcm = pd.read_csv(f'{OUT}/metacell_table.csv')
tg = pd.read_csv(TAGE)
tg = tg[(tg.model_type == 'Chronoage') & (tg['mode'] == 'EN')]
tage = tg[['Donor_id','Batch','CellType',TV]].rename(columns={TV:'tage_res'})
nv = mcm.drop_duplicates(['Donor_id','Batch']).groupby('Donor_id').size()
multi = set(nv[nv > 1].index)

w('=' * 110)
w('粗い指標 (読みが遠位 tCRE に落ちる割合) — 大域置換検定と特異性')
w('=' * 110)
w(f'メタセル {len(mcm):,} / ドナー {mcm.Donor_id.nunique()} / 複数時点ドナー {len(multi)}')
w(f'umi_sum と CRE 合計が一致するメタセル: {int((mcm.umi_sum == mcm.s_all).sum()):,}/{len(mcm):,}')
w('-> 分母をどちらにしても同じ量。近位割合は 1 − 遠位割合 の完全な鏡像であり独立な対照ではない')
w(f'遠位割合の水準 (全メタセル): 中央 {(mcm.s_distal/mcm.s_all).median():.4f} / '
  f'四分位 {(mcm.s_distal/mcm.s_all).quantile(0.25):.4f}〜{(mcm.s_distal/mcm.s_all).quantile(0.75):.4f}')

def pcorr(x, y, Z):
    x=np.asarray(x,float); y=np.asarray(y,float); Z=np.asarray(Z,float)
    if Z.ndim == 1: Z = Z[:, None]
    m = np.isfinite(x)&np.isfinite(y)&np.all(np.isfinite(Z),1)
    if m.sum() < 20: return np.nan
    xx,yy,zz = x[m],y[m],Z[m]
    rx,ry = rankdata(xx),rankdata(yy)
    RZ = np.c_[np.ones(m.sum()), np.apply_along_axis(rankdata,0,zz)]
    Bc,*_ = np.linalg.lstsq(RZ, np.c_[rx,ry], rcond=None)
    e = np.c_[rx,ry] - RZ@Bc
    return float(spearmanr(e[:,0], e[:,1]).statistic)

def fspear(x, y):
    m = np.isfinite(x)&np.isfinite(y)
    if m.sum() < 10: return np.nan
    rx = rankdata(x[m]); ry = rankdata(y[m])
    rx = rx-rx.mean(); ry = ry-ry.mean()
    dn = np.sqrt((rx*rx).sum()*(ry*ry).sum())
    return float(rx@ry/dn) if dn>0 else np.nan

base = mcm[mcm.visit.notna() & (mcm.n_cells >= 10)].merge(
    tage, on=['Donor_id','Batch','CellType'], how='left')
base['f_distal'] = base.s_distal / base.s_all
base['f_gb'] = base.s_genebody / base.s_all
base['lcov_new'] = np.log10(base.cov_new.astype(float) + 1)
base['lcov_old'] = np.log10(base.cov_old.astype(float) + 1)
base['lnc'] = np.log10(base.n_cells.astype(float) + 1)

DEL = {}
rows = []
for ct, g in base.groupby('CellType'):
    t = g.copy()
    for c in ['f_distal','f_gb','tage_res','lcov_new','lcov_old','lnc']:
        t[c+'_r'] = t[c] - t.groupby('Batch')[c].transform('mean')
    q = t[t.Donor_id.isin(multi)].sort_values(['Donor_id','visit'])
    f_, l_ = q.groupby('Donor_id').first(), q.groupby('Donor_id').last()
    keep = (l_.Age - f_.Age) > 0
    if int(keep.sum()) < 30: continue
    dd = pd.DataFrame({'Donor_id': f_.index[keep],
        'dF': (l_.f_distal_r - f_.f_distal_r)[keep].values,
        'dFraw': (l_.f_distal - f_.f_distal)[keep].values,
        'dG': (l_.f_gb_r - f_.f_gb_r)[keep].values,
        'dT': (l_.tage_res_r - f_.tage_res_r)[keep].values,
        'dTraw': (l_.tage_res - f_.tage_res)[keep].values,
        'dCn': (l_.lcov_new_r - f_.lcov_new_r)[keep].values,
        'dCo': (l_.lcov_old_r - f_.lcov_old_r)[keep].values,
        'dN': (l_.lnc_r - f_.lnc_r)[keep].values})
    DEL[ct] = dd
    rows.append(dict(CellType=ct, n=len(dd),
        rho_F_T=fspear(dd.dF.values, dd.dT.values),
        part_F_T_new=pcorr(dd.dF, dd.dT, np.c_[dd.dCn, dd.dN]),
        rho_F_cov=fspear(dd.dF.values, dd.dCn.values),
        rho_T_cov=fspear(dd.dT.values, dd.dCn.values),
        rho_F_ncell=fspear(dd.dF.values, dd.dN.values),
        rho_G_T=fspear(dd.dG.values, dd.dT.values),
        part_G_T_new=pcorr(dd.dG, dd.dT, np.c_[dd.dCn, dd.dN])))
RW = pd.DataFrame(rows); RW.to_csv(f'{OUT}/crude_perm_percelltype.csv', index=False)

w('\n' + '=' * 110)
w('(b) 粗い指標は深度の代理指標か — Δ(遠位割合) と Δ深度・Δ細胞数 の相関')
w('=' * 110)
w(f'{"subcluster":14s}{"n":>5s}{"Δ遠位xΔ齢":>11s}{"偏相関":>9s}'
  f'{"Δ遠位xΔ深度":>13s}{"Δ齢xΔ深度":>11s}{"Δ遠位xΔ細胞数":>15s}'
  f'{"Δ遺体内xΔ齢":>13s}{"偏相関":>9s}')
for _, r in RW.sort_values('rho_F_T', ascending=False).iterrows():
    w(f'{r.CellType:14s}{int(r.n):5d}{r.rho_F_T:+11.3f}{r.part_F_T_new:+9.3f}'
      f'{r.rho_F_cov:+13.3f}{r.rho_T_cov:+11.3f}{r.rho_F_ncell:+15.3f}'
      f'{r.rho_G_T:+13.3f}{r.part_G_T_new:+9.3f}')
w(f'\n  Δ(遠位割合) と Δ深度 の相関: 中央 {RW.rho_F_cov.median():+.3f} / '
  f'範囲 {RW.rho_F_cov.min():+.3f}〜{RW.rho_F_cov.max():+.3f}')
w(f'  Δ(遠位割合) と Δ細胞数 の相関: 中央 {RW.rho_F_ncell.median():+.3f} / '
  f'範囲 {RW.rho_F_ncell.min():+.3f}〜{RW.rho_F_ncell.max():+.3f}')
flush()

def perm_test(cts, key_x, use_partial, npermut=10000, seed=_v5):
    master = sorted(set().union(*[set(DEL[c].Donor_id) for c in cts]))
    mi = {d_:i for i,d_ in enumerate(master)}; nM = len(master)
    X = np.full((len(cts), nM), np.nan); T = np.full((len(cts), nM), np.nan)
    C = np.full((len(cts), nM), np.nan); N = np.full((len(cts), nM), np.nan)
    for j,c in enumerate(cts):
        dd = DEL[c]; ii = np.array([mi[x] for x in dd.Donor_id])
        X[j,ii] = dd[key_x].values; T[j,ii] = dd.dT.values
        C[j,ii] = dd.dCn.values;    N[j,ii] = dd.dN.values
    def stat(Tm):
        if use_partial:
            r = np.array([pcorr(X[j], Tm[j], np.c_[C[j], N[j]]) for j in range(len(cts))])
        else:
            r = np.array([fspear(X[j], Tm[j]) for j in range(len(cts))])
        return int(np.nansum(r > 0)), float(np.nansum(r)), r
    on, oS, r0 = stat(T)
    rng = np.random.default_rng(seed)
    nn = np.empty(npermut); SS = np.empty(npermut)
    for b in range(npermut):
        pi = rng.permutation(nM)
        nn[b], SS[b], _ = stat(T[:, pi])
    def pv(o, nul): 
        ge = int((nul >= o).sum()); return (ge+1)/(npermut+1), ge
    p1,g1 = pv(on, nn); p2,g2 = pv(oS, SS)
    return dict(n_ct=len(cts), n_donor=nM, obs_pos=on, obs_S=oS,
                p_pos=p1, exceed_pos=g1, p_S=p2, exceed_S=g2,
                null_pos_mean=nn.mean(), null_pos_sd=nn.std(ddof=1), null_pos_max=nn.max(),
                null_S_mean=SS.mean(), null_S_sd=SS.std(ddof=1), null_S_max=SS.max(),
                rhos=';'.join(f'{c}={v:+.3f}' for c,v in zip(cts, r0)))

w('\n' + '=' * 110)
w('(a) 大域置換検定 — ドナーラベル 1 通りを全細胞種に同時適用 (10,000 回)')
w('    統計量: 予測どおりの符号 (正) の細胞種数 と S = Σρ')
w('=' * 110)
cases = [('遠位割合 全12細胞種 生rho', sorted(DEL), 'dF', False),
         ('遠位割合 全12細胞種 偏相関', sorted(DEL), 'dF', True),
         ('遠位割合 9subcluster 生rho', [c for c in SUB9 if c in DEL], 'dF', False),
         ('遠位割合 9subcluster 偏相関', [c for c in SUB9 if c in DEL], 'dF', True),
         ('遺伝子体内割合 全12細胞種 生rho', sorted(DEL), 'dG', False)]
res = []
for lab, cts, key, part in cases:
    r = perm_test(cts, key, part)
    r['case'] = lab; res.append(r)
    w(f'\n[{lab}]  細胞種 {r["n_ct"]} / ドナー(和集合) {r["n_donor"]}')
    w(f'  実測: 正の細胞種 {r["obs_pos"]}/{r["n_ct"]} / S = {r["obs_S"]:+.4f}')
    w(f'  符号数: 帰無 平均 {r["null_pos_mean"]:.3f} SD {r["null_pos_sd"]:.3f} '
      f'最大 {r["null_pos_max"]:.0f} / 超過 {r["exceed_pos"]}/10000 -> p = {r["p_pos"]:.2e}')
    w(f'  S     : 帰無 平均 {r["null_S_mean"]:+.4f} SD {r["null_S_sd"]:.4f} '
      f'最大 {r["null_S_max"]:+.4f} / 超過 {r["exceed_S"]}/10000 -> p = {r["p_S"]:.2e} '
      f'({(r["obs_S"]-r["null_S_mean"])/r["null_S_sd"]:+.1f} SD)')
    w(f'  細胞種別 rho: {r["rhos"]}')
pd.DataFrame(res).to_csv(f'{OUT}/crude_permutation_summary.csv', index=False)
flush()

w('\n' + '=' * 110)
w('(d) 横断 — メタセル水準で 遠位割合 と 加齢残差 は相関するか (バッチ残差化後)')
w('=' * 110)
w(f'{"subcluster":14s}{"訪問1 n":>9s}{"rho":>8s}{"偏相関":>9s}{"訪問2 n":>9s}{"rho":>8s}{"偏相関":>9s}')
cs_rows = []
for ct, g in base.groupby('CellType'):
    t = g.copy()
    for c in ['f_distal','tage_res','lcov_new','lnc']:
        t[c+'_r'] = t[c] - t.groupby('Batch')[c].transform('mean')
    out = {'CellType': ct}
    line = f'{ct:14s}'
    for v in [1, 2]:
        q = t[(t.visit == v)].dropna(subset=['tage_res'])
        if len(q) < 40:
            line += f'{len(q):9d}{"na":>8s}{"na":>9s}'; continue
        r1 = fspear(q.f_distal_r.values, q.tage_res_r.values)
        r2 = pcorr(q.f_distal_r, q.tage_res_r, np.c_[q.lcov_new_r, q.lnc_r])
        line += f'{len(q):9d}{r1:+8.3f}{r2:+9.3f}'
        out[f'v{v}_n'] = len(q); out[f'v{v}_rho'] = r1; out[f'v{v}_part'] = r2
    w(line); cs_rows.append(out)
CS = pd.DataFrame(cs_rows); CS.to_csv(f'{OUT}/crude_crosssectional.csv', index=False)
w(f'\n  訪問1 で正: {int((CS.v1_rho>0).sum())}/{int(CS.v1_rho.notna().sum())} / '
  f'訪問2 で正: {int((CS.v2_rho>0).sum())}/{int(CS.v2_rho.notna().sum())}')
w(f'  訪問1 偏相関で正: {int((CS.v1_part>0).sum())}/{int(CS.v1_part.notna().sum())} / '
  f'訪問2 偏相関で正: {int((CS.v2_part>0).sum())}/{int(CS.v2_part.notna().sum())}')
flush()
print('DONE ->', OUT)


import pickle, os, time
import pandas as pd, numpy as np
from scipy.stats import spearmanr, rankdata

B   = '${PROJECT_DIR}/results'
D   = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
TS  = '_v4'
OUT = f'{D}/results/fixes/longitudinal_fix'
TV  = 'yugene_diff_EN_tAge'
TAGE = ('${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/'
        'subcluster/tage_all_combined_v2.csv')
T0=time.time(); L=[]
def w(s=''):
    print(s, flush=True); L.append(str(s))
def flush(): open(f'{OUT}/report_visit3.txt','w').write('\n'.join(L))

def fspear(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<8: return np.nan
    rx=rankdata(x[m]); ry=rankdata(y[m]); rx=rx-rx.mean(); ry=ry-ry.mean()
    dn=np.sqrt((rx*rx).sum()*(ry*ry).sum())
    return float(rx@ry/dn) if dn>0 else np.nan
def pcorr(x,y,Z):
    x=np.asarray(x,float); y=np.asarray(y,float); Z=np.asarray(Z,float)
    if Z.ndim==1: Z=Z[:,None]
    m=np.isfinite(x)&np.isfinite(y)&np.all(np.isfinite(Z),1)
    if m.sum()<15: return np.nan
    rx,ry=rankdata(x[m]),rankdata(y[m])
    RZ=np.c_[np.ones(m.sum()), np.apply_along_axis(rankdata,0,Z[m])]
    Bc,*_=np.linalg.lstsq(RZ,np.c_[rx,ry],rcond=None); e=np.c_[rx,ry]-RZ@Bc
    return fspear(e[:,0],e[:,1])

tg=pd.read_csv(TAGE); tg=tg[(tg.model_type=='Chronoage')&(tg['mode']=='EN')].copy()
tage=tg[['Donor_id','Batch','CellType',TV]].rename(columns={TV:'tage_res'})
mcm=pd.read_csv(f'{OUT}/metacell_table.csv')
w('='*110); w('追補(D) held-out を訪問ごとに分解 — 訪問2・3 プールの弱化の正体'); w('='*110)
w(f'訪問ごとのドナー数: ' + ' '.join(
    f'訪問{int(v)}={mcm[mcm.visit==v].Donor_id.nunique()}' for v in sorted(mcm.visit.dropna().unique())))

a=pd.read_csv(f'{OUR}/cre_corr_aida_subcluster_significant_{TS}.csv',
              usecols=['cre_id','cell_type','target_var','spearman_rho','fdr'])
a=a[(a.target_var==TV)&(a.fdr<0.05)]
ann=pd.read_csv(f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv',
                usecols=['cre_id','proximity'])
a=a.merge(ann,on='cre_id',how='left')
d=pickle.load(open(f'{OUR}/aggregation_cache_synapse_subcluster.pkl','rb'))
cres=np.array(d['cres']); keys=list(d['expr_sum'].keys()); cre_idx={c:i for i,c in enumerate(cres)}
w(f'キャッシュ読み込み完了 ({time.time()-T0:.0f}s)')

rows=[]
for ct in sorted(set(mcm.CellType)&set(a.cell_type)):
    sel=mcm[(mcm.CellType==ct)&mcm.visit.notna()&(mcm.n_cells>=10)]
    if sel.Donor_id.nunique()<50: continue
    kk=[keys[i] for i in sel.index.tolist()]
    M=np.vstack([np.asarray(d['expr_sum'][k]) for k in kk]).astype(np.float32)
    tot=M.sum(1,keepdims=True); tot[tot==0]=1
    M=np.log2(M/tot*1e6+1)
    det=(M>0).mean(0)>=0.20
    M=M-np.median(M[:,det],axis=1,keepdims=True)
    mu=M.mean(0); sd=M.std(0); sd[sd==0]=1
    Z=(M-mu)/sd; sm=sel.reset_index(drop=True); prg=a[a.cell_type==ct]
    def ids(sign,prx):
        s=prg[(np.sign(prg.spearman_rho)==sign)&(prg.proximity==prx)].cre_id
        ix=np.array([cre_idx[c] for c in s if c in cre_idx],dtype=int)
        return ix[det[ix]] if len(ix) else ix
    iu,idn=ids(1,'distal'),ids(-1,'proximal')
    if len(iu)<30 or len(idn)<30: 
        del M,Z; continue
    rng=np.random.default_rng(0); _ex=set(iu.tolist())|set(idn.tolist())
    pool=np.array([i for i in np.where(det)[0] if i not in _ex])
    qs=np.quantile(mu[pool],np.linspace(0,1,11))
    bins=np.clip(np.searchsorted(qs,mu[pool],side='right')-1,0,9)
    def bg(ix):
        tb=np.clip(np.searchsorted(qs,mu[ix],side='right')-1,0,9); o=[]
        for b_ in range(10):
            need=int((tb==b_).sum()); cand=pool[bins==b_]
            if need and len(cand): o.append(rng.choice(cand,min(need,len(cand)),replace=False))
        return np.concatenate(o) if o else np.array([],dtype=int)
    bu,bd=bg(iu),bg(idn)
    t=sm.copy()
    t['up']=Z[:,iu].mean(1)-(Z[:,bu].mean(1) if len(bu) else 0)
    t['dn']=Z[:,idn].mean(1)-(Z[:,bd].mean(1) if len(bd) else 0)
    t=t.merge(tage[tage.CellType==ct][['Donor_id','Batch','tage_res']],on=['Donor_id','Batch'],how='left')
    t['lcov_new']=np.log10(t.cov_new.astype(float)+1); t['lnc']=np.log10(t.n_cells.astype(float)+1)
    for c in ['up','dn','tage_res','lcov_new','lnc']:
        t[c+'_r']=t[c]-t.groupby('Batch')[c].transform('mean')
    for lab,q in [('訪問1',t[t.visit==1]),('訪問2',t[t.visit==2]),('訪問3',t[t.visit==3]),
                  ('訪問2・3',t[t.visit>=2])]:
        q=q.dropna(subset=['tage_res'])
        if len(q)<15: continue
        rows.append(dict(CellType=ct, subset=lab, n=len(q), n_donor=q.Donor_id.nunique(),
            rho_up=fspear(q.up_r.values,q.tage_res_r.values),
            p_up=spearmanr(q.up_r,q.tage_res_r).pvalue,
            rho_dn=fspear(q.dn_r.values,q.tage_res_r.values),
            p_dn=spearmanr(q.dn_r,q.tage_res_r).pvalue,
            part_up=pcorr(q.up_r,q.tage_res_r,np.c_[q.lcov_new_r,q.lnc_r]),
            part_dn=pcorr(q.dn_r,q.tage_res_r,np.c_[q.lcov_new_r,q.lnc_r])))
    del M,Z
    w(f'  [{ct}] 完了 ({time.time()-T0:.0f}s)')
H=pd.DataFrame(rows); H.to_csv(f'{OUT}/heldout_by_visit.csv', index=False)
w('\n'+'='*110)
w('訪問ごとの held-out 横断相関 (バッチ残差化後 / 偏相関は細胞種別 Coverage と n_cells を除去)')
w('='*110)
w(f'{"subcluster":14s}{"部分集合":10s}{"n":>5s}{"人":>5s}{"上昇rho":>9s}{"p":>8s}{"偏相関":>9s}'
  f'{"低下rho":>9s}{"p":>8s}{"偏相関":>9s}')
for _,r in H.iterrows():
    w(f'{r.CellType:14s}{r.subset:10s}{int(r.n):5d}{int(r.n_donor):5d}{r.rho_up:+9.3f}{r.p_up:8.3f}'
      f'{r.part_up:+9.3f}{r.rho_dn:+9.3f}{r.p_dn:8.3f}{r.part_dn:+9.3f}')
w('')
for s_ in ['訪問1','訪問2','訪問3','訪問2・3']:
    x=H[H.subset==s_]
    if not len(x): continue
    w(f'  {s_:8s}: 予測どおりの符号 生 {int((x.rho_up>0).sum()+(x.rho_dn<0).sum())}/{2*len(x)} / '
      f'偏相関 {int((x.part_up>0).sum()+(x.part_dn<0).sum())}/{2*len(x)}  '
      f'(細胞種 {len(x)} / 中央 n={int(x.n.median())})')
c4=H[H.CellType.isin(['CD4_Memory','CD4_Naive'])]
w('\n  CD4 の 2 群だけ抜き出し:')
for _,r in c4.iterrows():
    w(f'    {r.CellType:12s}{r.subset:10s} n={int(r.n):4d} 上昇 {r.rho_up:+.3f} (p={r.p_up:.3f}) / '
      f'低下 {r.rho_dn:+.3f} (p={r.p_dn:.3f})')
flush(); print('DONE ->', OUT)


import os
import numpy as np, pandas as pd
from scipy.stats import rankdata
D = '${PAPER_DIR}'
OLD = f'{D}/results/fixes/longitudinal_fix'
OUT = f'{D}/results/longitudinal_cd4drop'
os.makedirs(OUT, exist_ok=True)
TV = 'yugene_diff_EN_tAge'
TAGE = f'{D}/results/tage_v110_check/tage_all_combined_v4.csv'
CD4 = {'CD4_Naive', 'CD4_Memory'}
L = []
def w(s=''):
    print(s, flush=True); L.append(str(s))

def fspear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10: return np.nan
    rx = rankdata(x[m]); ry = rankdata(y[m]); rx = rx - rx.mean(); ry = ry - ry.mean()
    dn = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float(rx @ ry / dn) if dn > 0 else np.nan

def pcorr(x, y, Z):
    x = np.asarray(x, float); y = np.asarray(y, float); Z = np.asarray(Z, float)
    if Z.ndim == 1: Z = Z[:, None]
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), 1)
    if m.sum() < 20: return np.nan
    rx, ry = rankdata(x[m]), rankdata(y[m])
    RZ = np.c_[np.ones(m.sum()), np.apply_along_axis(rankdata, 0, Z[m])]
    Bc, *_ = np.linalg.lstsq(RZ, np.c_[rx, ry], rcond=None)
    e = np.c_[rx, ry] - RZ @ Bc
    return fspear(e[:, 0], e[:, 1])

Q = ['dist_gb', 'dist_nogb', 'prox_gb', 'prox_nogb']
mcm = pd.read_csv(f'{OLD}/metacell_table.csv')
X2 = pd.read_csv(f'{OLD}/metacell_2x2.csv')
tg = pd.read_csv(TAGE); tg = tg[(tg.model_type == 'Chronoage') & (tg['mode'] == 'EN')]
tage = tg[['Donor_id', 'Batch', 'CellType', TV]].rename(columns={TV: 'tage_res'})
base = mcm.merge(X2.drop(columns=[c for c in ['cohort', 's_all'] if c in X2.columns]), on=['Donor_id', 'Batch', 'CellType'], how='left')
base = base[base.visit.notna() & (base.n_cells >= 10)].merge(tage, on=['Donor_id', 'Batch', 'CellType'], how='left')
nv = mcm.drop_duplicates(['Donor_id', 'Batch']).groupby('Donor_id').size()
multi = set(nv[nv > 1].index)
MEAS = ['f_distal', 'f_gb'] + ['f_' + k for k in Q]
base['f_distal'] = base.s_distal / base.s_all
base['f_gb'] = base.s_genebody / base.s_all
for k in Q: base['f_' + k] = base['s_' + k] / base.s_all
base['lcov_new'] = np.log10(base.cov_new.astype(float) + 1)
base['lnc'] = np.log10(base.n_cells.astype(float) + 1)
DEL = {}
for ct, g in base.groupby('CellType'):
    t = g.copy()
    for c in MEAS + ['tage_res', 'lcov_new', 'lnc']:
        t[c + '_r'] = t[c] - t.groupby('Batch')[c].transform('mean')
    q = t[t.Donor_id.isin(multi)].sort_values(['Donor_id', 'visit'])
    f_, l_ = q.groupby('Donor_id').first(), q.groupby('Donor_id').last()
    keep = (l_.Age - f_.Age) > 0
    if int(keep.sum()) < 30: continue
    dd = pd.DataFrame({'Donor_id': f_.index[keep], 'dT': (l_.tage_res_r - f_.tage_res_r)[keep].values,
                       'dCn': (l_.lcov_new_r - f_.lcov_new_r)[keep].values, 'dN': (l_.lnc_r - f_.lnc_r)[keep].values})
    for c in MEAS: dd['d_' + c] = (l_[c + '_r'] - f_[c + '_r'])[keep].values
    DEL[ct] = dd
CTS = [c for c in sorted(DEL) if c not in CD4]
w(f'細胞種 {len(DEL)} -> CD4 除外 {len(CTS)}: {CTS}')

old = pd.read_csv(f'{OLD}/crude_perm_percelltype.csv').set_index('CellType')
oq = pd.read_csv(f'{OLD}/crude_2x2_percelltype.csv').set_index('CellType')
rows = []
for ct in CTS:
    dd = DEL[ct]
    r_raw = fspear(dd.d_f_distal, dd.dT); r_part = pcorr(dd.d_f_distal, dd.dT, np.c_[dd.dCn, dd.dN])
    assert abs(r_raw - old.loc[ct, 'rho_F_T']) < 1e-6, (ct, r_raw, old.loc[ct, 'rho_F_T'])
    row = dict(CellType=ct, n=len(dd), rho_F_T=r_raw, part_F_T_new=r_part)
    for k in Q:
        row['rho_' + k] = fspear(dd['d_f_' + k], dd.dT)
        assert abs(row['rho_' + k] - oq.loc[ct, 'rho_' + k]) < 1e-6
    rows.append(row)
pd.DataFrame(rows).to_csv(f'{OUT}/crude_percelltype_cd4drop.csv', index=False)
w('CD4 以外の細胞種別 rho は旧表と一致（1e-6）')

def perm(cts, key, part, npm=10000, seed=_v5):
    master = sorted(set().union(*[set(DEL[c].Donor_id) for c in cts]))
    mi = {x: i for i, x in enumerate(master)}; nM = len(master)
    X = np.full((len(cts), nM), np.nan); T = np.full((len(cts), nM), np.nan)
    C = np.full((len(cts), nM), np.nan); N = np.full((len(cts), nM), np.nan)
    for j, c in enumerate(cts):
        dd = DEL[c]; ii = np.array([mi[x] for x in dd.Donor_id])
        X[j, ii] = dd['d_' + key].values; T[j, ii] = dd.dT.values
        C[j, ii] = dd.dCn.values; N[j, ii] = dd.dN.values
    def st(Tm):
        if part: r = np.array([pcorr(X[j], Tm[j], np.c_[C[j], N[j]]) for j in range(len(cts))])
        else: r = np.array([fspear(X[j], Tm[j]) for j in range(len(cts))])
        return int(np.nansum(r > 0)), float(np.nansum(r))
    on, oS = st(T)
    rng = np.random.default_rng(seed); nn = np.empty(npm); SS = np.empty(npm)
    for b in range(npm):
        pi = rng.permutation(nM); nn[b], SS[b] = st(T[:, pi])
    return dict(n_ct=len(cts), n_donor=nM, obs_pos=on, obs_S=oS, exceed_pos=int((nn >= on).sum()),
                p_pos=(int((nn >= on).sum()) + 1) / (npm + 1), exceed_S=int((SS >= oS).sum()),
                p_S=(int((SS >= oS).sum()) + 1) / (npm + 1), null_S_mean=SS.mean(), null_S_sd=SS.std(ddof=1),
                null_S_max=SS.max()), nn, SS

out = []
for lab, key, part in [('distal raw', 'f_distal', False), ('distal partial', 'f_distal', True),
                       ('gene-body raw', 'f_gb', False), ('distal x gene-body raw', 'f_dist_gb', False),
                       ('distal x no-gene-body raw', 'f_dist_nogb', False)]:
    r, nn, SS = perm(CTS, key, part); r['case'] = lab; out.append(r)
    pd.DataFrame({'nsign': nn, 'S': SS}).to_csv(f'{OUT}/crude_null_{key}_{"part" if part else "raw"}_{len(CTS)}ct.csv', index=False)
    w(f'[{lab}] 正 {r["obs_pos"]}/{r["n_ct"]}  S {r["obs_S"]:+.4f}  帰無 最大 {r["null_S_max"]:+.4f}  '
      f'超過 {r["exceed_S"]}/10000  p = {r["p_S"]:.2e}')
pd.DataFrame(out).to_csv(f'{OUT}/crude_permutation_summary_cd4drop.csv', index=False)
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))



import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

D = Path('${PROJECT_DIR}/results/'
         'PAPER_CRE_MECHANISM')
SRC = D / 'results' / 'fixes' / 'longitudinal_fix' / 'metacell_scores.csv'
OUT = D / 'results' / 'longitudinal_cd4drop' / 'visit3power'
OUT.mkdir(parents=True, exist_ok=True)
L: list[str] = []

def w(s=''):
    print(s, flush=True)
    L.append(str(s))

def fspear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return np.nan
    rx = rankdata(x[m]); ry = rankdata(y[m])
    rx = rx - rx.mean(); ry = ry - ry.mean()
    dn = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float(rx @ ry / dn) if dn > 0 else np.nan

S = pd.read_csv(SRC)
ALL_CT = sorted(S.CellType.unique())
CTS7 = [c for c in ALL_CT if not c.startswith('CD4_')]
w(f'細胞種 {len(ALL_CT)}: {ALL_CT}')
w(f'CD4 除外 {len(CTS7)}: {CTS7}')

def stat_for(df, cts, donors=None):
    ns = Ssum = ok = 0
    for ct in cts:
        q = df[df.CellType == ct]
        if donors is not None:
            q = q[q.Donor_id.isin(donors)]
        ru = fspear(q.up_r.values, q.tage_res_r.values)
        rd = fspear(q.dn_r.values, q.tage_res_r.values)
        if np.isfinite(ru):
            ns += int(ru > 0); Ssum += ru; ok += 1
        if np.isfinite(rd):
            ns += int(rd < 0); Ssum -= rd; ok += 1
    return ns, Ssum, ok

v = {k: S[S.visit == k] for k in (1, 2, 3)}
_rr = []
for _k in (1, 2, 3):
    for _ct in CTS7:
        _q = v[_k][v[_k].CellType == _ct]
        _rr.append(dict(visit=_k, CellType=_ct, rho_up=fspear(_q.up_r.values, _q.tage_res_r.values),
                        rho_dn=fspear(_q.dn_r.values, _q.tage_res_r.values), n=_q.Donor_id.nunique()))
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(_rr).to_csv(OUT / 'per_visit_celltype_rho_14arm.csv', index=False)
w('\n--- 実測 ---')
obs = {}
for cts, tag in ((ALL_CT, '9 subcluster (18 腕, 公開値の再現)'),
                 (CTS7, '7 subcluster (14 腕, CD4 除外)')):
    line = []
    for k in (1, 2, 3):
        n_, s_, ok_ = stat_for(v[k], cts)
        obs[(tag, k)] = (n_, s_, ok_)
        line.append(f'訪問{k} 符号 {n_}/{ok_} S={s_:+.3f}')
    w(f'  {tag}: ' + ' / '.join(line))

n3 = v[3].Donor_id.nunique()
w(f'\n訪問3 のドナー数 {n3} -> 訪問1・2 から {n3} 人を無作為抽出して比較 (2,000 回)')
o3 = obs[('7 subcluster (14 腕, CD4 除外)', 3)]

rows = []
rng = np.random.default_rng(_v5)
for lab, k in (('訪問1', 1), ('訪問2', 2)):
    sub = v[k]
    dn_all = np.array(sorted(sub.Donor_id.unique()))
    ns_d = np.empty(2000); S_d = np.empty(2000)
    for b in range(2000):
        dsel = set(rng.choice(dn_all, size=min(n3, len(dn_all)), replace=False))
        ns_d[b], S_d[b], _ = stat_for(sub, CTS7, dsel)
    p_ns = float((ns_d <= o3[0]).mean()); p_S = float((S_d <= o3[1]).mean())
    pd.DataFrame({'nsign': ns_d, 'S': S_d}).to_csv(OUT / f'visit3_power_null_{lab}_14arm.csv', index=False)
    w(f'\n[{lab} を {n3} 人に縮小、2,000 回・14 腕]')
    w(f'  符号数: 平均 {ns_d.mean():.2f} SD {ns_d.std(ddof=1):.2f} '
      f'5%点 {np.percentile(ns_d, 5):.1f} 最小 {ns_d.min():.0f}')
    w(f'    -> 訪問3 の実測 {o3[0]}/14 以下になった割合 = {100 * p_ns:.1f}%')
    w(f'  S     : 平均 {S_d.mean():+.3f} SD {S_d.std(ddof=1):.3f} '
      f'5%点 {np.percentile(S_d, 5):+.3f} 最小 {S_d.min():+.3f}')
    w(f'    -> 訪問3 の実測 S={o3[1]:+.3f} 以下になった割合 = {100 * p_S:.1f}%')
    rows.append(dict(source=lab, n_sub=n3, obs_v3_nsign=o3[0], obs_v3_S=o3[1],
                     sub_nsign_mean=ns_d.mean(), sub_nsign_sd=ns_d.std(ddof=1),
                     sub_nsign_p5=np.percentile(ns_d, 5), frac_le_nsign=p_ns,
                     sub_S_mean=S_d.mean(), sub_S_sd=S_d.std(ddof=1),
                     sub_S_p5=np.percentile(S_d, 5), frac_le_S=p_S))

pd.DataFrame(rows).to_csv(OUT / 'visit3_power_cd4drop.csv', index=False)
(OUT / 'report.txt').write_text('\n'.join(L) + '\n')
print('->', OUT)


import pickle, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
OUT = f'{D}/results/tage_movers'
os.makedirs(OUT, exist_ok=True)
TV = 'yugene_diff_EN_tAge'
rng = np.random.default_rng(0)
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

rob = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv',
                  usecols=['cre_id', 'cell_type', 'program'])
rob = rob[rob.program.isin(['up_distal', 'down_proximal'])]
PROG = rob.drop_duplicates('cre_id').set_index('cre_id').program.to_dict()
CT_OF = rob.drop_duplicates('cre_id').set_index('cre_id').cell_type.to_dict()
jo = pd.read_csv(f'{D}/results/gene_to_cre/cre_gene_joined.csv',
                 usecols=['cre_id', 'cell_type', 'mean_expression', 'proximity', 'robust', 'near_gene'])
GMAP = dict(zip(jo.drop_duplicates('cre_id').cre_id, jo.drop_duplicates('cre_id').near_gene))

fg = rob.merge(jo[['cre_id', 'cell_type', 'mean_expression', 'proximity']],
               on=['cre_id', 'cell_type'], how='left').dropna(subset=['mean_expression'])
BG = {}
for (ct, prox), g in fg.groupby(['cell_type', 'proximity']):
    pool = jo[(jo.cell_type == ct) & (jo.proximity == prox) & (~jo.robust)
              & (~jo.cre_id.isin(set(rob.cre_id)))].sort_values('mean_expression')
    if not len(pool): continue
    pv = pool.mean_expression.values; ids = pool.cre_id.values
    used = set()
    for _, f in g.iterrows():
        i = int(np.searchsorted(pv, f.mean_expression))
        for off in range(400):
            hit = False
            for j in (i + off, i - off):
                if 0 <= j < len(ids) and ids[j] not in used:
                    used.add(ids[j]); BG[ids[j]] = f.program; hit = True; break
            if hit: break
w(f'頑健 {len(PROG):,} / 発現量マッチ背景 {len(BG):,}')

ts = pd.read_csv('${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/'
                 'subcluster/tage_all_combined_v2.csv')
ts = ts[(ts.model_type == 'Chronoage') & (ts['mode'] == 'EN')]
ts['tage_r'] = ts[TV] - ts.groupby(['Batch', 'CellType'])[TV].transform('mean')
vis = ts[['Donor_id', 'Batch', 'Age']].drop_duplicates().sort_values(['Donor_id', 'Age'])
vis['visit'] = vis.groupby('Donor_id').cumcount() + 1
AGEKEY = {(r.Donor_id, r.Batch): r.Age for _, r in vis.iterrows()}
VISKEY = {(r.Donor_id, r.Batch): r.visit for _, r in vis.iterrows()}
ts['visit'] = [VISKEY.get((a, b), np.nan) for a, b in zip(ts.Donor_id, ts.Batch)]
w(f'tAge 訪問単位 {len(ts):,} 行（バッチ×細胞種 平均を差引済み）')

w('Synapse キャッシュを開く')
d = pickle.load(open(f'{OUR}/aggregation_cache_synapse_subcluster.pkl', 'rb'))
cres = np.array(d['cres']); keys = list(d['expr_sum'].keys())
idx = {c: i for i, c in enumerate(cres)}
tgt = [c for c in list(PROG) + list(BG) if c in idx]
tid = np.array([idx[c] for c in tgt])
role = np.array(['robust' if c in PROG else 'background' for c in tgt])
prg = np.array([PROG.get(c, BG.get(c)) for c in tgt])
meta = pd.DataFrame(keys, columns=['cohort_k', 'Donor_id', 'Batch', 'CellType'])
meta['n_cells'] = [d['n_cells'][k] for k in keys]
meta['age'] = [AGEKEY.get((a, b), np.nan) for a, b in zip(meta.Donor_id, meta.Batch)]
meta['visit'] = [VISKEY.get((a, b), np.nan) for a, b in zip(meta.Donor_id, meta.Batch)]
keep = (meta.n_cells >= 10) & meta.age.notna()
kk = [k for k, ok in zip(keys, keep) if ok]
meta = meta[keep].reset_index(drop=True)
sub = kk[::max(1, len(kk) // 300)]
det = np.zeros(len(cres))
for k in sub:
    det += (d['expr_sum'][k] > 0)
det /= len(sub)
ref = np.nonzero(det >= 0.8)[0]
if len(ref) > 20000:
    ref = rng.choice(ref, 20000, replace=False)
M = np.zeros((len(kk), len(tgt)), np.float32)
for i, k in enumerate(kk):
    v = d['expr_sum'][k]
    M[i] = np.log(v[tid] + 0.5) - np.log(v[ref] + 0.5).mean()
del d
w(f'CLR 行列 {M.shape}')

meta['up_rob'] = M[:, (role == 'robust') & (prg == 'up_distal')].mean(1)
meta['dn_rob'] = M[:, (role == 'robust') & (prg == 'down_proximal')].mean(1)
meta['up_bg'] = M[:, (role == 'background') & (prg == 'up_distal')].mean(1)
meta['dn_bg'] = M[:, (role == 'background') & (prg == 'down_proximal')].mean(1)
meta['up_c'] = meta.up_rob - meta.up_bg
meta['dn_c'] = meta.dn_rob - meta.dn_bg
meta['bal_c'] = meta.up_c - meta.dn_c

tam = ts.set_index(['Donor_id', 'Batch', 'CellType']).tage_r.to_dict()
rows = []
for (dn, ct), g in meta.groupby(['Donor_id', 'CellType']):
    g = g.sort_values('visit')
    if g.visit.nunique() < 2: continue
    a = g.iloc[0]; b = g.iloc[-1]
    t0 = tam.get((dn, a.Batch, ct)); t1 = tam.get((dn, b.Batch, ct))
    if t0 is None or t1 is None or not np.isfinite(t0) or not np.isfinite(t1): continue
    i0 = a.name; i1 = b.name
    rows.append(dict(Donor_id=dn, CellType=ct, age0=a.age, dt=b.age - a.age,
                     d_tage=t1 - t0, d_up=b.up_c - a.up_c, d_dn=b.dn_c - a.dn_c,
                     d_bal=b.bal_c - a.bal_c, i0=i0, i1=i1))
P = pd.DataFrame(rows)
P.to_csv(f'{OUT}/donor_deltas.csv', index=False)
w(f'\n変化を計算できた (ドナー, 細胞種) {len(P):,} / ドナー {P.Donor_id.nunique()}')
w(f'Δ tAge: 中央 {P.d_tage.median():+.2f} 年 / SD {P.d_tage.std():.2f} / '
  f'範囲 {P.d_tage.min():+.1f}〜{P.d_tage.max():+.1f}')

w('\n[1] tAge が大きく動いた人だけを見る（細胞種内で上位/下位 25%）')
P['grp'] = pd.Series(pd.NA, index=P.index, dtype='object')
for ct, g in P.groupby('CellType'):
    lo, hi = g.d_tage.quantile([0.25, 0.75])
    P.loc[g.index[g.d_tage <= lo], 'grp'] = 'down'
    P.loc[g.index[g.d_tage >= hi], 'grp'] = 'up'
U = P[P.grp == 'up']; Dn = P[P.grp == 'down']
w(f'  tAge 上がった {len(U)} 対（Δ 中央 {U.d_tage.median():+.2f} 年）/ '
  f'下がった {len(Dn)} 対（Δ 中央 {Dn.d_tage.median():+.2f} 年）')
for col, lab in [('d_up', '点火プログラム'), ('d_dn', '減衰プログラム'), ('d_bal', '差分')]:
    u = mannwhitneyu(U[col], Dn[col])
    w(f'  {lab:16s} 上がった群 {U[col].median():+.4f} / 下がった群 {Dn[col].median():+.4f} '
      f'(MWU p={u.pvalue:.3g})')
r = spearmanr(P.d_tage, P.d_bal)
w(f'  全体: Δ tAge × Δ 差分 rho {r.statistic:+.3f} (p={r.pvalue:.3g}, n={len(P)})')
r2 = spearmanr(P.d_tage, P.d_up); r3 = spearmanr(P.d_tage, P.d_dn)
w(f'        Δ tAge × Δ 点火 rho {r2.statistic:+.3f} (p={r2.pvalue:.2g}) / '
  f'Δ 減衰 rho {r3.statistic:+.3f} (p={r3.pvalue:.2g})')

w('\n[2] 細胞種ごと（Δ tAge × Δ 差分）')
rows2 = []
for ct, g in P.groupby('CellType'):
    if len(g) < 25: continue
    rr = spearmanr(g.d_tage, g.d_bal)
    uu = g[g.grp == 'up']; dd = g[g.grp == 'down']
    mw = mannwhitneyu(uu.d_bal, dd.d_bal) if len(uu) >= 8 and len(dd) >= 8 else None
    rows2.append(dict(cell_type=ct, n=len(g), rho=rr.statistic, p=rr.pvalue,
                      up_med=uu.d_bal.median(), down_med=dd.d_bal.median(),
                      p_grp=mw.pvalue if mw else np.nan))
    w(f'  {ct:13s} n={len(g):3d} rho {rr.statistic:+.3f} (p={rr.pvalue:.3g}) '
      f'| 上がった群 {uu.d_bal.median():+.4f} 対 下がった群 {dd.d_bal.median():+.4f}')
R2 = pd.DataFrame(rows2)
R2.to_csv(f'{OUT}/per_celltype.csv', index=False)
w(f'  ★正の細胞種 {int((R2.rho>0).sum())}/{len(R2)}')

w('\n[3] どの tCRE が両群を分けたか')
rows3 = []
for ct, g in P.groupby('CellType'):
    uu = g[g.grp == 'up']; dd = g[g.grp == 'down']
    if len(uu) < 8 or len(dd) < 8: continue
    sel = np.array([(CT_OF.get(c) == ct) and (role[i] == 'robust') for i, c in enumerate(tgt)])
    if sel.sum() < 20: continue
    du = M[uu.i1.values][:, sel] - M[uu.i0.values][:, sel]
    dd_ = M[dd.i1.values][:, sel] - M[dd.i0.values][:, sel]
    ids = np.array(tgt)[sel]
    for j, c in enumerate(ids):
        try:
            mw = mannwhitneyu(du[:, j], dd_[:, j])
            rows3.append(dict(cre_id=c, cell_type=ct, program=PROG[c], near_gene=GMAP.get(c),
                              d_up_group=float(np.median(du[:, j])),
                              d_down_group=float(np.median(dd_[:, j])),
                              diff=float(np.median(du[:, j]) - np.median(dd_[:, j])),
                              p=mw.pvalue))
        except ValueError:
            pass
T3 = pd.DataFrame(rows3)
T3.to_csv(f'{OUT}/per_tcre_movers.csv', index=False)
if len(T3):
    w(f'  検定 {len(T3):,} 件 / p<0.05 {int((T3.p<0.05).sum())} 件（期待 {len(T3)*0.05:.0f}）')
    for prgm, sgn in [('up_distal', '+'), ('down_proximal', '−')]:
        s = T3[T3.program == prgm]
        exp_pos = int((s['diff'] > 0).sum())
        w(f'  {prgm}: tAge 上がった群でより上がった tCRE {exp_pos}/{len(s)} ({exp_pos/len(s)*100:.0f}%)')
    top = T3.nsmallest(20, 'p')
    w('  両群を最も強く分けた tCRE:')
    for _, r in top.iterrows():
        w(f'    {str(r.near_gene):14s} {r.cell_type:12s} {r.program:14s} '
          f'上がった群 {r.d_up_group:+.3f} / 下がった群 {r.d_down_group:+.3f} (p={r.p:.3g})')
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')


import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
R = Path('${PAPER_DIR}/results')
d = pd.read_csv(R / 'tage_movers/donor_deltas.csv')
d = d[~d.CellType.isin(['CD4_Naive', 'CD4_Memory'])]
L = [f'n={len(d)} donor x cell-type changes / {d.Donor_id.nunique()} donors / {d.CellType.nunique()} cell types (CD4 excluded)']
r = spearmanr(d.d_tage, d.d_bal); L.append(f'dtAge x d_diff rho {r[0]:+.3f} p={r[1]:.2e}')
L.append(f'ignition {spearmanr(d.d_tage, d.d_up)[0]:+.3f} / decay {spearmanr(d.d_tage, d.d_dn)[0]:+.3f}')
for ct, g in d.groupby('CellType'):
    rr = spearmanr(g.d_tage, g.d_bal); L.append(f'  {ct:12s} n={len(g)} rho {rr[0]:+.3f} p={rr[1]:.2g}')
L.append(f'positive: {sum(spearmanr(g.d_tage, g.d_bal)[0] > 0 for _, g in d.groupby("CellType"))}/{d.CellType.nunique()}')
(R / 'longitudinal_cd4drop/movers_continuous_cd4drop.txt').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
