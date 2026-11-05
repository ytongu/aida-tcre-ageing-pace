#!/usr/bin/env python3
import gzip, os, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, mannwhitneyu

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUT = f'{D}/results/raw_ctss_scan'
os.makedirs(OUT, exist_ok=True)
ROOT = '${DATA_DIR}/lnc_project_AIDA_dataset/aida_per_sample_bed'
TV = 'yugene_diff_EN_tAge'
rng = np.random.default_rng(0)
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

rob = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv',
                  usecols=['cre_id', 'program'])
uni = rob.drop_duplicates('cre_id').copy()
co = uni.cre_id.str.rsplit('_', n=3, expand=True)
uni['chrom'] = co[0]; uni['start'] = co[1].astype(int)
uni['end'] = co[2].astype(int); uni['strand'] = co[3]
prog_any = rob.groupby('cre_id').program.apply(set)
uni['is_dist_up'] = uni.cre_id.map(lambda c: 'up_distal' in prog_any[c])
uni['is_prox_dn'] = uni.cre_id.map(lambda c: 'down_proximal' in prog_any[c])
w(f'頑健 tCRE {len(uni):,}（up_distal {int(uni.is_dist_up.sum()):,} / down_proximal {int(uni.is_prox_dn.sum()):,}）')

bg_ids = set()
bgf = f'{D}/results/figures_main/cache/fig3b_matched_background.csv'
if os.path.exists(bgf):
    bg = pd.read_csv(bgf)
    col = [c for c in bg.columns if 'cre' in c.lower()][0]
    bg_ids = set(bg[col])
    w(f'発現マッチ背景: {len(bg_ids):,}')
else:
    w('★背景ファイル無し — G 率の背景比較は省略')
    bg = None
allids = list(uni.cre_id) + [c for c in bg_ids if c not in set(uni.cre_id)]
tab = uni.set_index('cre_id')
extra = []
for c in allids[len(uni):]:
    ch, s_, e_, st = c.rsplit('_', 3)
    extra.append(dict(cre_id=c, chrom=ch, start=int(s_), end=int(e_), strand=st,
                      is_dist_up=False, is_prox_dn=False))
if extra:
    tab = pd.concat([uni.set_index('cre_id'),
                     pd.DataFrame(extra).set_index('cre_id')])
tab = tab.reset_index()
N = len(tab)

by_ch = {}
for ch, g in tab.groupby('chrom'):
    g = g.sort_values('start').reset_index(drop=True)
    by_ch[ch] = dict(starts=g.start.values, ends=g.end.values,
                     strands=g.strand.values, idx=g.index.values,
                     gids=g.cre_id.values,
                     gpos={c: i for i, c in enumerate(g.cre_id.values)})
gid_row = {c: i for i, c in enumerate(tab.cre_id)}

def scan(bedpath, acc_row):
    with gzip.open(bedpath, 'rt') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            ch = p[0]
            d = by_ch.get(ch)
            if d is None: continue
            pos = int(p[1]); cnt = float(p[4]); strand = p[5]
            i = np.searchsorted(d['starts'], pos, side='right') - 1

            for j in (i, i - 1, i - 2):
                if 0 <= j < len(d['starts']) and d['starts'][j] <= pos < d['ends'][j]:
                    st = d['strands'][j]
                    if st in ('+', '-') and st != strand:
                        continue
                    acc_row[gid_row[d['gids'][j]]] += cnt

samples = sorted(os.listdir(ROOT))
samples = [s for s in samples if os.path.isdir(f'{ROOT}/{s}')]
w(f'検体 {len(samples)}')
M = np.zeros((len(samples), N), np.float64)
G = np.zeros((len(samples), N), np.float64)
done = []
for si, smp in enumerate(samples):
    bed = glob.glob(f'{ROOT}/{smp}/bam_to_ctss/*/bed/*[!G].collapse.ctss.bed.gz')
    bed = [b for b in bed if 'unencoded' not in b]
    gbed = glob.glob(f'{ROOT}/{smp}/bam_to_ctss/*/bed/*.unencoded_G.collapse.ctss.bed.gz')
    if not bed: continue
    scan(bed[0], M[si])
    if gbed: scan(gbed[0], G[si])
    done.append(smp)
    if (si + 1) % 30 == 0:
        w(f'  {si+1}/{len(samples)}')
keep = [i for i, s in enumerate(samples) if s in done]
M = M[keep]; G = G[keep]; samples = [samples[i] for i in keep]
w(f'スキャン完了: {len(samples)} 検体')
np.savez_compressed(f'{OUT}/raw_counts.npz', M=M, G=G,
                    samples=np.array(samples), cre=tab.cre_id.values)

rob_mask = tab.cre_id.isin(set(uni.cre_id)).values
du_mask = tab.is_dist_up.values
dp_mask = tab.is_prox_dn.values

tot = M[:, rob_mask].sum(1)
FLOOR_MIN = 20000
ok = tot >= FLOOR_MIN
floor = int(tot[ok].min())
w(f'\n[A] 頑健集合内 UMI: 中央 {np.median(tot):,.0f} / 床 {floor:,}（{FLOOR_MIN:,} 未満で除外 {int((~ok).sum())} 検体）')
sub = M[ok][:, rob_mask]
du_in = du_mask[rob_mask]
fr_reps = np.zeros((ok.sum(), 20))
for rep in range(20):
    for i in range(sub.shape[0]):
        v = sub[i]
        p = v / v.sum()
        draw = rng.multinomial(floor, p)
        fr_reps[i, rep] = draw[du_in].sum() / floor
frac = fr_reps.mean(1)
res = pd.DataFrame(dict(Sample=np.array(samples)[ok], dist_frac_eq=frac,
                        raw_frac=sub[:, du_in].sum(1) / sub.sum(1), tot_umi=tot[ok]))

ta = pd.read_csv('${DATA_DIR}/lnc_project_AIDA_dataset/by_subcluster/tage/'
                 'tage_all_combined_v1.csv')
ta = ta[(ta.model_type == 'Chronoage') & (ta['mode'] == 'EN')]
pace = ta.groupby('Sample').agg(pace=(TV, 'median'), age=('age', 'first')).reset_index()
res = res.merge(pace, on='Sample', how='inner')
res.to_csv(f'{OUT}/donor_distal_fraction_rawctss.csv', index=False)
r1 = spearmanr(res.dist_frac_eq, res.pace)
r2 = spearmanr(res.dist_frac_eq, res.age)
def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.c_[np.ones(len(x)), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return spearmanr(ex, ey)
rp = partial(res.dist_frac_eq, res.pace, res.age)
rq = partial(res.dist_frac_eq, res.pace, np.log10(res.tot_umi))
w(f'  ドナー n={len(res)}')
w(f'  等深度 遠位割合 × 加齢速度(中央値軸): rho {r1.statistic:+.3f} (p={r1.pvalue:.2g})')
w(f'    暦年齢を除いた偏相関:              rho {rp.statistic:+.3f} (p={rp.pvalue:.2g})')
w(f'    総UMIを除いた偏相関:               rho {rq.statistic:+.3f} (p={rq.pvalue:.2g})')
w(f'  参考 遠位割合 × 暦年齢:              rho {r2.statistic:+.3f}')

ta2 = ta[['Sample', 'Subcluster', TV]].rename(columns={TV: 'pace_ct'})
rows = []
for ct, g in ta2.groupby('Subcluster'):
    m = res.merge(g[['Sample', 'pace_ct']], on='Sample')
    if len(m) < 100: continue
    rr = spearmanr(m.dist_frac_eq, m.pace_ct)
    rows.append(dict(CellType=ct, n=len(m), rho=rr.statistic, p=rr.pvalue))
R = pd.DataFrame(rows).sort_values('rho', ascending=False)
R.to_csv(f'{OUT}/percelltype_axis_corr.csv', index=False)
w(f'  細胞種別軸: 正 {int((R.rho>0).sum())}/{len(R)} / 上位: ' +
  ', '.join(f'{r.CellType} {r.rho:+.2f}' for _, r in R.head(3).iterrows()))

w('\n[B] unencoded G 率（キャップ署名）')
K = 30
with np.errstate(invalid='ignore', divide='ignore'):
    def grate(mask):
        m_ = M[:, mask]; g_ = G[:, mask]
        okm = m_ >= K
        pt = np.where(okm, g_ / np.where(m_ > 0, m_, np.nan), np.nan)
        return pt, okm
    for lab, mask in [('up_distal', du_mask), ('down_proximal', dp_mask),
                      ('matched_bg', tab.cre_id.isin(bg_ids).values if bg_ids else None)]:
        if mask is None or mask.sum() == 0: continue
        pt, okm = grate(mask)
        per_t = np.nanmean(pt, axis=0)
        w(f'  {lab:14s} tCRE {int(mask.sum()):,} / UMI>={K} 充足 tCRE 中央 '
          f'{np.nanmedian(okm.sum(0)):.0f} 検体 / G 率 中央 {np.nanmedian(per_t):.3f}')
        np.save(f'{OUT}/grate_{lab}.npy', per_t)
    a = np.load(f'{OUT}/grate_up_distal.npy')
    if bg_ids:
        b = np.load(f'{OUT}/grate_matched_bg.npy')
        u = mannwhitneyu(a[np.isfinite(a)], b[np.isfinite(b)])
        w(f'  up_distal 対 マッチ背景: MWU p={u.pvalue:.3g} '
          f'(中央 {np.nanmedian(a):.3f} 対 {np.nanmedian(b):.3f})')

    gm = np.where(M[:, du_mask] >= K, G[:, du_mask] / np.maximum(M[:, du_mask], 1), np.nan)
    donor_g = np.nanmean(gm, axis=1)
    ok2 = np.isfinite(donor_g)
    m2 = pd.DataFrame(dict(Sample=samples, g=donor_g)).merge(pace, on='Sample')
    m2 = m2[np.isfinite(m2.g)]
    rg = spearmanr(m2.g, m2.pace)
    w(f'  ドナー水準 G 率 × 加齢速度: rho {rg.statistic:+.3f} (p={rg.pvalue:.2g}) — 0 近傍が望ましい')
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'\n-> {OUT}')


import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
D = '${PAPER_DIR}'
OUT = f'{D}/results/rawctss_canonical'; os.makedirs(OUT, exist_ok=True)
L = []
def w(s=''):
    print(s); L.append(s)
def partial(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    Z = np.c_[np.ones(len(x)), np.column_stack([rankdata(z) for z in Z])]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return spearmanr(ex, ey)
old = pd.read_csv(f'{D}/results/raw_ctss_scan/donor_distal_fraction_rawctss.csv')
pc = pd.read_csv(f'{D}/results/tage_canonical/pace_aida_canonical.csv')
at = pd.read_csv(f'{D}/results/donor_attributes/donor_attributes.csv').query("cohort == 'AIDA'")
don = pc.groupby('donor').agg(pace=('pace', 'median'), age=('age', 'first')).reset_index()
res = old[['Sample', 'dist_frac_eq', 'raw_frac', 'tot_umi']].merge(don, left_on='Sample', right_on='donor').merge(
    at[['donor', 'country']], on='donor')
res.to_csv(f'{OUT}/donor_table.csv', index=False)
jp = (res.country == 'JP').astype(float).values
r1 = spearmanr(res.dist_frac_eq, res.pace); r2 = spearmanr(res.dist_frac_eq, res.age)
ra = partial(res.dist_frac_eq, res.pace, [res.age]); ru = partial(res.dist_frac_eq, res.pace, [np.log10(res.tot_umi)])
rc = partial(res.dist_frac_eq, res.pace, [jp])
w(f'ドナー n={len(res)}（正典 pace = pace_aida_canonical の細胞種中央値）')
w(f'  遠位割合 × pace            rho {r1.statistic:+.3f} (p={r1.pvalue:.2g})   [旧 +0.711]')
w(f'    暦年齢を除く             rho {ra.statistic:+.3f} (p={ra.pvalue:.2g})   [旧 +0.720]')
w(f'    総 UMI を除く            rho {ru.statistic:+.3f} (p={ru.pvalue:.2g})   [旧 +0.713]')
w(f'    国を除く                 rho {rc.statistic:+.3f} (p={rc.pvalue:.2g})   [旧 +0.251]')
for c in ('JP', 'KR'):
    g = res[res.country == c]; r = spearmanr(g.dist_frac_eq, g.pace)
    w(f'    {c} 内                    rho {r.statistic:+.3f} (p={r.pvalue:.2g}, n={len(g)})')
w(f'  遠位割合 × 暦年齢          rho {r2.statistic:+.3f}   [旧 +0.228]')
w(f'  pace × 暦年齢（時計の傾き診断）rho {spearmanr(res.pace, res.age).statistic:+.3f}   [本文 +0.582 と比較]')
rows = []
for ct, g in pc.groupby('subset'):
    m = res[['Sample', 'dist_frac_eq', 'country']].merge(g[['donor', 'pace']].rename(columns={'donor': 'Sample', 'pace': 'pace_ct'}), on='Sample')
    if len(m) < 100: continue
    r = spearmanr(m.dist_frac_eq, m.pace_ct); rcc = partial(m.dist_frac_eq, m.pace_ct, [(m.country == 'JP').astype(float)])
    rows.append(dict(CellType=ct, n=len(m), rho=r.statistic, p=r.pvalue, rho_country_partial=rcc.statistic))
R = pd.DataFrame(rows).sort_values('rho', ascending=False); R.to_csv(f'{OUT}/percelltype_axis_corr.csv', index=False)
w(f'  細胞種別軸: 正 {int((R.rho > 0).sum())}/{len(R)}、国を除いても正 {int((R.rho_country_partial > 0).sum())}/{len(R)}   [旧 11/11, 11/11]')
w('    ' + ', '.join(f'{r.CellType} {r.rho:+.2f}' for r in R.itertuples()))
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))


import gzip, os, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, binomtest, mannwhitneyu

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUT = f'{D}/results/shape_scan/v2'
os.makedirs(OUT, exist_ok=True)
ROOT = '${DATA_DIR}/lnc_project_AIDA_dataset/aida_per_sample_bed'
TV = 'yugene_diff_EN_tAge'
K = 50
rng = np.random.default_rng(0)
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

rob = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv',
                  usecols=['cre_id', 'cell_type', 'program'])
jo = pd.read_csv(f'{D}/results/gene_to_cre/cre_gene_joined.csv',
                 usecols=['cre_id', 'cell_type', 'mean_expression', 'proximity',
                          'enhancer_class', 'robust', 'near_gene'])
rob = rob.merge(jo, on=['cre_id', 'cell_type'], how='left').dropna(subset=['mean_expression'])
w(f'頑健行 {len(rob):,}（発現量が引けたもの）')

pairs = []
for (ct, prox), fg in rob.groupby(['cell_type', 'proximity']):
    pool = jo[(jo.cell_type == ct) & (jo.proximity == prox) & (~jo.robust)
              & (~jo.cre_id.isin(set(rob.cre_id)))]
    if not len(pool): continue
    pv = pool.mean_expression.values
    order = np.argsort(pv); pv_s = pv[order]; ids = pool.cre_id.values[order]
    used = set()
    for _, f in fg.iterrows():
        i = int(np.searchsorted(pv_s, f.mean_expression))
        for off in range(0, 400):
            for j in (i + off, i - off):
                if 0 <= j < len(ids) and ids[j] not in used:
                    used.add(ids[j])
                    pairs.append(dict(cre_id=ids[j], cell_type=ct, proximity=prox,
                                      program=f.program, role='background',
                                      mean_expression=pv_s[j]))
                    off = 999; break
            if off == 999: break
bg = pd.DataFrame(pairs)
fgm = rob.assign(role='robust')[['cre_id', 'cell_type', 'proximity', 'program', 'role', 'mean_expression']]
tgt = pd.concat([fgm, bg], ignore_index=True)
w(f'背景マッチ {len(bg):,} 行 / 発現量 中央 前景 {fgm.mean_expression.median():.1f} 対 背景 {bg.mean_expression.median():.1f}')
uni = tgt.drop_duplicates('cre_id').copy()
co = uni.cre_id.str.rsplit('_', n=3, expand=True)
uni['chrom'] = co[0]; uni['start'] = co[1].astype(int)
uni['end'] = co[2].astype(int); uni['strand'] = co[3]
STA = dict(zip(uni.cre_id, uni.start)); END = dict(zip(uni.cre_id, uni.end))
w(f'スキャン対象 異なり tCRE {len(uni):,}')
uni.to_csv(f'{OUT}/target_set.csv', index=False)

by_ch = {}
for ch, g in uni.groupby('chrom'):
    g = g.sort_values('start').reset_index(drop=True)
    by_ch[ch] = dict(starts=g.start.values, ends=g.end.values,
                     strands=g.strand.values, gids=g.cre_id.values)

cand = pd.read_csv(f'{D}/results/paper_additions/candidate_redistribution_loci.csv')
WINDOWS = {}
for _, r in cand.sort_values('dist').iterrows():
    name = f'{r.cell_type}_{r.up_gene}_{r.dn_gene}'
    if name in WINDOWS: continue
    pad = 3000
    WINDOWS[name] = (r.chrom, int(r.wstart) - pad, int(r.wend) + pad)
    if len(WINDOWS) >= 6: break
for k, v in WINDOWS.items():
    w(f'  窓 {k}: {v[0]}:{v[1]:,}-{v[2]:,} ({v[2]-v[1]:,} bp)')

samples = sorted(d for d in os.listdir(ROOT) if os.path.isdir(f'{ROOT}/{d}'))
w(f'検体 {len(samples)}')
win_prof = {k: np.zeros((len(samples), v[2] - v[1], 2), np.float32) for k, v in WINDOWS.items()}
shape_rows, bal_rows = [], []
for si, smp in enumerate(samples):
    beds = [b for b in glob.glob(f'{ROOT}/{smp}/bam_to_ctss/*/bed/*.collapse.ctss.bed.gz')
            if 'unencoded' not in b]
    if not beds: continue
    prof, bal = {}, {}
    with gzip.open(beds[0], 'rt') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            ch = p[0]; pos = int(p[1]); cnt = float(p[4]); strand = p[5]
            for wname, (wch, ws, we) in WINDOWS.items():
                if ch == wch and ws <= pos < we:
                    win_prof[wname][si, pos - ws, 0 if strand == '+' else 1] += cnt
            d = by_ch.get(ch)
            if d is None: continue
            i = np.searchsorted(d['starts'], pos, side='right') - 1
            for j in (i, i - 1, i - 2):
                if 0 <= j < len(d['starts']) and d['starts'][j] <= pos < d['ends'][j]:
                    cid = d['gids'][j]
                    b_ = bal.setdefault(cid, [0.0, 0.0])
                    b_[0 if strand == '+' else 1] += cnt
                    st = d['strands'][j]
                    if st in ('+', '-') and st != strand:
                        continue
                    a = prof.get(cid)
                    if a is None:
                        a = prof[cid] = np.zeros(END[cid] - STA[cid], np.float64)
                    a[pos - STA[cid]] += cnt
    for cid, a in prof.items():
        tot = a.sum()
        if tot < K: continue
        draw = rng.multinomial(K, a / tot)
        q = draw[draw > 0] / K
        cs = np.cumsum(draw) / K
        shape_rows.append((smp, cid, tot, float(-(q * np.log2(q)).sum() / np.log2(K)),
                           int(np.searchsorted(cs, 0.9) - np.searchsorted(cs, 0.1)),
                           float(draw.max()) / K))
    for cid, (pl, mi) in bal.items():
        n = pl + mi
        if n < K: continue
        fp = pl / n
        bal_rows.append((smp, cid, n, fp, 1 - abs(2 * fp - 1)))
    if (si + 1) % 40 == 0: w(f'  {si+1}/{len(samples)}')

sh = pd.DataFrame(shape_rows, columns=['Sample', 'cre_id', 'total', 'entropy', 'span1090', 'maxfrac'])
ba = pd.DataFrame(bal_rows, columns=['Sample', 'cre_id', 'total', 'frac_plus', 'balance'])
sh.to_parquet(f'{OUT}/shape_per_donor_tcre.parquet')
ba.to_parquet(f'{OUT}/balance_per_donor_tcre.parquet')
np.savez_compressed(f'{OUT}/locus_profiles.npz', samples=np.array(samples),
                    **{f'{k}__prof': v for k, v in win_prof.items()})
pd.DataFrame([(k,) + v for k, v in WINDOWS.items()],
             columns=['name', 'chrom', 'start', 'end']).to_csv(f'{OUT}/locus_windows.csv', index=False)
w(f'\n形状 {len(sh):,} 行 / バランス {len(ba):,} 行')

ta = pd.read_csv('${DATA_DIR}/lnc_project_AIDA_dataset/by_subcluster/tage/'
                 'tage_all_combined_v1.csv')
ta = ta[(ta.model_type == 'Chronoage') & (ta['mode'] == 'EN')]
pace = ta.groupby('Sample').agg(pace=(TV, 'median'), age=('age', 'first')).reset_index()
meta = uni.set_index('cre_id')

w('\n[主解析 tCRE 単位] 各素子の中で「峰のにじみ」が老化速度と動くか')
per = []
for cid, g in sh.groupby('cre_id'):
    m = g.merge(pace, on='Sample')
    if len(m) < 100: continue
    r = spearmanr(m.entropy, m.pace); rs = spearmanr(m.span1090, m.pace)
    rd = spearmanr(m.total, m.pace)
    mm = meta.loc[cid]
    per.append(dict(cre_id=cid, n_donor=len(m), role=mm.role, program=mm.program,
                    proximity=mm.proximity, rho_ent=r.statistic, p_ent=r.pvalue,
                    rho_span=rs.statistic, rho_depth=rd.statistic))
pt = pd.DataFrame(per)
jo2 = jo.drop_duplicates('cre_id')[['cre_id', 'near_gene']]
pt = pt.merge(jo2, on='cre_id', how='left')
pt.to_csv(f'{OUT}/per_tcre_blur.csv', index=False)
w(f'  検定できた素子 {len(pt):,}（ドナー 100 名以上で K={K} を満たす）')
for (role, prog), g in pt.groupby(['role', 'program']):
    if len(g) < 20: continue
    pos = int((g.rho_ent > 0).sum())
    bt = binomtest(pos, len(g), 0.5)
    w(f'  {role:10s} {prog:14s} n={len(g):4d}  にじみ正 {pos:4d} ({pos/len(g)*100:4.0f}%) '
      f'符号検定 p={bt.pvalue:.2g}  rho 中央 {g.rho_ent.median():+.3f}  '
      f'（深度 rho 中央 {g.rho_depth.median():+.3f}）')
w('\n  ★前景と背景で「にじみ正の割合」が同じなら、にじみは老化でなく選抜または全体傾向。')
for prog in pt.program.dropna().unique():
    a = pt[(pt.role == 'robust') & (pt.program == prog)]
    b = pt[(pt.role == 'background') & (pt.program == prog)]
    if len(a) < 20 or len(b) < 20: continue
    u = mannwhitneyu(a.rho_ent, b.rho_ent)
    w(f'  {prog}: 頑健 rho 中央 {a.rho_ent.median():+.3f} 対 背景 {b.rho_ent.median():+.3f} (MWU p={u.pvalue:.2g})')
top = pt[pt.role == 'robust'].nlargest(12, 'rho_ent')
w('  にじみ上位（頑健）: ' + ', '.join(f'{r.near_gene}({r.rho_ent:+.2f})'
                                       for _, r in top.iterrows() if pd.notna(r.near_gene)))

sh2 = sh.merge(meta[['role', 'program', 'proximity']], left_on='cre_id', right_index=True)
w('\n[従 ドナー平均] 素子構成が深度に依存するため参考値')
for (role, prog), g in sh2.groupby(['role', 'program']):
    d = g.groupby('Sample').agg(ent=('entropy', 'mean'), n=('cre_id', 'size')).reset_index()
    d = d[d.n >= 20].merge(pace, on='Sample')
    if len(d) < 100: continue
    r = spearmanr(d.ent, d.pace)
    w(f'  {role:10s} {prog:14s} n={len(d)} ドナー: rho {r.statistic:+.3f} (p={r.pvalue:.2g})')

w('\n[双方向バランス] enhancer_class 層別（unidirectional と divergent は期待値が逆）')
ecl = jo.drop_duplicates('cre_id').set_index('cre_id').enhancer_class
ba2 = ba.assign(role=ba.cre_id.map(meta.role), program=ba.cre_id.map(meta.program),
                eclass=ba.cre_id.map(ecl))
for (ec, role), g in ba2.groupby(['eclass', 'role']):
    if g.cre_id.nunique() < 30: continue
    rows = []
    for cid, gg in g.groupby('cre_id'):
        m = gg.merge(pace, on='Sample')
        if len(m) < 100: continue
        rows.append(spearmanr(m.balance, m.pace).statistic)
    if len(rows) < 20: continue
    pos = int(np.sum(np.array(rows) > 0))
    w(f'  {ec:28s} {role:10s} 素子 {len(rows):4d}  バランス正 {pos/len(rows)*100:4.0f}%  '
      f'rho 中央 {np.median(rows):+.3f}')
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')
