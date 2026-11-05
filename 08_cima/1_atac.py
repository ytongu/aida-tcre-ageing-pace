#!/usr/bin/env python3
import os, gzip
import numpy as np, pandas as pd
from scipy.stats import fisher_exact

D = '${PAPER_DIR}'
W = '${DATA_DIR}/cima'
OUT = f'{D}/results/cima_crosswalk'
GTF = '${DATA_DIR}/adachi/reference/GRCh38.113.gtf.gz'
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

pk = pd.read_csv(f'{W}/atac/CIMA_ATAC_CellType_Peakset.csv', usecols=[0])
pk.columns = ['Peaks']
co = pk.Peaks.str.extract(r'^(chr[^:]+):(\d+)-(\d+)$')
pk['chrom'] = co[0]; pk['start'] = co[1].astype(int); pk['end'] = co[2].astype(int)
pk = pk.dropna(subset=['chrom'])
w(f'CIMA ピーク {len(pk):,} / 幅 中央 {int((pk.end-pk.start).median())} bp（501 固定のはず）')

tss = {}
with gzip.open(GTF, 'rt') as f:
    for line in f:
        if line[0] == '#': continue
        p = line.split('\t')
        if p[2] != 'gene': continue
        c = 'chr' + p[0] if not p[0].startswith('chr') else p[0]
        t = int(p[3]) if p[6] == '+' else int(p[4])
        tss.setdefault(c, []).append(t)
TS = {c: np.array(sorted(v)) for c, v in tss.items()}
def dist_to_tss(c, mid):
    a = TS.get(c)
    if a is None or not len(a): return 10**9
    i = np.searchsorted(a, mid)
    d = 10**9
    for j in (i - 1, i):
        if 0 <= j < len(a): d = min(d, abs(int(a[j]) - mid))
    return d
pk['mid'] = (pk.start + pk.end) // 2
pk['tss_dist'] = [dist_to_tss(c, m) for c, m in zip(pk.chrom, pk.mid)]
pk['pclass'] = np.where(pk.tss_dist <= 1000, 'promoter-TSS', 'distal')
frac = (pk.pclass == 'promoter-TSS').mean() * 100
w(f'自前アノテーション: promoter-TSS {frac:.1f}%（原著 6.9%。±2 点なら整合とみなす）')
pk[['chrom', 'start', 'end', 'Peaks', 'pclass', 'tss_dist']].to_csv(
    f'{OUT}/cima_peaks_annotated.bed', sep='\t', header=False, index=False)

S = pd.read_csv(f'{OUT}/frozen_set_summits.csv')
by = {}
for c, g in pk.groupby('chrom'):
    g = g.sort_values('start')
    by[c] = dict(s=g.start.values, e=g.end.values, p=g.Peaks.values, cl=g.pclass.values)
hit_peak, hit_class = [], []
for c, sm in zip(S.chrom, S.summit):
    d = by.get(c)
    if d is None: hit_peak.append(None); hit_class.append(None); continue
    i = np.searchsorted(d['s'], sm, side='right') - 1
    got = None; cl = None
    for j in (i, i - 1):
        if 0 <= j < len(d['s']) and d['s'][j] <= sm < d['e'][j]:
            got = d['p'][j]; cl = d['cl'][j]; break
    hit_peak.append(got); hit_class.append(cl)
S['cima_peak'] = hit_peak; S['cima_class'] = hit_class
S.to_csv(f'{OUT}/frozen_x_cima.csv', index=False)
w('\n[ヒット率]')
for prog, lab in [('up_distal', '上昇遠位'), ('down_proximal', '低下近位')]:
    g = S[S.program == prog]
    n = g.cima_peak.notna().sum()
    w(f'  {lab} {n:,}/{len(g):,} ({n/len(g)*100:.1f}%)  '
      f'うち CIMA で promoter-TSS {int((g.cima_class=="promoter-TSS").sum())}')
a = S[S.program == 'up_distal']; b = S[S.program == 'down_proximal']
orr, p = fisher_exact([[a.cima_peak.notna().sum(), a.cima_peak.isna().sum()],
                       [b.cima_peak.notna().sum(), b.cima_peak.isna().sum()]])
w(f'  ヒット率の差: OR {orr:.2f} (p={p:.3g})')
w(f'\n[ゲート] 各アーム 100 以上 → 個別エレメント検定へ / 50 未満 → セットスコアのみ')
for prog, lab in [('up_distal', '上昇遠位'), ('down_proximal', '低下近位')]:
    n = S[(S.program == prog)].cima_peak.notna().sum()
    w(f'  {lab}: {n} → {"個別検定可" if n >= 100 else "セットスコアのみ" if n < 50 else "個別検定可(境界)"}')
open(f'{OUT}/report_part5.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')


import os, gzip, glob
import numpy as np, pandas as pd

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUT = f'{D}/results/cima_crosswalk'
os.makedirs(OUT, exist_ok=True)
ROOT = '${DATA_DIR}/lnc_project_AIDA_dataset/aida_per_sample_bed'
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

rob = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv')
rob = rob[rob.program.isin(['up_distal', 'down_proximal'])]
uni = rob.drop_duplicates('cre_id').copy()
co = uni.cre_id.str.rsplit('_', n=3, expand=True)
uni['chrom'] = co[0]; uni['start'] = co[1].astype(int)
uni['end'] = co[2].astype(int); uni['strand'] = co[3]
uni['width'] = uni.end - uni.start
w(f'頑健 tCRE {len(uni):,}（' + ', '.join(f'{k} {v}' for k, v in uni.program.value_counts().items()) + '）')
w(f'  幅: 中央 {uni.width.median():.0f} bp / 範囲 {uni.width.min()}〜{uni.width.max()}')

by_ch = {}
for ch, g in uni.groupby('chrom'):
    g = g.sort_values('start').reset_index(drop=True)
    by_ch[ch] = dict(s=g.start.values, e=g.end.values, st=g.strand.values, ids=g.cre_id.values)
STA = dict(zip(uni.cre_id, uni.start)); END = dict(zip(uni.cre_id, uni.end))
prof = {c: np.zeros(END[c] - STA[c], np.int64) for c in uni.cre_id}

samples = sorted(d for d in os.listdir(ROOT) if os.path.isdir(f'{ROOT}/{d}'))
w(f'AIDA {len(samples)} 検体を走査（サミット決定のため全検体合算）')
for si, smp in enumerate(samples):
    beds = [b for b in glob.glob(f'{ROOT}/{smp}/bam_to_ctss/*/bed/*.collapse.ctss.bed.gz')
            if 'unencoded' not in b]
    if not beds: continue
    with gzip.open(beds[0], 'rt') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            d = by_ch.get(p[0])
            if d is None: continue
            pos = int(p[1])
            i = np.searchsorted(d['s'], pos, side='right') - 1
            for j in (i, i - 1, i - 2):
                if 0 <= j < len(d['s']) and d['s'][j] <= pos < d['e'][j]:
                    st = d['st'][j]
                    if st in ('+', '-') and st != p[5]: continue
                    prof[d['ids'][j]][pos - d['s'][j]] += int(float(p[4]))
    if (si + 1) % 40 == 0: w(f'  {si+1}/{len(samples)}')

rows = []
for _, r in uni.iterrows():
    a = prof[r.cre_id]; tot = a.sum()
    summit = int(r.start + (np.argmax(a) if tot else (r.end - r.start) // 2))
    rows.append(dict(cre_id=r.cre_id, chrom=r.chrom, start=r.start, end=r.end, strand=r.strand,
                     width=r.width, summit=summit, total_umi=int(tot),
                     summit_frac=float(a.max() / tot) if tot else np.nan,
                     program=r.program, cell_type=r.cell_type,
                     mean_expression=r.mean_expression, direction=r.direction))
S = pd.DataFrame(rows)
S.to_csv(f'{OUT}/frozen_set_summits.csv', index=False)
w(f'\nサミット決定 {len(S):,}（UMI 0 で中点にした tCRE {int((S.total_umi==0).sum())}）')
w(f'  サミット位置の占有率 中央 {S.summit_frac.median():.3f}')
for prog, fn in [('up_distal', 'rising_distal'), ('down_proximal', 'falling_proximal')]:
    g = S[S.program == prog]
    with open(f'{OUT}/{fn}.bed', 'w') as f:
        for _, r in g.iterrows():
            f.write(f'{r.chrom}\t{r.summit}\t{r.summit+1}\t{r.cre_id}|{r.cell_type}|'
                    f'{r.width}|{r.mean_expression:.4f}\t{r.total_umi}\t{r.strand}\n')
    w(f'  {fn}.bed: {len(g):,} 行')
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')


import os, gzip, glob
import numpy as np, pandas as pd
from scipy.stats import binomtest, fisher_exact

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUT = f'{D}/results/cima_crosswalk'
ROOT = '${DATA_DIR}/lnc_project_AIDA_dataset/aida_per_sample_bed'
NDEC = 10
PER_CELL = 1200
rng = np.random.default_rng(0)
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

rob = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv')
rob = rob[rob.program.isin(['up_distal', 'down_proximal'])]
uni = rob.drop_duplicates('cre_id').copy()
ROBIDS = set(uni.cre_id)
jo = pd.read_csv(f'{D}/results/gene_to_cre/cre_gene_joined.csv',
                 usecols=['cre_id', 'cell_type', 'mean_expression', 'proximity', 'robust'])
jo1 = jo.drop_duplicates('cre_id')
FG = uni.merge(jo1[['cre_id', 'proximity']], on='cre_id', how='left')
w(f'前景（頑健）{len(FG):,}: ' + ', '.join(f'{k} {v}' for k, v in FG.program.value_counts().items()))

pool = jo1[(~jo1.robust) & (~jo1.cre_id.isin(ROBIDS))].copy()
w(f'背景候補プール {len(pool):,}（distal {int((pool.proximity=="distal").sum()):,} / '
  f'proximal {int((pool.proximity=="proximal").sum()):,}）')

pool['lx'] = np.log10(pool.mean_expression + 0.1)
FG['lx'] = np.log10(FG.mean_expression + 0.1)
EDGES = {}
samp = []
for px, g in pool.groupby('proximity'):
    q = np.quantile(g.lx, np.linspace(0, 1, NDEC + 1)); q[0] = -np.inf; q[-1] = np.inf
    EDGES[px] = q
    g = g.assign(dec=np.clip(np.searchsorted(q, g.lx, side='right') - 1, 0, NDEC - 1))
    for dec, gg in g.groupby('dec'):
        take = gg if len(gg) <= PER_CELL else gg.sample(PER_CELL, random_state=int(dec))
        samp.append(take.assign(dec=dec))
S = pd.concat(samp, ignore_index=True)
w(f'層化サンプル {len(S):,}（proximity × 十分位 {NDEC}、各セル上限 {PER_CELL}）')
for px in EDGES:
    w(f'  {px}: ' + ', '.join(str(int(v)) for v in S[S.proximity == px].dec.value_counts().sort_index()))

need = S[['cre_id', 'proximity', 'mean_expression', 'lx', 'dec']].copy()
co = need.cre_id.str.rsplit('_', n=3, expand=True)
need['chrom'] = co[0]; need['start'] = co[1].astype(int)
need['end'] = co[2].astype(int); need['strand'] = co[3]
starts = need.start.values; ends = need.end.values
chroms = need.chrom.values; strands = need.strand.values
w(f'\nサミットを決める背景 {len(need):,} / 総塩基 {int((ends-starts).sum())/1e6:.0f} Mb')

BINW = 1000
bins = {}
for k in range(len(need)):
    for b in range(starts[k] // BINW, ends[k] // BINW + 1):
        bins.setdefault((chroms[k], b), []).append(k)
prof = [np.zeros(ends[k] - starts[k], np.int32) for k in range(len(need))]

samples = sorted(d for d in os.listdir(ROOT) if os.path.isdir(f'{ROOT}/{d}'))
w(f'AIDA {len(samples)} 検体を走査')
for si, smp in enumerate(samples):
    beds = [b for b in glob.glob(f'{ROOT}/{smp}/bam_to_ctss/*/bed/*.collapse.ctss.bed.gz')
            if 'unencoded' not in b]
    if not beds: continue
    with gzip.open(beds[0], 'rt') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            lst = bins.get((p[0], int(p[1]) // BINW))
            if not lst: continue
            pos = int(p[1]); val = int(float(p[4])); st6 = p[5]
            for k in lst:
                if starts[k] <= pos < ends[k]:
                    if strands[k] in ('+', '-') and strands[k] != st6: continue
                    prof[k][pos - starts[k]] += val
    if (si + 1) % 40 == 0: w(f'  {si+1}/{len(samples)}')

need['summit'] = [starts[k] + (int(np.argmax(prof[k])) if prof[k].sum()
                               else (ends[k] - starts[k]) // 2) for k in range(len(need))]
need['total_umi'] = [int(a.sum()) for a in prof]
del prof
need.to_csv(f'{OUT}/background_summits_v2.csv', index=False)
w(f'  UMI 0 で中点にした背景 {int((need.total_umi==0).sum()):,}')

pk = pd.read_csv(f'{OUT}/cima_peaks_annotated.bed', sep='\t', header=None,
                 names=['chrom', 'start', 'end', 'Peaks', 'pclass', 'tss_dist'])
by = {}
for c, g in pk.groupby('chrom'):
    g = g.sort_values('start')
    by[c] = dict(s=g.start.values, e=g.end.values)
def hit(chrom, sm):
    d = by.get(chrom)
    if d is None: return False
    i = np.searchsorted(d['s'], sm, side='right') - 1
    for j in (i, i - 1):
        if 0 <= j < len(d['s']) and d['s'][j] <= sm < d['e'][j]: return True
    return False
need['cima_hit'] = [hit(c, s) for c, s in zip(need.chrom, need.summit)]
need.to_csv(f'{OUT}/background_x_cima_v2.csv', index=False)

w('\n[背景における 発現量 → CIMA ヒット率]')
curve = {}
for px, g in need.groupby('proximity'):
    w(f'  {px}:')
    d = {}
    for dec, gg in g.groupby('dec'):
        r = gg.cima_hit.mean()
        d[dec] = r
        w(f'    十分位 {int(dec)}  発現中央 {gg.mean_expression.median():8.3f}  '
          f'n={len(gg):5d}  ヒット {r*100:5.1f}%')
    curve[px] = d

FXC = pd.read_csv(f'{OUT}/frozen_x_cima.csv')
FXC = FXC.merge(jo1[['cre_id', 'proximity']], on='cre_id', how='left')
FXC['lx'] = np.log10(FXC.mean_expression + 0.1)
FXC['dec'] = [int(np.clip(np.searchsorted(EDGES[px], v, side='right') - 1, 0, NDEC - 1))
              if isinstance(px, str) else -1 for px, v in zip(FXC.proximity, FXC.lx)]
w('\n[前景の実測 対 発現量から期待される値]')
rows = []
for prog, lab in [('up_distal', '上昇遠位'), ('down_proximal', '低下近位')]:
    g = FXC[(FXC.program == prog) & (FXC.dec >= 0)]
    obs = int(g.cima_peak.notna().sum()); n = len(g)
    exp = float(sum(curve[px].get(dc, np.nan) for px, dc in zip(g.proximity, g.dec)))
    pexp = exp / n
    bt = binomtest(obs, n, pexp)
    orr, pf = fisher_exact([[obs, n - obs], [int(round(exp)), n - int(round(exp))]])
    w(f'  {lab}  実測 {obs}/{n} ({obs/n*100:.1f}%)  期待 {exp:.0f} ({pexp*100:.1f}%)  '
      f'比 {obs/exp:.2f}  二項 p={bt.pvalue:.3g}  OR {orr:.2f}')
    rows.append(dict(program=prog, observed=obs, n=n, expected=exp, exp_rate=pexp,
                     obs_rate=obs / n, ratio=obs / exp, p_binom=bt.pvalue, OR=orr))

    for dec, gg in g.groupby('dec'):
        e = curve[gg.proximity.iloc[0]].get(dec, np.nan)
        w(f'      十分位 {int(dec)} n={len(gg):4d}  実測 {gg.cima_peak.notna().mean()*100:5.1f}%  '
          f'期待 {e*100:5.1f}%')
pd.DataFrame(rows).to_csv(f'{OUT}/expected_vs_observed.csv', index=False)
open(f'{OUT}/report_background_v2.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')


import numpy as np, pandas as pd

D = '${PAPER_DIR}'
OUT = f'{D}/results/cima_crosswalk'
NDEC = 10
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

bg = pd.read_csv(f'{OUT}/background_x_cima_v2.csv')
fg = pd.read_csv(f'{OUT}/frozen_x_cima.csv')
jo = pd.read_csv(f'{D}/results/gene_to_cre/cre_gene_joined.csv',
                 usecols=['cre_id', 'mean_expression', 'proximity', 'robust']).drop_duplicates('cre_id')
fg = fg.merge(jo[['cre_id', 'proximity']], on='cre_id', how='left')

pool = jo[~jo.robust].copy()
pool['lx'] = np.log10(pool.mean_expression + 0.1)
POOLN = {}
for px, g in pool.groupby('proximity'):
    q = np.quantile(g.lx, np.linspace(0, 1, NDEC + 1)); q[0] = -np.inf; q[-1] = np.inf
    dec = np.clip(np.searchsorted(q, g.lx, side='right') - 1, 0, NDEC - 1)
    POOLN[px] = pd.Series(dec).value_counts().to_dict()

w('CIMA の 338,036 ピークに当たった割合')
w('（我々の tCRE のサミット 1 塩基が、CIMA の 501bp ピークの中に入るかどうか）')
w('')
rows = []
for prog, px, lab in [('up_distal', 'distal', '上昇遠位'),
                      ('down_proximal', 'proximal', '低下近位')]:
    g = fg[fg.program == prog]
    fr = g.cima_peak.notna().mean() * 100

    b = bg[bg.proximity == px].copy()
    cnt = b.dec.value_counts().to_dict()
    b['wt'] = [POOLN[px].get(d, 0) / cnt.get(d, 1) for d in b.dec]
    br = float((b.cima_hit * b.wt).sum() / b.wt.sum() * 100)

    w(f'  {lab}（我々の頑健セット）      {g.cima_peak.notna().sum():5d}/{len(g):5d}  {fr:5.1f}%'
      f'    発現量 中央 {g.mean_expression.median():8.3f}')
    w(f'  {px} tCRE 一般              {"":11s}  {br:5.1f}%'
      f'    発現量 中央 {pool[pool.proximity==px].mean_expression.median():8.3f}')
    w('')
    rows.append(dict(program=prog, proximity=px, robust_rate=fr, general_rate=br,
                     robust_n=len(g), robust_hit=int(g.cima_peak.notna().sum()),
                     bg_n=len(b),
                     robust_expr_med=float(g.mean_expression.median()),
                     bg_expr_med=float(pool[pool.proximity == px].mean_expression.median())))
pd.DataFrame(rows).to_csv(f'{OUT}/plain_compare.csv', index=False)
open(f'{OUT}/report_plain.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}/report_plain.txt')


import os, time
import h5py, numpy as np, pandas as pd
from scipy.sparse import csr_matrix, coo_matrix

D = '${PAPER_DIR}'
W = '${DATA_DIR}/cima'
OUT = f'{D}/results/cima_part3'
os.makedirs(OUT, exist_ok=True)
F = f'{W}/atac/CIMA_ATAC_3762242cells_338036peaks_compressed.h5ad'
CHUNK = 50_000
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

MAP = dict(pd.read_csv(f'{W}/meta/celltype_map.csv').values)

with h5py.File(F, 'r') as h:
    def cat(n):
        g = h[f'obs/{n}']
        return np.array(g['categories']).astype(str)[np.array(g['codes'])]
    smp = cat('sample'); l4 = cat('cell_type_l4')
    age = np.array(h['obs/age']); sex = cat('sex'); lib = cat('library')
    peaks = np.array(h['var/index']).astype(str)
    ncell = len(smp); npk = len(peaks)
    indptr = np.array(h['X/indptr'])
w(f'細胞 {ncell:,} / ピーク {npk:,} / 非ゼロ {indptr[-1]:,}')

sub = np.array([MAP.get(x, '') for x in l4])
keep = sub != ''
w(f'我々の 11 subset に対応する細胞 {keep.sum():,} ({keep.mean()*100:.1f}%)')

key = pd.Series([f'{s}||{c}' for s, c in zip(smp, sub)])
key[~keep] = ''
gnames = sorted(set(key[keep]))
gidx = {g: i for i, g in enumerate(gnames)}
grp = np.array([gidx.get(k, -1) for k in key], np.int32)
ng = len(gnames)
w(f'グループ（ドナー × subset）{ng:,}')

PB = np.zeros((ng, npk), np.float32)
gcells = np.zeros(ng, np.int64)
t0 = time.time()
with h5py.File(F, 'r') as h:
    dX = h['X/data']; iX = h['X/indices']
    for c0 in range(0, ncell, CHUNK):
        c1 = min(c0 + CHUNK, ncell)
        p0, p1 = int(indptr[c0]), int(indptr[c1])
        g = grp[c0:c1]
        ok = g >= 0
        if not ok.any(): continue
        dat = np.asarray(dX[p0:p1], np.float32)
        ind = np.asarray(iX[p0:p1], np.int64)
        ip = indptr[c0:c1 + 1] - p0
        Xc = csr_matrix((dat, ind, ip), shape=(c1 - c0, npk))

        rows = g[ok]; cols = np.nonzero(ok)[0]
        G = coo_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                       shape=(ng, c1 - c0)).tocsr()
        PB += np.asarray((G @ Xc).todense(), np.float32)
        np.add.at(gcells, rows, 1)
        if (c0 // CHUNK) % 10 == 0:
            el = time.time() - t0
            w(f'  {c1:,}/{ncell:,}  経過 {el/60:.1f} 分  推定残 {el/max(c1,1)*(ncell-c1)/60:.1f} 分')
w(f'畳み込み完了 {(time.time()-t0)/60:.1f} 分')

np.save(f'{OUT}/pseudobulk.npy', PB)
gm = pd.DataFrame({'group': gnames})
gm['donor'] = [g.split('||')[0] for g in gnames]
gm['subset'] = [g.split('||')[1] for g in gnames]
gm['n_cells'] = gcells
gm['total_frag'] = PB.sum(1)
dmeta = pd.DataFrame(dict(donor=smp, age=age, sex=sex)).drop_duplicates('donor')
gm = gm.merge(dmeta, on='donor', how='left')
gm.to_csv(f'{OUT}/groups.csv', index=False)
w(f'\nグループ {len(gm):,} / ドナー {gm.donor.nunique()} / subset {gm["subset"].nunique()}')
w('  subset ごとのグループ数と細胞数中央:')
for s, g in gm.groupby('subset'):
    w(f'    {s:14s} n_group={len(g):4d}  細胞中央 {int(g.n_cells.median()):6d}  '
      f'断片中央 {int(g.total_frag.median()):9d}')

pk = pd.DataFrame({'peak': peaks})
co = pk.peak.str.extract(r'^(chr[^:]+):(\d+)-(\d+)$')
pk['chrom'] = co[0]; pk['start'] = co[1].astype(float); pk['end'] = co[2].astype(float)
ann = pd.read_csv(f'{D}/results/cima_crosswalk/cima_peaks_annotated.bed', sep='\t',
                  header=None, names=['chrom', 'start', 'end', 'peak', 'pclass', 'tss_dist'])
pk = pk.merge(ann[['peak', 'pclass', 'tss_dist']], on='peak', how='left')
pk.to_csv(f'{OUT}/peaks.csv', index=False)
w(f'\nピーク注釈: promoter-TSS {int((pk.pclass=="promoter-TSS").sum()):,} / '
  f'distal {int((pk.pclass=="distal").sum()):,} / 未注釈 {int(pk.pclass.isna().sum()):,}')
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')


import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

D = '${PAPER_DIR}'
P3 = f'{D}/results/cima_part3'
OUT = f'{D}/results/cima_part4a'
os.makedirs(OUT, exist_ok=True)
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

PB = np.load(f'{P3}/pseudobulk.npy')
gm = pd.read_csv(f'{P3}/groups.csv')
pk = pd.read_csv(f'{P3}/peaks.csv')
w(f'擬似バルク {PB.shape} / グループ {len(gm):,}')

dis = (pk.pclass == 'distal').values
pro = (pk.pclass == 'promoter-TSS').values
w(f'遠位ピーク {dis.sum():,} / 近位ピーク {pro.sum():,}')
gm['distal_frag'] = PB[:, dis].sum(1)
gm['prox_frag'] = PB[:, pro].sum(1)
gm['distal_share'] = gm.distal_frag / (gm.distal_frag + gm.prox_frag)
gm['log_depth'] = np.log10(gm.total_frag + 1)

MINCELL = 50
g = gm[(gm.n_cells >= MINCELL) & gm.age.notna()].copy()
w(f'細胞 {MINCELL} 個以上のグループ {len(g):,} / ドナー {g.donor.nunique()}')
w(f'遠位割合 全体 中央 {g.distal_share.median():.4f}')

def pcorr(x, y, z):
    R = np.column_stack([rankdata(x), rankdata(y)] + [rankdata(c) for c in z])
    R = R - R.mean(0)
    A = R[:, 2:]
    rx = R[:, 0] - A @ np.linalg.lstsq(A, R[:, 0], rcond=None)[0]
    ry = R[:, 1] - A @ np.linalg.lstsq(A, R[:, 1], rcond=None)[0]
    n = len(rx)
    r = float(np.corrcoef(rx, ry)[0, 1])
    from scipy.stats import t as tdist
    df = n - 2 - A.shape[1]
    tv = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
    return r, float(2 * tdist.sf(abs(tv), df))

w('\n[遠位割合 × 暦年齢]（グループ = ドナー × subset）')
rows = []
for s, gg in g.groupby('subset'):
    if len(gg) < 30: continue
    r = spearmanr(gg.age, gg.distal_share)
    rp, pp = pcorr(gg.age.values, gg.distal_share.values, [gg.log_depth.values])
    rows.append(dict(subset=s, n=len(gg), rho=r.statistic, p=r.pvalue,
                     rho_depth_partial=rp, p_depth_partial=pp,
                     depth_x_share=spearmanr(gg.log_depth, gg.distal_share).statistic))
    w(f'  {s:14s} n={len(gg):4d}  rho {r.statistic:+.3f} (p={r.pvalue:.2g})  '
      f'深度を除くと {rp:+.3f} (p={pp:.2g})')
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/per_subset.csv', index=False)
w(f'  正の subset {int((R.rho>0).sum())}/{len(R)} / 深度を除いても正 {int((R.rho_depth_partial>0).sum())}/{len(R)}')
w(f'  深度 × 遠位割合 の相関 中央 {R.depth_x_share.median():+.3f}'
  '（大きいなら遠位割合は深度の代理になっている）')

dn = g.groupby('donor').agg(age=('age', 'first'), share=('distal_share', 'mean'),
                            depth=('log_depth', 'mean')).reset_index()
r = spearmanr(dn.age, dn.share); rp, pp = pcorr(dn.age.values, dn.share.values, [dn.depth.values])
w(f'\n  ドナー単位 n={len(dn)}  rho {r.statistic:+.3f} (p={r.pvalue:.2g})  '
  f'深度を除くと {rp:+.3f} (p={pp:.2g})')
gm.to_csv(f'{OUT}/groups_with_share.csv', index=False)
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')
