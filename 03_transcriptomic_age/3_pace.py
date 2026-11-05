#!/usr/bin/env python3
import os
import numpy as np, pandas as pd

D = '${PAPER_DIR}'
SRC = f'{D}/results/gene_mc_canonical'
OUT = f'{D}/results/tage_canonical'
L = []
def w(s):
    print(s, flush=True); L.append(s)

md = pd.read_csv(f'{SRC}/Metadata_metacells_canonical.csv')
w(f'メタセル {len(md):,} / {md.cohort.value_counts().to_dict()}')

a = pd.read_csv('${DATA_DIR}/lnc_project_AIDA_dataset/by_subcluster/tage/'
                'tage_all_combined_v1.csv').drop_duplicates('Sample')
A = a.set_index('Sample')[['age', 'sex', 'country']]
s = pd.read_csv('${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/'
                'subcluster/tage_all_combined_v2.csv')
S = s.drop_duplicates(['Donor_id', 'Batch']).set_index(['Donor_id', 'Batch'])[['Age', 'Sex']]

E = pd.read_csv(f'{SRC}/Exprs_metacells_canonical.csv', index_col=0)
w(f'発現行列 {E.shape[0]:,} 遺伝子 × {E.shape[1]:,} メタセル')

for coh, tag in [('AIDA', 'aida'), ('Synapse', 'synapse')]:
    os.makedirs(f'{OUT}/{tag}', exist_ok=True)
    m = md[md.cohort == coh].copy()
    if coh == 'AIDA':
        m = m.join(A, on='Sample')
        m['Batch'] = m.Sample
    else:
        m['Batch'] = [i.split('|')[0].split('.')[-1] if '.' in i.split('|')[0] else ''
                      for i in m.ID]

        key = s.drop_duplicates('Donor_id').set_index('Donor_id')[['Age', 'Sex']]
        m = m.join(key, on='Sample').rename(columns={'Age': 'age', 'Sex': 'sex'})
        m['country'] = ''
    m['Treatment'] = 'all'
    m['CellType'] = m.Subcluster
    m = m.dropna(subset=['age'])
    cols = ['ID', 'Sample', 'Subcluster', 'n_cells', 'Coverage', 'age', 'sex',
            'country', 'Treatment', 'CellType']
    m[cols].set_index('ID').to_csv(f'{OUT}/{tag}/Metadata_metacells.csv')
    sub = [c for c in E.columns if c in set(m.ID)]
    E[sub].to_csv(f'{OUT}/{tag}/Exprs_metacells.csv')
    w(f'  {coh}: {len(m):,} メタセル / 年齢 {m.age.min():.0f}〜{m.age.max():.0f} / '
      f'発現列 {len(sub):,}')
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')


import pandas as pd
from pathlib import Path

AUDIT_OK = {
    'aida_tage.csv': 'sex / country だけを抜き出す。pace / tAge / age は読まない',
    'synapse_tage.csv': '同上（Sex のみ）',
}

D = Path('${PROJECT_DIR}/results/'
         'PAPER_CRE_MECHANISM')
SA = D / 'results' / 'shared_assets'
OUT = D / 'results' / 'donor_attributes'
OUT.mkdir(parents=True, exist_ok=True)

a = (pd.read_csv(SA / 'aida_tage.csv').drop_duplicates('Sample')
       [['Sample', 'sex', 'country']].rename(columns={'Sample': 'donor'}))
a['cohort'] = 'AIDA'
s = (pd.read_csv(SA / 'synapse_tage.csv').drop_duplicates('Donor_id')
       [['Donor_id', 'Sex']].rename(columns={'Donor_id': 'donor', 'Sex': 'sex'}))
s['country'] = pd.NA
s['cohort'] = 'Terekhova'
r = pd.concat([a, s], ignore_index=True)[['cohort', 'donor', 'sex', 'country']]
assert r.donor.is_unique and len(r) == 499, (len(r), r.donor.is_unique)
r.to_csv(OUT / 'donor_attributes.csv', index=False)
print('ドナー属性', r.shape, '->', OUT)


import os
import numpy as np, pandas as pd

D = '${PAPER_DIR}'
FD = f'{D}/results/figures_data/figures_data'
OUT = f'{D}/results/srcdata_fig2f_fig3i_canonical'
os.makedirs(OUT, exist_ok=True)
L = []
def w(s):
    print(s, flush=True); L.append(s)

PA = pd.read_csv(f'{D}/results/tage_canonical/pace_aida_canonical.csv')
PS = pd.read_csv(f'{D}/results/tage_canonical/pace_synapse_canonical.csv')
pace_a = PA.groupby('donor').pace.median()
pace_s = PS.groupby('donor').pace.median()
w(f'正典 pace: AIDA {len(pace_a)} 名 / Terekhova {len(pace_s)} 名')

a = pd.read_csv(f'{FD}/aida_setscores.csv'); a['cohort'] = 'AIDA'
t = pd.read_csv(f'{FD}/terekhova_setscores.csv'); t = t[t.visit == 1]; t['cohort'] = 'Terekhova'
old_a = a.groupby('donor').pace.median()
w(f'旧 pace との差: 中央 {(pace_a.reindex(old_a.index) - old_a).abs().median():.3f} 年 / '
  f'最大 {(pace_a.reindex(old_a.index) - old_a).abs().max():.3f} 年')
a['pace'] = a.donor.map(pace_a)
t['pace'] = t.donor.map(pace_s)
a['tAge'] = np.nan; t['tAge'] = np.nan
ta = PA.set_index(['donor','subset']).pace.to_dict()
ts = PS.set_index(['donor','subset']).pace.to_dict()
a['tAge_celltype_diff'] = [ta.get((d,s), np.nan) for d,s in zip(a.donor, a['subset'])]
t['tAge_celltype_diff'] = [ts.get((d,s), np.nan) for d,s in zip(t.donor, t['subset'])]

cg = pd.read_csv(f'{D}/results/cima_setscore/groups_with_scores.csv')
cd = pd.read_csv(f'{D}/results/cima_part4b/donor_level.csv')[['donor','pace']]
c = cg[cg.n_cells >= 50].merge(cd, on='donor', how='inner')
C = pd.DataFrame(dict(cohort='CIMA', donor=c.donor, visit=1, subset=c['subset'],
    n_cells=c.n_cells, total_counts=c.total_frag, age=c.age, sex=c.sex, site='',
    tAge=np.nan, tAge_celltype_diff=np.nan, pace=c.pace,
    score_up=c.score_up, score_down=c.score_down, diff=c['diff']))

COLS = ['cohort','donor','visit','subset','n_cells','total_counts','age','sex','site',
        'tAge','tAge_celltype_diff','pace','score_up','score_down','diff']
I3 = pd.concat([a.reindex(columns=COLS), t.reindex(columns=COLS), C.reindex(columns=COLS)],
               ignore_index=True)

def canonical_tertiles(df, site_col='site'):
    dn = df.drop_duplicates('donor')[['donor', site_col, 'age', 'pace']].dropna(subset=['pace'])
    dn = dn.copy()
    dn['grp'] = dn[site_col].replace('', 'ALL').fillna('ALL')
    adj = []
    for g, sub in dn.groupby('grp'):
        if len(sub) < 15:
            adj.append(sub.pace - sub.pace.mean()); continue
        b = np.polyfit(sub.age, sub.pace, 1)
        adj.append(sub.pace - np.polyval(b, sub.age))
    dn['pace_adj'] = pd.concat(adj)
    dn['tert'] = dn.groupby('grp').pace_adj.transform(
        lambda s: pd.qcut(s, 3, labels=['slow','mid','fast']) if len(s) >= 15 else 'mid')
    return dn.set_index('donor')[['pace_adj','tert']]

parts = []
for co, g in I3.groupby('cohort'):
    tt = canonical_tertiles(g)
    g = g.copy()
    g['pace_adj'] = g.donor.map(tt.pace_adj)
    g['pace_tertile'] = g.donor.map(tt.tert).astype(str)
    a1, a2 = g.drop_duplicates('donor').age.quantile([1/3, 2/3])
    g['age_tertile'] = np.where(g.age <= a1, 'young', np.where(g.age >= a2, 'old', 'mid'))
    parts.append(g)
I3 = pd.concat(parts, ignore_index=True)
I3.to_csv(f'{OUT}/Fig3i.csv', index=False)
F2 = I3[I3.cohort == 'AIDA'].copy(); F2.to_csv(f'{OUT}/Fig2f.csv', index=False)
w(f'\nFig3i {I3.shape} / Fig2f {F2.shape}')

w('\n=== 三分位 × 国（正典定義。交絡が解消したか）===')
d = I3[I3.cohort == 'AIDA'].drop_duplicates('donor')
ct = pd.crosstab(d.pace_tertile, d.site)
w(ct.to_string())
w(f'  年齢中央値: ' + ' / '.join(f'{k} {v:.0f}' for k, v in
                              d.groupby('pace_tertile').age.median().items()))
w('\n（旧・素の三分位は fast 9 JP/102 KR、slow 93 JP/18 KR、年齢中央 slow 30 / fast 43 だった）')
open(f'{OUT}/report.txt','w').write('\n'.join(L))
print(f'-> {OUT}')


import pandas as pd, numpy as np, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family': 'sans-serif',
                     'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans'],
                     'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.8})
W = '${SCRATCH}'
B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUT = f'{D}/results/clock_contribution'
os.makedirs(OUT, exist_ok=True)
C_UP, C_DN, C_GRAY = '#C0392B', '#2471A3', '#7F8C8D'

co = pd.read_csv(f'{W}/clock_coef.csv')
mm = pd.read_csv('${CONDA_ENVS}/tage/lib/R/library/tAge/extdata/metadata/Gene_table_mouse.csv')
mm['Entrez'] = pd.to_numeric(mm.Entrez, errors='coerce')
e2s = dict(zip(mm.Entrez.dropna().astype(int), mm['Gene.Symbol']))
co['mouse_entrez'] = pd.to_numeric(co.mouse_entrez, errors='coerce')
co['mouse_sym'] = co.mouse_entrez.map(e2s)
co['human_sym'] = co.mouse_sym.astype(str).str.upper()
co = co.dropna(subset=['mouse_sym'])
print(f'非ゼロ係数 {len(co):,} / シンボル変換 {co.mouse_sym.notna().sum():,}')
print(f'  正の係数 {int((co.coef>0).sum()):,} / 負の係数 {int((co.coef<0).sum()):,}')
co.to_csv(f'{OUT}/clock_coefficients_named.csv', index=False)

up = co[co.coef > 0].nlargest(20, 'coef')
dn = co[co.coef < 0].nsmallest(20, 'coef')
print('\n★時計を「老けている」方向に押す遺伝子 上位20:')
print('  ' + ', '.join(f'{r.human_sym}({r.coef:+.3f})' for _, r in up.iterrows()))
print('\n★時計を「若い」方向に押す遺伝子 上位20:')
print('  ' + ', '.join(f'{r.human_sym}({r.coef:+.3f})' for _, r in dn.iterrows()))

ae = pd.read_csv(f'{D}/results/cd4_redo/aging_enhancers_all.csv')
enh_genes = set(ae.gene.dropna().astype(str).str.upper())
clock_genes = set(co.human_sym)
ov = enh_genes & clock_genes
print(f'\n=== 老化エンハンサーの近接遺伝子 と 時計遺伝子 の重なり ===')
print(f'  老化エンハンサー近接 {len(enh_genes):,} / 時計 {len(clock_genes):,} / 重なり {len(ov):,}')
print(f'  重なる遺伝子: {", ".join(sorted(ov)[:40])}')

KEY = ['ZEB2', 'PRDM1', 'KLF6', 'KLF2', 'KLF4', 'KLF14', 'LMNA', 'TNFAIP3', 'FOS', 'AHNAK',
       'ANXA1', 'BIRC3', 'BCL2L11', 'IL15', 'CD69', 'HSPA5', 'ATF4', 'PPP1R15A', 'PABPC1']
print('\n=== ★主要遺伝子が時計に入っているか（循環判定）===')
rows = []
for g in KEY:
    inc = g in clock_genes
    cf = co[co.human_sym == g].coef.values
    rows.append(dict(gene=g, in_clock=inc, coef=float(cf[0]) if len(cf) else np.nan))
    print(f'  {g:10s} {"★循環（時計の入力）" if inc else "クリーン"}'
          + (f'  係数 {cf[0]:+.4f}' if len(cf) else ''))
pd.DataFrame(rows).to_csv(f'{OUT}/key_gene_clock_status.csv', index=False)

print('\n=== 時計遺伝子の経路濃縮（Hallmark、超幾何）===')
import glob
gmt = None
for pat in ['**/h.all*.gmt', '**/*hallmark*.gmt', '**/*Hallmark*.gmt']:
    f = glob.glob(f'{D}/rawdata_links/ref_genesets/{pat}', recursive=True)
    if f: gmt = f[0]; break
if gmt is None:
    f = glob.glob(f'{D}/rawdata_links/ref_genesets/**/*.gmt', recursive=True)
    print('  GMT 候補:', [os.path.basename(x) for x in f[:12]])
    gmt = next((x for x in f if 'h.all' in x.lower() or 'hallmark' in x.lower()), f[0] if f else None)
if gmt:
    print(f'  使用: {os.path.basename(gmt)}')
    sets = {}
    for line in open(gmt):
        p = line.rstrip('\n').split('\t')
        if len(p) > 2: sets[p[0]] = set(x.upper() for x in p[2:] if x)
    UNIV = clock_genes | enh_genes
    res = []
    for k, v in sets.items():
        a = len(clock_genes & v); b = len(clock_genes) - a
        c = len(v & UNIV) - a; d = len(UNIV) - a - b - c
        if a < 3: continue
        orr, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        res.append(dict(term=k, n_clock=a, n_set=len(v & UNIV), OR=orr, p=p))
    R = pd.DataFrame(res).sort_values('p')
    from statsmodels.stats.multitest import multipletests
    if len(R):
        R['fdr'] = multipletests(R.p, method='fdr_bh')[1]
        R.to_csv(f'{OUT}/clock_gsea_hallmark.csv', index=False)
        print(R.head(15)[['term', 'n_clock', 'n_set', 'OR', 'p', 'fdr']].to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={'width_ratios': [1.15, 1]})
a0 = ax[0]
sc = co.sort_values('coef')
a0.scatter(np.arange(len(sc)), sc.coef, s=8,
           c=np.where(sc.coef > 0, C_UP, C_DN), linewidth=0, rasterized=True)
a0.axhline(0, color='k', lw=0.8)
a0.set_xlabel(f'Genes ranked by coefficient (n = {len(sc):,} non-zero)')
a0.set_ylabel('Elastic-net coefficient')
a0.set_title('A   Genes that build the transcriptomic clock', loc='left', fontweight='bold')
for _, r in pd.concat([up.head(6), dn.head(6)]).iterrows():
    i = int(np.where(sc.mouse_entrez.values == r.mouse_entrez)[0][0])
    a0.annotate(r.human_sym, (i, r.coef), fontsize=8, ha='left' if r.coef < 0 else 'right',
                va='center', xytext=(6 if r.coef < 0 else -6, 0), textcoords='offset points',
                color=C_DN if r.coef < 0 else C_UP)

a1 = ax[1]
lab = ['Clock only', 'Both', 'Ageing enhancer\nonly']
val = [len(clock_genes - enh_genes), len(ov), len(enh_genes - clock_genes)]
bars = a1.bar(np.arange(3), val, color=[C_GRAY, '#8E44AD', C_UP], width=0.6, linewidth=0)
a1.set_xticks(np.arange(3)); a1.set_xticklabels(lab)
a1.set_ylabel('Genes (n)')
a1.set_title('B   Overlap with ageing-enhancer target genes', loc='left', fontweight='bold')
for b, v in zip(bars, val):
    a1.text(b.get_x() + b.get_width() / 2, v, f'{v:,}', ha='center', va='bottom', fontsize=10)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/clock_contribution.{ext}', dpi=300, bbox_inches='tight')
print(f'\n-> {OUT}/clock_contribution.pdf / .png')
