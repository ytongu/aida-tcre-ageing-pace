#!/usr/bin/env python3
import os
import numpy as np, pandas as pd
from scipy import stats
D = '${PAPER_DIR}'
O = f'{D}/results/singlecell_canonical'
SC = '${DATA_DIR}/integrated_aida_synapse_hg38_440/output/sc_count'
L = []
def w(s=''):
    print(s); L.append(str(s))
f = pd.read_csv(f'{D}/results/srcdata_fig2f_fig3i_canonical/Fig3i.csv',
                usecols=['cohort', 'donor', 'age', 'sex', 'site', 'pace', 'pace_adj', 'pace_tertile'])
f = f[f.cohort == 'AIDA'].drop_duplicates('donor').rename(columns={'site': 'country'})
f['sc_available'] = [os.path.exists(f'{SC}/{d}/{d}/matrix/matrix.mtx') for d in f.donor]
w(f'AIDA ドナー {len(f)} / 単一細胞行列あり {int(f.sc_available.sum())}')
pairs = []
for c, g in f[f.sc_available].groupby('country'):
    fast = g[g.pace_tertile == 'fast'].sort_values('pace_adj', ascending=False)
    slow = g[g.pace_tertile == 'slow'].copy(); used = set()
    for _, r in fast.iterrows():
        cand = slow[~slow.donor.isin(used)]
        if not len(cand): break
        j = (cand.age - r.age).abs().idxmin(); s = slow.loc[j]
        if abs(s.age - r.age) > 5: continue
        used.add(s.donor); pairs.append((c, r.donor, s.donor))
w(f'マッチしたペア {len(pairs)}')
f['group'] = f.pace_tertile; f['matched'] = False; f['matched_pair'] = np.nan
for i, (c, a, b) in enumerate(pairs):
    f.loc[f.donor == a, ['matched', 'matched_pair']] = [True, i]
    f.loc[f.donor == b, ['matched', 'matched_pair']] = [True, i]
f = f.rename(columns={'donor': 'Donor_id'})
f.to_csv(f'{O}/donor_split.csv', index=False)
m = f[f.matched]
w(pd.crosstab(m.group, m.country).to_string())
w(f'暦年齢 fast {m[m.group=="fast"].age.median():.0f} / slow {m[m.group=="slow"].age.median():.0f}  '
  f'MWU p={stats.mannwhitneyu(m[m.group=="fast"].age, m[m.group=="slow"].age).pvalue:.3g}')
w(f'pace_adj fast {m[m.group=="fast"].pace_adj.median():+.2f} / slow {m[m.group=="slow"].pace_adj.median():+.2f}')
w(f'性: fast 女性 {100*(m[m.group=="fast"].sex=="female").mean():.0f}% / slow {100*(m[m.group=="slow"].sex=="female").mean():.0f}%')
old = pd.read_csv(f'{D}/results/singlecell/scd_agingcre/donor_split_PINNED.csv')
old = old[old.matched & old.group.isin(['fast', 'slow'])]
both = set(old.Donor_id) & set(m.Donor_id)
w(f'旧 PINNED {len(old)} 名との共通 {len(both)} 名')
open(f'{O}/report_01.txt', 'w').write('\n'.join(L))


import os, time
import numpy as np, pandas as pd
np.random.seed(1)
D = '${PAPER_DIR}'
O = f'{D}/results/singlecell_canonical'
W = '${HOME}/projects/integrated_aida_synapse/work/singlecell_canonical'
os.makedirs(f'{W}/thin', exist_ok=True)
SC = '${DATA_DIR}/integrated_aida_synapse_hg38_440/output/sc_count'
DEP = 2000; NROW = 329471
SUBS = ['CD4_Naive', 'CD4_Memory', 'CD8_Effector', 'CD8_Naive', 'NK', 'B_Naive', 'CD14_Mono']
SUBID = {s: i for i, s in enumerate(SUBS)}
don = pd.read_csv(f'{O}/donor_split.csv'); don = don[don.matched & don.sc_available]
dmap = dict(zip(don.Donor_id, don.group)); pmap = dict(zip(don.Donor_id, don.matched_pair)); amap = dict(zip(don.Donor_id, don.age))
bm = pd.read_parquet(f'{D}/results/canonical_labels/barcode_subset_canonical.parquet',
                     columns=['cohort', 'sample', 'barcode_full', 'Donor_id', 'subset_canonical'])
bm = bm[(bm.cohort == 'AIDA') & bm.Donor_id.isin(dmap) & bm.subset_canonical.isin(SUBS)]
print('donors', len(dmap), 'candidate cells', len(bm), flush=True)
acc = np.zeros((len(SUBS), NROW)); ncells = np.zeros(len(SUBS), np.int64)
cells = []; t0 = time.time()
for si, (smp, meta) in enumerate(bm.groupby('sample')):
    d = f'{SC}/{smp}/{smp}/matrix'
    if not os.path.exists(f'{d}/matrix.mtx'): print('  !! no matrix', smp); continue
    bc = pd.read_csv(f'{d}/barcodes.tsv', header=None)[0].values
    m = pd.read_csv(f'{d}/matrix.mtx', sep=' ', skiprows=3, header=None, dtype=np.int32).values
    r = m[:, 0] - 1; c = m[:, 1] - 1; v = m[:, 2].astype(np.int64); ncell = len(bc)
    umi = np.bincount(c, weights=v, minlength=ncell); ndet = np.bincount(c, minlength=ncell)
    sub = meta.set_index('barcode_full')['subset_canonical'].reindex([f'{smp}_{b}' for b in bc]).values
    dn = meta.Donor_id.iloc[0]
    keep = (~pd.isna(sub)) & (umi >= DEP)
    if keep.sum() == 0: continue
    kc = np.where(keep)[0]; n = len(kc)
    loc = np.full(ncell, -1, np.int64); loc[kc] = np.arange(n)
    p = np.zeros(ncell); p[keep] = DEP / umi[keep]
    vt = np.random.binomial(v, np.clip(p[c], 0, 1))
    nz = (vt > 0) & (loc[c] >= 0)
    rr = r[nz].astype(np.int32); ri = loc[c[nz]].astype(np.int32); vv = vt[nz].astype(np.int16)
    np.savez(f'{W}/thin/{smp}.npz', row=rr, cell=ri, val=vv)
    code = np.array([SUBID[x] for x in sub[kc]])
    for k in range(len(SUBS)):
        sel = code[ri] == k
        acc[k] += np.bincount(rr[sel], weights=vv[sel].astype(float), minlength=NROW)
        ncells[k] += int((code == k).sum())
    cells.append(pd.DataFrame({'sample': smp, 'cell': np.arange(n), 'donor': dn, 'group': dmap[dn], 'pair': pmap[dn],
                               'age': amap[dn], 'subcluster': sub[kc], 'umi_raw': umi[kc], 'ndet_raw': ndet[kc],
                               'umi_thin': np.bincount(ri, weights=vv.astype(float), minlength=n),
                               'ndet_thin': np.bincount(ri, minlength=n)}))
    if si % 20 == 0: print(si, smp, 'kept', n, '/', ncell, '%.0fs' % (time.time() - t0), flush=True)
C = pd.concat(cells, ignore_index=True); C.to_parquet(f'{W}/cells.parquet', index=False)
np.savez_compressed(f'{O}/percre_sums.npz', acc=acc, ncells=ncells, subs=np.array(SUBS))
print('DONE cells', len(C), 'donors', C.donor.nunique(), '%.0fs' % (time.time() - t0), flush=True)
print(C.groupby(['subcluster', 'group']).size())


import pickle
import numpy as np, pandas as pd
D = '${PAPER_DIR}'
O = f'{D}/results/singlecell_canonical'
GATE = 50
L = []
def w(s=''):
    print(s); L.append(str(s))
genes = pd.read_csv('${DATA_DIR}/integrated_aida_synapse_hg38_440/output/sc_count/${ANY_LIBRARY}/${ANY_LIBRARY}/matrix/genes.tsv', sep='\t', header=None)
row = {c: i for i, c in enumerate(genes[0].values)}
rob = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv')
z = np.load(f'{O}/percre_sums.npz', allow_pickle=True); SUBS = list(z['subs']); acc = z['acc']; NC = z['ncells']
out = {}
for k, s in enumerate(SUBS):
    r = rob[rob.cell_type == s]
    up = sorted({row[c] for c in r[r.program == 'up_distal'].cre_id if c in row})
    dn = sorted({row[c] for c in r[r.program == 'down_proximal'].cre_id if c in row})
    ok = len(up) >= GATE and len(dn) >= GATE
    w(f'{s:13s} up_distal {len(up):4d} / down_proximal {len(dn):4d}  -> {"採用" if ok else "不採用（50 未満の腕あり）"}')
    if not ok: continue
    mu = acc[k] / max(NC[k], 1)
    lab = np.zeros(len(mu), np.int8); lab[[row[c] for c in r.cre_id if c in row]] = 1
    bgrow = np.where((lab == 0) & (mu > 0))[0]; bgrow = bgrow[np.argsort(mu[bgrow])]; bmu = mu[bgrow]
    used = np.zeros(len(bgrow), bool); res = {}
    for g, hr in (('distal_up', np.array(up)), ('proximal_down', np.array(dn))):
        hr = hr[mu[hr] > 0]; pick = []
        for m_ in mu[hr]:
            i = np.searchsorted(bmu, m_); best = -1; bd = 1e18
            for j in range(max(0, i - 200), min(len(bgrow), i + 200)):
                if used[j]: continue
                dd = abs(bmu[j] - m_)
                if dd < bd: bd = dd; best = j
            if best >= 0: used[best] = True; pick.append(bgrow[best])
        res[g] = hr; res['mbg_' + g] = np.array(pick)
        w(f'    {g}: hit {len(hr)} mu {mu[hr].sum():.4f} | 背景 {len(pick)} mu {mu[np.array(pick)].sum():.4f}')
    out[s] = res
pickle.dump(out, open(f'{O}/matchsets.pkl', 'wb'))
w(f'採用 subcluster: {list(out)}')
open(f'{O}/report_03.txt', 'w').write('\n'.join(L))


import pickle
import numpy as np, pandas as pd
D = '${PAPER_DIR}'
O = f'{D}/results/singlecell_canonical'
W = '${HOME}/projects/integrated_aida_synapse/work/singlecell_canonical'
SETS = ['distal_up', 'mbg_distal_up', 'proximal_down', 'mbg_proximal_down']
ms = pickle.load(open(f'{O}/matchsets.pkl', 'rb')); NROW = 329471
lab = {}
for s, d in ms.items():
    v = np.zeros(NROW, np.int8)
    for i, k in enumerate(SETS, 1): v[d[k]] = i
    lab[s] = v
C = pd.read_parquet(f'{W}/cells.parquet')
parts = []
for smp, cs in C.groupby('sample'):
    cs = cs[cs.subcluster.isin(lab)].copy()
    if not len(cs): continue
    z = np.load(f'{W}/thin/{smp}.npz'); rr, ri, vv = z['row'], z['cell'], z['val'].astype(float)
    n = int(C[C['sample'] == smp].cell.max()) + 1
    subc = np.array([''] * n, dtype=object); subc[cs.cell.values] = cs.subcluster.values
    for k in SETS: cs['n_' + k] = 0.0; cs['s_' + k] = 0.0
    for s in lab:
        em = subc[ri] == s
        lr = lab[s][rr[em]]; rie = ri[em]; ve = vv[em]
        for i, k in enumerate(SETS, 1):
            sel = lr == i
            nn = np.bincount(rie[sel], minlength=n); ss = np.bincount(rie[sel], weights=ve[sel], minlength=n)
            m = cs.subcluster.values == s
            cs.loc[m, 'n_' + k] += nn[cs.cell.values[m]]; cs.loc[m, 's_' + k] += ss[cs.cell.values[m]]
    parts.append(cs)
P = pd.concat(parts, ignore_index=True).drop(columns=['sample', 'cell'])
P.to_parquet(f'{O}/percell_canonical.parquet', index=False)
print('cells', len(P), P.groupby(['subcluster', 'group']).size().to_dict())


import pandas as pd, numpy as np
from scipy import stats
O='${PAPER_DIR}/results/singlecell_canonical'
C=pd.read_parquet(f'{O}/percell_canonical.parquet')
C['site']=C.donor.str.split('_').str[:2].str.join('_')
SETS=['distal_up','mbg_distal_up','proximal_down','mbg_proximal_down']
out=open(f'{O}/FINAL.txt','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.write(s+'\n'); out.flush()

P('='*86)
P('単位: 細胞 %d / ドナー %d / subcluster %d'%(len(C),C.donor.nunique(),C.subcluster.nunique()))
u=C.drop_duplicates('donor')
P('  fast %d / slow %d'%((u.group=='fast').sum(),(u.group=='slow').sum()))
P('  サイト x 群:'); P(pd.crosstab(u.site,u.group).to_string())
P('  chi2 p=%.3g (1 に近いほど交絡なし)'%stats.chi2_contingency(pd.crosstab(u.site,u.group))[1])
P('  暦年齢 fast %.1f slow %.1f MWU p=%.3g'%(u[u.group=="fast"].age.mean(),u[u.group=="slow"].age.mean(),
    stats.mannwhitneyu(u[u.group=="fast"].age,u[u.group=="slow"].age).pvalue))
P('深さ: 全細胞 binomial thinning で期待 2000 UMI (元 UMI<2000 は除外)')
P('mbg_* = 同一 subcluster 内で mu を 1:1 最近傍マッチした tested non-hit 背景 (CRE 数・総 mu 一致)')
P('老化 CRE: 正典の頑健集合 canon166 (subcluster 単位)。各腕 50 素子以上の subcluster のみ (report_03.txt)')

P('\n--- 深さの群間差 (単位: ドナー、細胞平均) ---')
P('  thinning 後は umi_thin が全細胞で期待 2000 に固定されるが、元の深さと')
P('  深さ統一後の検出 tCRE 数 (ライブラリ複雑度) は群間で違いうるので実測する')
dq=C.groupby(['donor','group'])[['umi_raw','ndet_raw','umi_thin','ndet_thin']].mean().reset_index()
for v in ['umi_raw','ndet_raw','umi_thin','ndet_thin']:
    a=dq[dq['group']=='fast'][v]; b=dq[dq['group']=='slow'][v]
    P('  %-10s fast %8.1f slow %8.1f  MWU p=%.3g'%(v,a.mean(),b.mean(),stats.mannwhitneyu(a,b).pvalue))

P('\n'+'='*86)
P('[A] ★本題: 老化 CRE は全細胞で薄く上がるのか、一部の細胞で強く上がるのか')
P('単位: ドナー x subcluster (同一人・同一細胞状態、細胞 200 以上)。その中央値を報告')
P('  det%   セットのいずれかの CRE を検出する細胞の割合')
P('  DI     per-cell セットカウントの分散/平均。全細胞が同じ強度なら 1')
P('  c50%   総シグナルの 50% を担う細胞の割合。均一なら 50')
P('  top10% 上位 10% の細胞が担う割合。均一なら 10')
rows=[]
for (d,s,g),cs in C.groupby(['donor','subcluster','group']):
    if len(cs)<200: continue
    for k in SETS:
        x=cs['s_'+k].values
        if x.mean()<=0: continue
        srt=np.sort(x)[::-1]; tot=srt.sum(); cum=np.cumsum(srt)
        rows.append(dict(donor=d,sub=s,group=g,set=k,ncell=len(x),mu=x.mean(),det=100*np.mean(x>0),
            DI=x.var()/x.mean(),c50=100*(np.searchsorted(cum,tot*0.5)+1)/len(x),
            top10=100*cum[int(0.1*len(x))]/tot))
W=pd.DataFrame(rows); W.to_csv(f'{O}/final_within_donor.csv',index=False)
t=W.groupby(['sub','set']).agg(n=('DI','size'),mu=('mu','median'),det=('det','median'),
        DI=('DI','median'),c50=('c50','median'),top10=('top10','median')).reset_index()
P('')
P(t.round(2).to_string(index=False))
P('\n  同一人・同一細胞状態の中での hit 対 発現量マッチ背景 (対応 Wilcoxon、単位=ドナー x subcluster):')
for s in sorted(W['sub'].unique()):
    tt=t[t['sub']==s].set_index('set')
    for g in ['distal_up','proximal_down']:
        if g not in tt.index: continue
        a=W[(W['sub']==s)&(W['set']==g)].set_index('donor')['DI']
        b=W[(W['sub']==s)&(W['set']=='mbg_'+g)].set_index('donor')['DI']
        ix=a.index.intersection(b.index)
        P('  %-13s %-14s DI %.2f 対 背景 %.2f (n=%d, p=%.2g) | c50%% %.1f 対 %.1f | top10%% %.1f 対 %.1f | det%% %.1f'%(
          s,g,tt.loc[g,'DI'],tt.loc['mbg_'+g,'DI'],len(ix),stats.wilcoxon(a.loc[ix],b.loc[ix]).pvalue,
          tt.loc[g,'c50'],tt.loc['mbg_'+g,'c50'],tt.loc[g,'top10'],tt.loc['mbg_'+g,'top10'],tt.loc[g,'det']))

P('\n[B] ドナー内の細胞間ばらつきは連続か、別集団か')
P('単位: 細胞。ドナー x subcluster 内で z 化してプール')
for s in sorted(C.subcluster.unique()):
    cs=C[C.subcluster==s]
    for k in ['distal_up','mbg_distal_up']:
        g=cs.groupby('donor')['s_'+k]
        z=((cs['s_'+k]-g.transform('mean'))/g.transform('std')).values
        z=z[np.isfinite(z)]
        P('  %-13s %-18s 歪度 %+.2f 尖度 %+.2f | z>2 %.2f%% z>3 %.2f%% z>4 %.3f%% (正規: 2.28/0.13/0.003)'%(
          s,k,stats.skew(z),stats.kurtosis(z),100*np.mean(z>2),100*np.mean(z>3),100*np.mean(z>4)))

P('\n[C] 分散の分解 (ドナー間 対 ドナー内)')
P('単位: 細胞。subcluster 固定、log1p(セットカウント)。ドナー間にはサイト・バッチ差も含む')
for s in sorted(C.subcluster.unique()):
    cs=C[C.subcluster==s]
    for k in ['distal_up','mbg_distal_up']:
        y=np.log1p(cs['s_'+k].values)
        dm=cs.groupby('donor')['s_'+k].transform(lambda v:np.log1p(v).mean()).values
        vb=np.var(dm); vw=np.var(y-dm)
        P('  %-13s %-18s ドナー間 %.1f%% / ドナー内 %.1f%% (全分散 %.4f)'%(s,k,100*vb/(vb+vw),100*vw/(vb+vw),vb+vw))

P('\n'+'='*86)
P('[D] fast/slow: 老化速度の速い人で細胞あたり老化 CRE 使用が多いか')
P('単位: 検定はドナー (暦年齢マッチペアで対応 Wilcoxon)。細胞を n にしない')
P('  特異性比 = hit の fast/slow 比 / 発現量マッチ背景の fast/slow 比')
res=[]
for s in sorted(C.subcluster.unique()):
    cs=C[C.subcluster==s]
    dd=cs.groupby(['donor','group','pair'])[['s_'+k for k in SETS]].mean().reset_index()
    for g in ['distal_up','proximal_down']:
        pv=dd.pivot_table(index='pair',columns='group',values='s_'+g).dropna()
        pb=dd.pivot_table(index='pair',columns='group',values='s_mbg_'+g).dropna()
        ix=pv.index.intersection(pb.index); pv=pv.loc[ix]; pb=pb.loc[ix]
        if len(ix)<20: continue
        rh=pv['fast'].mean()/pv['slow'].mean(); rb=pb['fast'].mean()/pb['slow'].mean()
        res.append(dict(sub=s,grp=g,nPair=len(ix),hit_ratio=rh,bg_ratio=rb,spec=rh/rb,
            p_hit=stats.wilcoxon(pv['fast'],pv['slow']).pvalue,
            p_vs_bg=stats.wilcoxon(np.log(pv['fast']/pv['slow']),np.log(pb['fast']/pb['slow'])).pvalue))
P(pd.DataFrame(res).round(4).to_string(index=False))

P('\n[E] サイト別に分けても fast/slow 差は残るか (サイト交絡の直接検証)')
P('単位: 検定はドナー (サイト内、対応なし MWU)')
for site in ['JP_RIK','KR_SGI']:
    P('  [%s]'%site)
    for s in sorted(C.subcluster.unique()):
        cs=C[(C.subcluster==s)&(C.site==site)]
        dd=cs.groupby(['donor','group'])[['s_distal_up','s_mbg_distal_up']].mean().reset_index()
        f=dd[dd['group']=='fast']; sl=dd[dd['group']=='slow']
        if len(f)<10 or len(sl)<10: continue
        rh=f.s_distal_up.mean()/sl.s_distal_up.mean(); rb=f.s_mbg_distal_up.mean()/sl.s_mbg_distal_up.mean()
        P('    %-13s ドナー %d/%d | hit比 %.2f 背景比 %.2f 特異性 %.2f  MWU p=%.3g'%(
            s,len(f),len(sl),rh,rb,rh/rb,stats.mannwhitneyu(f.s_distal_up,sl.s_distal_up).pvalue))

P('\n[F] 増え方の形: 分布全体が平行移動するか、高値の別集団が生えるか')
P('単位: 細胞の分位点をドナーごとに取り、ドナー平均で比 (n=ドナー)')
for s in sorted(C.subcluster.unique()):
    cs=C[C.subcluster==s]; qs=[10,25,50,75,90,99]; rec=[]
    for (d,g),cd in cs.groupby(['donor','group']):
        if len(cd)<200: continue
        rec.append([d,g]+list(np.percentile(cd['s_distal_up'],qs))+list(np.percentile(cd['s_mbg_distal_up'],qs)))
    if len(rec)<40: continue
    Q=pd.DataFrame(rec,columns=['donor','group']+['h%d'%q for q in qs]+['b%d'%q for q in qs])
    f=Q[Q['group']=='fast']; sl=Q[Q['group']=='slow']
    if len(f)<10 or len(sl)<10: continue
    P('  [%s] ドナー fast %d slow %d'%(s,len(f),len(sl)))
    P('    分位点   '+' '.join('%7d%%'%q for q in qs))
    P('    hit比    '+' '.join('%8.2f'%(f['h%d'%q].mean()/sl['h%d'%q].mean()) if sl['h%d'%q].mean()>0 else '      NA' for q in qs))
    P('    背景比   '+' '.join('%8.2f'%(f['b%d'%q].mean()/sl['b%d'%q].mean()) if sl['b%d'%q].mean()>0 else '      NA' for q in qs))

P('\n[G] ライブラリ複雑度 (深さ統一後の検出 tCRE 数) 層別')
P('単位: 細胞を ndet_thin 5 分位に層別、層内でドナー平均 → ペア対応 Wilcoxon')
for s in sorted(C.subcluster.unique()):
    cs=C[C.subcluster==s].copy()
    if len(cs)<5000: continue
    cs['q']=pd.qcut(cs.ndet_thin,5,labels=False)
    P('  [%s]'%s)
    for q in range(5):
        cq=cs[cs.q==q]
        dd=cq.groupby(['donor','group','pair'])[['s_distal_up','s_mbg_distal_up']].mean().reset_index()
        pv=dd.pivot_table(index='pair',columns='group',values='s_distal_up').dropna()
        pb=dd.pivot_table(index='pair',columns='group',values='s_mbg_distal_up').dropna()
        ix=pv.index.intersection(pb.index); pv=pv.loc[ix]; pb=pb.loc[ix]
        if len(ix)<20: continue
        P('    Q%d (ndet中央 %.0f) hit比 %.2f 背景比 %.2f 特異性 %.2f  p(hit vs bg)=%.2g'%(
          q+1,cq.ndet_thin.median(),pv['fast'].mean()/pv['slow'].mean(),pb['fast'].mean()/pb['slow'].mean(),
          (pv['fast'].mean()/pv['slow'].mean())/(pb['fast'].mean()/pb['slow'].mean()),
          stats.wilcoxon(np.log(pv['fast']/pv['slow']),np.log(pb['fast']/pb['slow'])).pvalue))

P('\n[G2] 元の深さ (umi_raw) 5 分位で層別')
P('単位: 細胞を umi_raw 5 分位に層別、層内でドナー平均 → ペア対応 Wilcoxon')
for s in sorted(C.subcluster.unique()):
    cs=C[C.subcluster==s].copy()
    if len(cs)<5000: continue
    cs['q']=pd.qcut(cs.umi_raw,5,labels=False)
    P('  [%s]'%s)
    for q in range(5):
        cq=cs[cs.q==q]
        dd=cq.groupby(['donor','group','pair'])[['s_distal_up','s_mbg_distal_up']].mean().reset_index()
        pv=dd.pivot_table(index='pair',columns='group',values='s_distal_up').dropna()
        pb=dd.pivot_table(index='pair',columns='group',values='s_mbg_distal_up').dropna()
        ix=pv.index.intersection(pb.index); pv=pv.loc[ix]; pb=pb.loc[ix]
        if len(ix)<20: continue
        P('    Q%d (umi_raw中央 %.0f) hit比 %.2f 背景比 %.2f 特異性 %.2f  p(hit vs bg)=%.2g'%(
          q+1,cq.umi_raw.median(),pv['fast'].mean()/pv['slow'].mean(),pb['fast'].mean()/pb['slow'].mean(),
          (pv['fast'].mean()/pv['slow'].mean())/(pb['fast'].mean()/pb['slow'].mean()),
          stats.wilcoxon(np.log(pv['fast']/pv['slow']),np.log(pb['fast']/pb['slow'])).pvalue))

P('\n[G3] 検出細胞あたりの平均カウント')
P('単位: CRE 単位ではなくセット単位。セット合計は 1 細胞あたり何カウントか')
for s in sorted(C.subcluster.unique()):
    cs=C[C.subcluster==s]
    P('  %-13s distal_up セット合計 %.1f カウント/細胞 (検出 CRE 数 %.1f 個 → 1 CRE あたり %.2f) | 背景 %.1f'%(
        s,cs.s_distal_up.mean(),cs.n_distal_up.mean(),
        cs.s_distal_up.sum()/max(cs.n_distal_up.sum(),1),cs.s_mbg_distal_up.mean()))

P('\n[H] 細胞組成 (参考、単位=ドナー)')
_CA=pd.read_parquet('${HOME}/projects/integrated_aida_synapse/work/singlecell_canonical/cells.parquet')
f=_CA.groupby(['donor','group','subcluster']).size().rename('n').reset_index()
tot=f.groupby('donor').n.sum(); f['frac']=100*f.n/f.donor.map(tot)
for s in sorted(_CA.subcluster.unique()):
    d=f[f.subcluster==s]; a=d[d['group']=='fast'].frac; b=d[d['group']=='slow'].frac
    P('  %-13s fast %5.1f%% slow %5.1f%%  MWU p=%.3g'%(s,a.mean(),b.mean(),stats.mannwhitneyu(a,b).pvalue))
out.close()
