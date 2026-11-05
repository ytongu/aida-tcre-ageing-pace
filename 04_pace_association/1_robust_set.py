#!/usr/bin/env python3
import pandas as pd, numpy as np, pickle, os, sys
from scipy.stats import rankdata
from scipy.stats import t as tdist
from statsmodels.stats.multitest import multipletests

B='${PROJECT_DIR}/results'
P=f'{B}/phase3b_ourcd4'
TS='_v4'
TV='yugene_diff_EN_tAge'
MIN_CELLS=10
O='${PAPER_DIR}/results/fixes/cd4_donor_filter'
SYN_TAGE_SC='${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/subcluster/tage_all_combined_v2.csv'
L=[]
def w(s=''):
    print(s, flush=True); L.append(str(s))

w('='*90)
w('(2) Synapse CD4 をドナー単位に直して Spearman を本計算')
w('='*90)

o=pd.read_csv(SYN_TAGE_SC)
o=o[(o.model_type=='Chronoage')&(o['mode']=='EN')].copy()
o=o.rename(columns={'Age':'age'})
pair=o[['Donor_id','Batch','age']].drop_duplicates()
keep_canon=pair.sort_values(['Donor_id','age']).groupby('Donor_id').head(1)[['Donor_id','Batch']]
w(f'元 Synapse subcluster tAge: {len(o):,} 行 / (Donor,Batch) 組 {len(pair)} / ドナー {pair.Donor_id.nunique()}')
w(f'  正典 baseline keep-set: {len(keep_canon)} 訪問 (= 他 9 細胞種が使っている訪問)')

c=pd.read_csv(f'{B}/PAPER_CRE_MECHANISM/results/cd4_redo/cd4_our_tage.csv')
sy=c[c.cohort=='Synapse'].copy()
pair2=sy[['Donor_id','Batch','age']].drop_duplicates()
keep_file=pair2.sort_values(['Donor_id','age']).groupby('Donor_id').head(1)[['Donor_id','Batch']]
ov=keep_canon.merge(keep_file,on=['Donor_id','Batch'],how='inner')
w(f'  cd4 ファイル順の keep-set: {len(keep_file)} 訪問 / 正典と一致する訪問 {len(ov)} (=> {len(keep_canon)-len(ov)} 名で別訪問)')

w('\n集約キャッシュ読込中 (約 10GB) ...')
with open(f'{P}/aggregation_cache_synapse_subcluster.pkl','rb') as f:
    agg=pickle.load(f)
cres=agg['cres']; expr_sum=agg['expr_sum']; umi_sum=agg['umi_sum']; ncell=agg['n_cells']
w(f'  cres = {len(cres):,}  metacell キー = {len(expr_sum):,}')

w('\n--- キャッシュのキー検証 ---')
chk=[]
for ct in ['CD4_Memory','CD4_Naive']:
    sub=sy[sy.Group==ct]
    miss=[]; tot_cells=0; small=0
    for _,r in sub.iterrows():
        k=('Synapse', r.Donor_id, r.Batch, ct)
        if k not in expr_sum: miss.append(k)
        else:
            tot_cells+=ncell[k]
            if ncell[k]<MIN_CELLS: small+=1
    w(f'  {ct}: tAge 行 {len(sub)} / キャッシュ欠落 {len(miss)} / n_cells<{MIN_CELLS} {small} / 合計細胞 {tot_cells:,}')
    chk.append(dict(cell_type=ct,n_rows=len(sub),n_missing=len(miss),n_small=small,total_cells=tot_cells))
pd.DataFrame(chk).to_csv(f'{O}/cache_key_check.csv',index=False)
w('  参照値 (MANIFEST 3.3 Synapse): ナイーブ 630,540 / 記憶 315,927 / 除外 55,314')

def corr_one(rows, ct):

    keys=[('Synapse',r.Donor_id,r.Batch,ct) for _,r in rows.iterrows()]
    ok=[i for i,k in enumerate(keys) if k in expr_sum and ncell[k]>=MIN_CELLS]
    keys=[keys[i] for i in ok]
    y=rows[TV].values.astype(np.float64)[ok]
    good=~np.isnan(y)
    keys=[k for k,g in zip(keys,good) if g]; y=y[good]
    n=len(y)
    X=np.empty((len(cres),n),dtype=np.float64)
    for j,k in enumerate(keys):
        u=umi_sum[k]
        X[:,j]= (expr_sum[k]/u*1e6) if u>0 else 0.0
    yr=rankdata(y,method='average')
    xr=rankdata(X,axis=1,method='average')
    xm=xr.mean(axis=1,keepdims=True); xs=xr.std(axis=1,ddof=0)
    ym=yr.mean(); ys=yr.std(ddof=0)
    cov=((xr-xm)*(yr-ym)).mean(axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        rho=np.where((xs>0)&(ys>0),cov/(xs*ys+1e-12),np.nan)
        t=rho*np.sqrt((n-2)/np.clip(1-rho**2,1e-12,None))
    p=2*tdist.sf(np.abs(t),n-2)
    me=X.mean(axis=1)
    del X,xr
    return pd.DataFrame({'cre_id':cres,'cell_type':ct,'target_var':TV,'n_samples':n,
                         'mean_expression':me.astype(np.float32),
                         'spearman_rho':rho.astype(np.float32),
                         'spearman_pval':p.astype(np.float32)})

def add_fdr(df):
    df=df.copy(); df['fdr']=np.nan
    for ct,g in df.groupby('cell_type'):
        pv=g.spearman_pval.values; okm=~np.isnan(pv)
        f=np.full(len(pv),np.nan)
        if okm.sum()>0:
            _,fo,_,_=multipletests(pv[okm],method='fdr_bh'); f[okm]=fo
        df.loc[g.index,'fdr']=f
    return df

variants={'all306':None,'canon166':keep_canon,'cd4file166':keep_file}
res={}
for vn,keep in variants.items():
    w(f'\n--- 変種 {vn} ---')
    parts=[]
    for ct in ['CD4_Memory','CD4_Naive']:
        sub=sy[sy.Group==ct][['Donor_id','Batch',TV]].copy()
        if keep is not None:
            sub=sub.merge(keep,on=['Donor_id','Batch'],how='inner')
        r=corr_one(sub,ct)
        nn=int((~np.isnan(r.spearman_pval.values)).sum())
        w(f'  {ct}: n_samples={int(r.n_samples.iloc[0])}  非 NaN p 値 {nn:,} / {len(cres):,} (BH 家族サイズ)')
        parts.append(r)
    df=add_fdr(pd.concat(parts,ignore_index=True))
    for ct,g in df.groupby('cell_type'):
        w(f'  {ct}: FDR<0.05 = {int((g.fdr<0.05).sum()):,}   rho 平均 {np.nanmean(g.spearman_rho):+.4f}')
    df.to_csv(f'{O}/syn_cd4_recomputed_{vn}.csv',index=False)
    res[vn]=df

w('\n--- 自作実装の検証: all306 と正典表の一致 ---')
pub=pd.read_csv(f'{P}/cre_corr_synapse_subcluster_full_{TS}.csv',
                usecols=['cre_id','cell_type','target_var','n_samples','spearman_rho','fdr'])
pub=pub[(pub.target_var==TV)&(pub.cell_type.isin(['CD4_Memory','CD4_Naive']))]
mm=res['all306'].merge(pub,on=['cre_id','cell_type'],suffixes=('_new','_pub'))
w(f'  結合 {len(mm):,} 行')
d=(mm.spearman_rho_new-mm.spearman_rho_pub).abs()
w(f'  rho 最大絶対差 = {np.nanmax(d.values):.3e}  (NaN 一致: new {int(mm.spearman_rho_new.isna().sum())} / pub {int(mm.spearman_rho_pub.isna().sum())})')
df2=(mm.fdr_new-mm.fdr_pub).abs()
w(f'  fdr 最大絶対差 = {np.nanmax(df2.values):.3e}')
w(f'  n_samples: new {sorted(mm.n_samples_new.unique())} / pub {sorted(mm.n_samples_pub.unique())}')

open(f'{O}/report_02_recompute.txt','w').write('\n'.join(L))
print('\n-> done')


import pandas as pd, numpy as np, gzip, os
from scipy.stats import fisher_exact

B='${PROJECT_DIR}/results'
P=f'{B}/phase3b_ourcd4'
D=f'{B}/PAPER_CRE_MECHANISM'
TS='_v4'
TV='yugene_diff_EN_tAge'
O=f'{D}/results/fixes/cd4_donor_filter'
L=[]
def w(s=''):
    print(s, flush=True); L.append(str(s))

ann=pd.read_csv(f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv',
                usecols=['cre_id','proximity','enhancer_class'])
A=pd.read_csv(f'{P}/cre_corr_aida_subcluster_significant_{TS}.csv',
              usecols=['cre_id','cell_type','target_var','spearman_rho','fdr','mean_expression'])
a=A[(A.target_var==TV)&(A.fdr<0.05)].copy()
S=pd.read_csv(f'{P}/cre_corr_synapse_subcluster_full_{TS}.csv',
              usecols=['cre_id','cell_type','target_var','spearman_rho','fdr','n_samples'])
s_all=S[S.target_var==TV].copy()
w(f'AIDA 選抜 {len(a):,} 行 / Synapse 全件(軸限定) {len(s_all):,} 行')

def syn_table(variant):
    if variant=='OLD':
        return s_all[['cre_id','cell_type','spearman_rho','fdr','n_samples']].copy()
    new=pd.read_csv(f'{O}/syn_cd4_recomputed_{variant}.csv',
                    usecols=['cre_id','cell_type','spearman_rho','fdr','n_samples'])
    base=s_all[~s_all.cell_type.isin(['CD4_Memory','CD4_Naive'])][['cre_id','cell_type','spearman_rho','fdr','n_samples']]
    return pd.concat([base,new],ignore_index=True)

def clock_linked(cre_ids):
    clock=set(pd.read_csv(f'{D}/results/clock_contribution/clock_coefficients_named.csv')
              .human_sym.astype(str).str.upper())
    tssd={}
    with gzip.open('${DATA_DIR}/adachi/reference/GRCh38.113.gtf.gz','rt') as f:
        for line in f:
            if line[0]=='#': continue
            p=line.split('\t')
            if p[2]!='gene' or 'gene_biotype "protein_coding"' not in p[8]: continue
            i=p[8].find('gene_name "')
            if i<0: continue
            gn=p[8][i+11:p[8].find('"',i+11)]
            ch=p[0] if p[0].startswith('chr') else 'chr'+p[0]
            tssd.setdefault(ch,[]).append((int(p[3]) if p[6]=='+' else int(p[4]),gn))
    for ch in tssd: tssd[ch].sort()
    POS={ch:np.array([x[0] for x in v]) for ch,v in tssd.items()}
    def nearest(cid):
        try:
            ch,st,en,strand=cid.rsplit('_',3); mid=(int(st)+int(en))//2
        except Exception: return None
        arr=tssd.get(ch)
        if not arr: return None
        i=int(np.searchsorted(POS[ch],mid)); best,bd=None,None
        for j in (i-1,i):
            if 0<=j<len(arr):
                dd=abs(arr[j][0]-mid)
                if bd is None or dd<bd: bd,best=dd,arr[j][1]
        return best
    e2g=pd.read_csv(f'{B}/re2g_targets_clean/targets_clean.csv',
                    usecols=['cre_id','target_gene'])
    e2g['target_gene']=e2g.target_gene.astype(str).str.upper()
    e2g_clock=set(e2g[e2g.target_gene.isin(clock)].cre_id.unique())
    nb={c:nearest(c) for c in cre_ids}
    near_clock={c for c,g in nb.items() if g is not None and g.upper() in clock}
    return near_clock, e2g_clock, len(clock)

def build(variant):
    st=syn_table(variant).rename(columns={'spearman_rho':'rho_s','fdr':'fdr_s','n_samples':'n_syn'})
    m=a.merge(st,on=['cre_id','cell_type'],how='inner').merge(ann,on='cre_id',how='left')
    n0=len(m); nnan=int(m.rho_s.isna().sum())
    m=m.dropna(subset=['rho_s']).copy()
    m['same_sign']=np.sign(m.spearman_rho)==np.sign(m.rho_s)
    m['robust']=m.same_sign & (m.fdr_s<0.05)
    m['dir_aida']=np.where(m.spearman_rho>0,'up','down')
    m['dir_syn']=np.where(m.rho_s>0,'up','down')
    return m,n0,nnan

def prog(rb,tag):
    d=rb[rb.proximity=='distal']; p=rb[rb.proximity=='proximal']
    du=100*(d.spearman_rho>0).mean() if len(d) else np.nan
    pu=100*(p.spearman_rho>0).mean() if len(p) else np.nan
    tab=[[int((d.spearman_rho>0).sum()),int((d.spearman_rho<=0).sum())],
         [int((p.spearman_rho>0).sum()),int((p.spearman_rho<=0).sum())]]
    orv,pv=fisher_exact(tab) if min(map(min,tab))>=0 else (np.nan,np.nan)
    return dict(tag=tag,n_pairs=len(rb),n_cre=rb.cre_id.nunique(),
                distal_n=len(d),distal_up=du,proximal_n=len(p),proximal_up=pu,OR=orv,p=pv)

VARS=['OLD','canon166','cd4file166']
NAMES={'OLD':'旧版 (CD4 は訪問単位 n=306)','canon166':'★修正 (CD4 ドナー単位 n=166, 他 9 群と同一訪問)',
       'cd4file166':'感度 (CD4 ドナー単位 n=166, 別のタイブレーク)'}
M={}; summ=[]; bycell=[]; conc=[]
for v in VARS:
    m,n0,nnan=build(v)
    M[v]=m
    w('')
    w('='*90); w(f'[{v}] {NAMES[v]}'); w('='*90)
    w(f'AIDA 有意 x Synapse 対応: {n0:,} 行 (Synapse rho NaN {nnan:,} を dropna -> {len(m):,} 行)')
    rb=m[m.robust]
    st=prog(rb,v); summ.append(dict(variant=v,heldout_n=len(m),nan_dropped=nnan,**st))
    w(f'★頑健セット: 延べ {st["n_pairs"]:,} ペア / 異なり tCRE {st["n_cre"]:,}')
    w(f'  遠位 n={st["distal_n"]:,} 上昇率 {st["distal_up"]:.4f}%   近位 n={st["proximal_n"]:,} 上昇率 {st["proximal_up"]:.4f}%   OR={st["OR"]:.4f} p={st["p"]:.3e}')
    w('  細胞種別:')
    w(f'  {"細胞種":14s}{"Synapse n":>10s}{"AIDA選抜":>10s}{"頑健":>8s}{"頑健率%":>9s}{"遠位":>7s}{"近位":>7s}')
    for ct,x in m.groupby('cell_type'):
        r=x[x.robust]
        nsy=int(x.n_syn.iloc[0])
        bycell.append(dict(variant=v,cell_type=ct,n_syn=nsy,n_sel=len(x),n_robust=len(r),
                           robust_pct=100*len(r)/len(x) if len(x) else np.nan,
                           n_distal=int((r.proximity=='distal').sum()),
                           n_proximal=int((r.proximity=='proximal').sum())))
        w(f'  {ct:14s}{nsy:10d}{len(x):10,d}{len(r):8,d}{100*len(r)/len(x):8.3f}%{int((r.proximity=="distal").sum()):7,d}{int((r.proximity=="proximal").sum()):7,d}')
    cd4=int(m[m.robust & m.cell_type.str.startswith('CD4')].shape[0])
    w(f'  CD4 の占有: {cd4:,} / {len(rb):,} = {100*cd4/len(rb):.1f}%')

    nc,e2gc,nclock=clock_linked(rb.cre_id.unique())
    rb2=rb.copy()
    rb2['clock_near']=rb2.cre_id.isin(nc); rb2['clock_e2g']=rb2.cre_id.isin(e2gc)
    rb2['clock_any']=rb2.clock_near|rb2.clock_e2g
    w(f'\n  時計非ゼロ遺伝子 {nclock:,} 個に紐づく頑健 tCRE:')
    w(f'    最近接が時計遺伝子 {int(rb2.clock_near.sum()):,} / {len(rb2):,} ({100*rb2.clock_near.mean():.1f}%)')
    w(f'    rE2G 標的に時計遺伝子 {int(rb2.clock_e2g.sum()):,} ({100*rb2.clock_e2g.mean():.1f}%)')
    w(f'    ★除外対象 (どちらか) {int(rb2.clock_any.sum()):,} ({100*rb2.clock_any.mean():.1f}%)')
    ex=rb2[~rb2.clock_any]
    st2=prog(ex,v+'_clockfree'); summ.append(dict(variant=v+'_clockfree',heldout_n=np.nan,nan_dropped=np.nan,**st2))
    w(f'  時計除外後: 延べ {st2["n_pairs"]:,} ペア / 異なり tCRE {st2["n_cre"]:,}')
    w(f'    遠位 n={st2["distal_n"]:,} 上昇率 {st2["distal_up"]:.4f}%   近位 n={st2["proximal_n"]:,} 上昇率 {st2["proximal_up"]:.4f}%   OR={st2["OR"]:.4f} p={st2["p"]:.3e}')

    w('\n  符号一致率 (AIDA 選抜のみ、Synapse は方向のみ使用):')
    w(f'  {"細胞種":14s}{"上昇x遠位 n":>13s}{"一致%":>8s}{"低下x近位 n":>13s}{"一致%":>8s}')
    for ct,x in m.groupby('cell_type'):
        ud=x[(x.dir_aida=='up')&(x.proximity=='distal')]
        dp=x[(x.dir_aida=='down')&(x.proximity=='proximal')]
        u=100*(ud.dir_syn=='up').mean() if len(ud) else np.nan
        p2=100*(dp.dir_syn=='down').mean() if len(dp) else np.nan
        conc.append(dict(variant=v,cell_type=ct,n_updist=len(ud),updist_pct=u,
                         n_downprox=len(dp),downprox_pct=p2,
                         mean_rho_syn_updist=float(ud.rho_s.mean()) if len(ud) else np.nan))
        w(f'  {ct:14s}{len(ud):13,d}{u:7.2f}%{len(dp):13,d}{p2:7.2f}%')
    cdf=pd.DataFrame([r for r in conc if r['variant']==v])
    cu=cdf.dropna(subset=['updist_pct']); cp=cdf.dropna(subset=['downprox_pct'])
    w(f'  ★プール (CD4 込・tCRE 単位加重): 上昇x遠位 {np.average(cu.updist_pct,weights=cu.n_updist):.4f}% (n={int(cu.n_updist.sum()):,}) / '
      f'低下x近位 {np.average(cp.downprox_pct,weights=cp.n_downprox):.4f}% (n={int(cp.n_downprox.sum()):,})')
    cu2=cu[~cu.cell_type.str.startswith('CD4')]; cp2=cp[~cp.cell_type.str.startswith('CD4')]
    w(f'   門番 (CD4 除外・選抜総数加重): 上昇x遠位 {np.average(cu2.updist_pct,weights=[len(m[m.cell_type==c]) for c in cu2.cell_type]):.4f}% / '
      f'低下x近位 {np.average(cp2.downprox_pct,weights=[len(m[m.cell_type==c]) for c in cp2.cell_type]):.4f}%')

pd.DataFrame(summ).to_csv(f'{O}/robust_summary.csv',index=False)
pd.DataFrame(bycell).to_csv(f'{O}/robust_bycelltype.csv',index=False)
pd.DataFrame(conc).to_csv(f'{O}/concordance_bycelltype.csv',index=False)

w('')
w('='*90); w('まとめ: 旧版 vs 修正版'); w('='*90)
sm=pd.DataFrame(summ).set_index('variant')
for k in ['OLD','canon166','cd4file166']:
    r=sm.loc[k]; r2=sm.loc[k+'_clockfree']
    w(f'{k:12s} 頑健 {int(r.n_pairs):5,d} ペア / {int(r.n_cre):5,d} tCRE | 遠位 {int(r.distal_n):5,d} {r.distal_up:5.1f}% | 近位 {int(r.proximal_n):5,d} {r.proximal_up:5.1f}% | OR {r.OR:6.2f}')
    w(f'{"":12s} 時計除外 {int(r2.n_pairs):5,d} ペア / {int(r2.n_cre):5,d} tCRE | 遠位 {int(r2.distal_n):5,d} {r2.distal_up:5.1f}% | 近位 {int(r2.proximal_n):5,d} {r2.proximal_up:5.1f}% | OR {r2.OR:6.2f}')
bc=pd.DataFrame(bycell)
w('\n細胞種別 頑健ペア数の変化 (旧 -> canon166):')
piv=bc.pivot_table(index='cell_type',columns='variant',values='n_robust',aggfunc='first').fillna(0).astype(int)
for ct in piv.sort_values('OLD',ascending=False).index:
    w(f'  {ct:14s} {piv.loc[ct,"OLD"]:5,d} -> {piv.loc[ct,"canon166"]:5,d}  (感度版 {piv.loc[ct,"cd4file166"]:5,d})')
open(f'{O}/report_03_robust.txt','w').write('\n'.join(L))
print('\n-> done')


import pandas as pd, numpy as np
B='${PROJECT_DIR}/results'
P=f'{B}/phase3b_ourcd4'
D=f'{B}/PAPER_CRE_MECHANISM'
TS='_v4'; TV='yugene_diff_EN_tAge'
O=f'{D}/results/fixes/cd4_donor_filter'
ann=pd.read_csv(f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv',
                usecols=['cre_id','proximity','enhancer_class'])
A=pd.read_csv(f'{P}/cre_corr_aida_subcluster_significant_{TS}.csv',
              usecols=['cre_id','cell_type','target_var','spearman_rho','fdr','mean_expression','n_samples'])
a=A[(A.target_var==TV)&(A.fdr<0.05)].rename(columns={'spearman_rho':'rho_aida','fdr':'fdr_aida','n_samples':'n_aida'})
S=pd.read_csv(f'{P}/cre_corr_synapse_subcluster_full_{TS}.csv',
              usecols=['cre_id','cell_type','target_var','spearman_rho','fdr','n_samples'])
s_all=S[S.target_var==TV]
def mk(variant):
    if variant=='OLD':
        st=s_all[['cre_id','cell_type','spearman_rho','fdr','n_samples']].copy()
    else:
        new=pd.read_csv(f'{O}/syn_cd4_recomputed_{variant}.csv',
                        usecols=['cre_id','cell_type','spearman_rho','fdr','n_samples'])
        base=s_all[~s_all.cell_type.isin(['CD4_Memory','CD4_Naive'])][['cre_id','cell_type','spearman_rho','fdr','n_samples']]
        st=pd.concat([base,new],ignore_index=True)
    st=st.rename(columns={'spearman_rho':'rho_syn','fdr':'fdr_syn','n_samples':'n_syn'})
    m=a.drop(columns=['target_var']).merge(st,on=['cre_id','cell_type'],how='inner').merge(ann,on='cre_id',how='left')
    m=m.dropna(subset=['rho_syn']).copy()
    m['same_sign']=np.sign(m.rho_aida)==np.sign(m.rho_syn)
    r=m[m.same_sign & (m.fdr_syn<0.05)].copy()
    r['direction']=np.where(r.rho_aida>0,'up','down')
    r['program']=r.direction+'_'+r.proximity.astype(str)
    return r
old=mk('OLD'); new=mk('canon166')
new.sort_values(['cell_type','fdr_syn']).to_csv(f'{O}/robust_set_canon166.csv',index=False)
old.sort_values(['cell_type','fdr_syn']).to_csv(f'{O}/robust_set_OLD.csv',index=False)
ko=set(zip(old.cre_id,old.cell_type)); kn=set(zip(new.cre_id,new.cell_type))
print(f'OLD {len(old):,} pairs / {old.cre_id.nunique():,} tCRE')
print(f'canon166 {len(new):,} pairs / {new.cre_id.nunique():,} tCRE')
print(f'共通 {len(ko&kn):,} / 旧のみ {len(ko-kn):,} / 新のみ {len(kn-ko):,}')
lost=old[[ (c,t) in (ko-kn) for c,t in zip(old.cre_id,old.cell_type)]]
print('失われたペアの細胞種:'); print(lost.cell_type.value_counts().to_string())
print('失われたペアの program:'); print(lost.program.value_counts().to_string())
gained=new[[ (c,t) in (kn-ko) for c,t in zip(new.cre_id,new.cell_type)]]
print('新規に入ったペアの細胞種:'); print(gained.cell_type.value_counts().to_string())
print('新規の program:'); print(gained.program.value_counts().to_string())
print('\ncanon166 の program 内訳:'); print(new.program.value_counts().to_string())
print('\nOLD の program 内訳:'); print(old.program.value_counts().to_string())


import pandas as pd, numpy as np, os

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
P = f'{B}/phase3b_ourcd4'
OUT = f'{D}/results/gate3_recalc'
os.makedirs(OUT, exist_ok=True)
TV = 'yugene_diff_EN_tAge'
NE = f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv'
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

ne = pd.read_csv(NE, usecols=['cre_id', 'proximity']).dropna()
ne = ne[ne.proximity.isin(['proximal', 'distal'])].drop_duplicates('cre_id')
prox = dict(zip(ne.cre_id, ne.proximity))

def run(TS, tag):
    a = pd.read_csv(f'{P}/cre_corr_aida_subcluster_significant_{TS}.csv',
                    usecols=['cre_id', 'cell_type', 'target_var', 'spearman_rho', 'fdr'])
    a = a[(a.target_var == TV) & (a.fdr < 0.05)].copy()
    a['prox'] = a.cre_id.map(prox); a = a.dropna(subset=['prox'])
    a['dir_aida'] = np.where(a.spearman_rho > 0, 'up', 'down')
    s = pd.read_csv(f'{P}/cre_corr_synapse_subcluster_full_{TS}.csv',
                    usecols=['cre_id', 'cell_type', 'target_var', 'spearman_rho'])
    s = s[s.target_var == TV].rename(columns={'spearman_rho': 'rho_syn'})
    m = a.merge(s[['cre_id', 'cell_type', 'rho_syn']], on=['cre_id', 'cell_type'],
                how='left').dropna(subset=['rho_syn'])
    m['dir_syn'] = np.where(m.rho_syn > 0, 'up', 'down')
    rows = []
    for g, x in m.groupby('cell_type'):
        ud = x[(x.dir_aida == 'up') & (x.prox == 'distal')]
        dp = x[(x.dir_aida == 'down') & (x.prox == 'proximal')]
        rows.append(dict(TS=tag, group=g, n_sel=len(x), n_ud=len(ud), n_dp=len(dp),
                         updist=100 * (ud.dir_syn == 'up').mean() if len(ud) else np.nan,
                         downprox=100 * (dp.dir_syn == 'down').mean() if len(dp) else np.nan))
    return pd.DataFrame(rows), len(m)

w('=' * 96)
w('門番 66.9% を正典 TS で計算し直す')
w('  MANIFEST §3.5: TS 192344 は「Synapse CD4 欠落の失敗版。使わないでください」')
w('  MANIFEST §3.6: 門番 66.9% ← これは 192344 で計算されていた（自己矛盾）')
w('=' * 96)

out = []
for TS, tag in [('_v3', '192344 (禁止版)'), ('_v4', '231626 (正典)')]:
    df, nrow = run(TS, tag)
    out.append(df)
    w(f'\n--- {tag} : AIDA選抜×Synapse検証 {nrow:,} 行 ---')
    w(f'{"群":14s}{"選抜n":>9s}{"上昇遠位n":>10s}{"一致%":>8s}{"低下近位n":>10s}{"一致%":>8s}')
    for _, x in df.iterrows():
        w(f'{x.group:14s}{int(x.n_sel):9,d}{int(x.n_ud):10,d}{x.updist:7.1f}%'
          f'{int(x.n_dp):10,d}{x.downprox:7.1f}%')
    non = df[~df.group.str.startswith('CD4')].dropna(subset=['updist'])
    nonp = df[~df.group.str.startswith('CD4')].dropna(subset=['downprox'])
    allc = df.dropna(subset=['updist']); allp = df.dropna(subset=['downprox'])
    w(f'\n  [7月と同じ定義] CD4 除外・選抜総数で加重: '
      f'up-dist {np.average(non.updist, weights=non.n_sel):.1f}% / '
      f'dn-prox {np.average(nonp.downprox, weights=nonp.n_sel):.1f}%')
    w(f'  CD4 除外・上昇遠位数で加重(素子単位): '
      f'up-dist {np.average(non.updist, weights=non.n_ud):.1f}% / '
      f'dn-prox {np.average(nonp.downprox, weights=nonp.n_dp):.1f}%')
    w(f'  ★CD4 を含める・素子単位(=私が 8/4 に出した定義): '
      f'up-dist {np.average(allc.updist, weights=allc.n_ud):.1f}% / '
      f'dn-prox {np.average(allp.downprox, weights=allp.n_dp):.1f}%')
    w(f'  CD4 を含める・細胞種の非加重平均: '
      f'up-dist {allc.updist.mean():.1f}% / dn-prox {allp.downprox.mean():.1f}%')

R = pd.concat(out, ignore_index=True)
R.to_csv(f'{OUT}/gate_recompute_231626.csv', index=False)
w('\n' + '=' * 96)
w('結論')
w('=' * 96)
w('  同じデータでも「CD4 を含めるか」「何で加重するか」で 58〜67% まで動く。')
w('  本文で使う定義を 1 つに固定し、その定義を数字と一緒に必ず書くこと。')
open(f'{OUT}/report_gate.txt', 'w').write('\n'.join(L))
print(f'\n-> {OUT}/report_gate.txt')


import pandas as pd, numpy as np, gzip, os
from scipy.stats import fisher_exact

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
TS = '_v4'
OUT = f'{D}/results/gene_to_cre'
os.makedirs(OUT, exist_ok=True)
TV = 'yugene_diff_EN_tAge'
GTF = '${DATA_DIR}/adachi/reference/GRCh38.113.gtf.gz'
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

TOPICS = {
    '分化 TF（上がる側の主役）': ['ZEB2', 'PRDM1', 'KLF6', 'KLF2', 'KLF4', 'KLF14',
                                  'RORA', 'BHLHE40', 'RUNX3', 'TBX21'],
    'T 細胞の身元と TCR 入口（静かになる側）': ['LCK', 'PTPRC', 'PRKCQ', 'CCR7', 'CD7',
                                                'TCF7', 'LEF1', 'SATB1', 'SELL', 'BCL11B',
                                                'IKZF1', 'FOXO1', 'ETS1', 'LRRN3', 'MYC'],
    'ER ストレス → ISR → ストレス顆粒': ['HSPA5', 'ATF4', 'PPP1R15A', 'EIF2AK2', 'PABPC1',
                                          'ATXN2L', 'UBAP2L', 'CSDE1', 'DDX3X', 'YBX1',
                                          'NUFIP2', 'DDIT4', 'XBP1', 'CAPRIN1',
                                          'G3BP1', 'G3BP2', 'TIA1', 'TIAL1'],
    'NF-kB のブレーキと AP-1 のアクセル': ['NFKBIA', 'TNFAIP3', 'ZFP36', 'ZC3H12A', 'BIRC3',
                                           'FOS', 'FOSB', 'JUN', 'JUNB', 'ATF3', 'DUSP1',
                                           'NR4A2', 'EGR1'],
    '細胞傷害系の恒常性': ['IL15', 'SRGN', 'GZMK', 'GZMB', 'IFNG', 'NKG7', 'PRF1', 'KLRG1'],
    'アポトーシスの綱引き': ['BCL2L11', 'PMAIP1', 'MCL1', 'BCL2', 'BAX'],
    'その他の頑健な名前': ['LMNA', 'ANXA1', 'ARID5B', 'SAMSN1', 'NAMPT', 'ATM', 'G6PD',
                            'AHNAK', 'SYNE2', 'PIK3R1', 'CD69'],
}
ALL = sorted({g for v in TOPICS.values() for g in v})
w('=' * 108)
w('遺伝子単位の話題を、紐づくシス制御素子で調べ直す')
w(f'  対象 {len(ALL)} 遺伝子 / {len(TOPICS)} 話題')
w('  軸: 加齢残差 (yugene_diff_EN_tAge)。頑健 = AIDA FDR<0.05 かつ Synapse FDR<0.05 かつ同符号')
w('  ★NaN 行は除外してから符号を比較している')
w('=' * 108)

ga = pd.read_csv(f'{B}/phase2_gene_corr/aida_gene_corr_subcluster_significant.csv',
                 usecols=['gene_symbol', 'cell_type', 'target_var', 'spearman_rho', 'fdr'])
ga = ga[(ga.target_var == TV) & (ga.fdr < 0.05)]
gs = pd.read_csv(f'{B}/phase2_gene_corr/synapse_gene_corr_subcluster_full.csv',
                 usecols=['gene_symbol', 'cell_type', 'target_var', 'spearman_rho', 'fdr'])
gs = gs[gs.target_var == TV].copy()

e2s = {}
with gzip.open(GTF, 'rt') as f:
    for line in f:
        if line[0] == '#': continue
        p = line.split('\t')
        if p[2] != 'gene': continue
        i = p[8].find('gene_id "'); j = p[8].find('gene_name "')
        if i < 0 or j < 0: continue
        e2s[p[8][i+9:p[8].find('"', i+9)].split('.')[0]] = p[8][j+11:p[8].find('"', j+11)]
gs['gene_symbol'] = gs.gene_symbol.map(lambda x: e2s.get(str(x).split('.')[0], x))
gs = gs.rename(columns={'spearman_rho': 'rho_s', 'fdr': 'fdr_s'})
G = ga.merge(gs[['gene_symbol', 'cell_type', 'rho_s', 'fdr_s']], on=['gene_symbol', 'cell_type'])
G = G.dropna(subset=['spearman_rho', 'rho_s'])
G['same'] = np.sign(G.spearman_rho) == np.sign(G.rho_s)
G['robust'] = G.same & (G.fdr_s < 0.05)
G['gene_symbol'] = G.gene_symbol.str.upper()
w(f'\n遺伝子層: AIDA 有意 x Synapse 対応 {len(G):,} 行 / 両コホート頑健 {int(G.robust.sum()):,} 行')

ann = pd.read_csv(f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv',
                  usecols=['cre_id', 'proximity', 'enhancer_class']).drop_duplicates('cre_id')
ca = pd.read_csv(f'{OUR}/cre_corr_aida_subcluster_significant_{TS}.csv',
                 usecols=['cre_id', 'cell_type', 'target_var', 'spearman_rho', 'fdr', 'mean_expression'])
ca = ca[(ca.target_var == TV) & (ca.fdr < 0.05)]
cs = pd.read_csv(f'{OUR}/cre_corr_synapse_subcluster_full_{TS}.csv',
                 usecols=['cre_id', 'cell_type', 'target_var', 'spearman_rho', 'fdr'])
cs = cs[cs.target_var == TV].rename(columns={'spearman_rho': 'rho_s', 'fdr': 'fdr_s'})
C = ca.merge(cs[['cre_id', 'cell_type', 'rho_s', 'fdr_s']], on=['cre_id', 'cell_type'])
C = C.dropna(subset=['spearman_rho', 'rho_s'])
C = C.merge(ann, on='cre_id', how='left')
C['same'] = np.sign(C.spearman_rho) == np.sign(C.rho_s)
C['robust'] = C.same & (C.fdr_s < 0.05)
w(f'素子層: AIDA 有意 x Synapse 対応 {len(C):,} ペア(素子x細胞種) / '
  f'両コホート頑健 {int(C.robust.sum()):,} ペア = 実素子 {C[C.robust].cre_id.nunique():,} 個')

tss = {}
with gzip.open(GTF, 'rt') as f:
    for line in f:
        if line[0] == '#': continue
        p = line.split('\t')
        if p[2] != 'gene' or 'gene_biotype "protein_coding"' not in p[8]: continue
        i = p[8].find('gene_name "')
        if i < 0: continue
        gn = p[8][i+11:p[8].find('"', i+11)]
        ch = p[0] if p[0].startswith('chr') else 'chr' + p[0]
        tss.setdefault(ch, []).append((int(p[3]) if p[6] == '+' else int(p[4]), gn))
for ch in tss: tss[ch].sort()
POS = {ch: np.array([x[0] for x in v]) for ch, v in tss.items()}
def nearest(cid):
    try:
        ch, s, e, st = cid.rsplit('_', 3); mid = (int(s) + int(e)) // 2
    except Exception:
        return None, None
    a = tss.get(ch)
    if not a: return None, None
    i = int(np.searchsorted(POS[ch], mid)); best, bd = None, None
    for j in (i-1, i):
        if 0 <= j < len(a):
            d = abs(a[j][0] - mid)
            if bd is None or d < bd: bd, best = d, a[j][1]
    return best, bd

uni = pd.DataFrame({'cre_id': C.cre_id.unique()})
r = [nearest(c) for c in uni.cre_id]
uni['near_gene'] = [(x[0] or '').upper() for x in r]
uni['near_dist'] = [x[1] if x[1] is not None else np.nan for x in r]
C = C.merge(uni, on='cre_id', how='left')

e2g = pd.read_csv(f'{B}/re2g_targets_clean/targets_clean.csv',
                  usecols=['cre_id', 'target_gene'])
e2g['target_gene'] = e2g.target_gene.astype(str).str.upper()
e2g_map = e2g.groupby('cre_id').target_gene.apply(set).to_dict()
w(f'E2G リンクつき素子: {len(e2g_map):,}')

rows = []
for _, x in C.iterrows():
    gs_ = set()
    if x.near_gene: gs_.add((x.near_gene, 'near', x.near_dist))
    for t in e2g_map.get(x.cre_id, ()):
        gs_.add((t, 'e2g', np.nan))
    for gname, how, dist in gs_:
        if gname in ALL:
            rows.append(dict(gene=gname, link=how, dist=dist, cre_id=x.cre_id,
                             cell_type=x.cell_type, proximity=x.proximity,
                             enhancer_class=x.enhancer_class, rho_a=x.spearman_rho,
                             rho_s=x.rho_s, robust=x.robust, expr=x.mean_expression))
CG = pd.DataFrame(rows)
CG.to_csv(f'{OUT}/cre_by_topic_gene.csv', index=False)
w(f'\n話題遺伝子に紐づく素子x細胞種ペア: {len(CG):,} '
  f'(実素子 {CG.cre_id.nunique():,} / 遺伝子 {CG.gene.nunique()})')

for topic, genes in TOPICS.items():
    w('')
    w('=' * 108)
    w(f'■ {topic}')
    w('=' * 108)
    w(f'{"遺伝子":10s}{"細胞種":13s}{"遺伝子rhoA":>10s}{"rhoS":>7s}{"頑健":>5s}  '
      f'{"素子計":>6s}{"近位":>5s}{"遠位":>5s}{"頑健素子":>8s}  素子の向き（頑健のみ）')
    for g in genes:
        gg = G[(G.gene_symbol == g) & G.robust]
        cc = CG[CG.gene == g]
        if not len(gg) and not len(cc):
            w(f'{g:10s}(遺伝子・素子とも該当なし)'); continue
        cts = sorted(set(gg.cell_type) | set(cc.cell_type))
        for ct in cts:
            gr = gg[gg.cell_type == ct]
            cr = cc[cc.cell_type == ct]
            ra = f'{gr.spearman_rho.iloc[0]:+.2f}' if len(gr) else '  -  '
            rs = f'{gr.rho_s.iloc[0]:+.2f}' if len(gr) else '  -  '
            rb = '○' if len(gr) else ' '
            crr = cr[cr.robust]
            up = int((crr.rho_a > 0).sum()); dn = int((crr.rho_a < 0).sum())
            dsc = ''
            if len(crr):
                pu = int(((crr.proximity == 'proximal') & (crr.rho_a > 0)).sum())
                pd_ = int(((crr.proximity == 'proximal') & (crr.rho_a < 0)).sum())
                du = int(((crr.proximity == 'distal') & (crr.rho_a > 0)).sum())
                dd = int(((crr.proximity == 'distal') & (crr.rho_a < 0)).sum())
                dsc = f'近位 上{pu}/下{pd_}  遠位 上{du}/下{dd}'
                mx = crr.loc[crr.rho_a.abs().idxmax()]
                dsc += f'   最大 {mx.cre_id} ({mx.proximity[:4]}, {mx.rho_a:+.2f}/{mx.rho_s:+.2f})'
            mark = ' ★CD4' if ct.startswith('CD4') else ''
            w(f'{g:10s}{ct:13s}{ra:>10s}{rs:>7s}{rb:>5s}  '
              f'{len(cr):6d}{int((cr.proximity=="proximal").sum()):5d}'
              f'{int((cr.proximity=="distal").sum()):5d}{len(crr):8d}  {dsc}{mark}')

w('')
w('=' * 108)
w('■ 遺伝子の変化と、その遺伝子に紐づく素子の変化は一致するか（話題遺伝子に限らず全体）')
w('=' * 108)
allc = C[C.near_gene.notna() & (C.near_gene != '')].copy()
gk = G[['gene_symbol', 'cell_type', 'spearman_rho', 'robust']].rename(
    columns={'gene_symbol': 'near_gene', 'spearman_rho': 'g_rho', 'robust': 'g_robust'})
M = allc.merge(gk, on=['near_gene', 'cell_type'], how='left')
M['g_moved'] = M.g_robust.fillna(False)
w(f'\n頑健な素子x細胞種ペア {int(M.robust.sum()):,} のうち、最近接遺伝子も両コホート頑健なもの: '
  f'{int((M.robust & M.g_moved).sum()):,} ({100*M[M.robust].g_moved.mean():.1f}%)')
w(f'  -> 裏返すと {100*(1-M[M.robust].g_moved.mean()):.1f}% は「素子は動いたが最近接遺伝子は動いていない」')

w('\n近位素子と遠位素子で、遺伝子との符号一致はどう違うか（両方が頑健なペアのみ）')
both = M[M.robust & M.g_moved].copy()
both['agree'] = np.sign(both.spearman_rho) == np.sign(both.g_rho)
w(f'  {"クラス":10s}{"ペア数":>8s}{"遺伝子と同符号":>14s}')
for prox in ['proximal', 'distal']:
    x = both[both.proximity == prox]
    if len(x): w(f'  {prox:10s}{len(x):8,d}{100*x.agree.mean():13.1f}%')
if len(both[both.proximity == 'proximal']) and len(both[both.proximity == 'distal']):
    a = both[both.proximity == 'proximal']; b = both[both.proximity == 'distal']
    orr, pv = fisher_exact([[int(a.agree.sum()), int((~a.agree).sum())],
                            [int(b.agree.sum()), int((~b.agree).sum())]])
    w(f'  近位 vs 遠位: OR {orr:.2f}, p={pv:.3g}')
w('  -> 近位素子は遺伝子の転写開始点そのものなので高一致が当然。')
w('     遠位素子の一致率がそれより低ければ、遠位は遺伝子発現とは別の層で動いている。')

w('\n逆から見る: 頑健な遺伝子のうち、紐づく素子も頑健なものはどれくらいか')
gr = G[G.robust][['gene_symbol', 'cell_type']].drop_duplicates()
has = M[M.robust][['near_gene', 'cell_type']].drop_duplicates().rename(columns={'near_gene': 'gene_symbol'})
gr2 = gr.merge(has.assign(has_cre=True), on=['gene_symbol', 'cell_type'], how='left')
gr2['has_cre'] = gr2.has_cre.fillna(False)
w(f'  頑健な遺伝子x細胞種 {len(gr2):,} のうち、最近接に頑健素子を持つもの: '
  f'{int(gr2.has_cre.sum()):,} ({100*gr2.has_cre.mean():.1f}%)')
M.to_csv(f'{OUT}/cre_gene_joined.csv', index=False)

w('')
w('★注意: CD4_Memory と CD4_Naive は、遺伝子層が 2026-04-29 の subcluster ラベル、')
w('   素子層が 2026-07-24 の our-CD4 ラベルで、細胞集合が同一ではない。CD4 の行は参考値。')
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'\n-> {OUT}')


import pandas as pd, numpy as np, gzip, os

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
F = f'{D}/results/fixes'
OUT = f'{D}/results/gene_to_cre'
GTF = '${DATA_DIR}/adachi/reference/GRCh38.113.gtf.gz'
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

rob = pd.read_csv(f'{F}/cd4_donor_filter/robust_set_canon166.csv')
w(f'canon166 頑健集合: {len(rob):,} ペア / 異なり tCRE {rob.cre_id.nunique():,}')
cd4 = rob[rob.cell_type.isin(['CD4_Memory', 'CD4_Naive'])].copy()
w(f'CD4 の頑健ペア: {len(cd4):,} (CD4_Memory {int((cd4.cell_type=="CD4_Memory").sum())} / '
  f'CD4_Naive {int((cd4.cell_type=="CD4_Naive").sum())})')

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
def nearest(cid):
    try:
        ch, s, e, st = cid.rsplit('_', 3); mid = (int(s) + int(e)) // 2
    except Exception:
        return None, None
    a = tssd.get(ch)
    if not a: return None, None
    i = int(np.searchsorted(POS[ch], mid)); best, bd = None, None
    for j in (i - 1, i):
        if 0 <= j < len(a):
            dd = abs(a[j][0] - mid)
            if bd is None or dd < bd: bd, best = dd, a[j][1]
    return best, bd
r_ = [nearest(c) for c in cd4.cre_id]
cd4['near_gene'] = [(x[0] or '').upper() for x in r_]
cd4['near_dist'] = [x[1] if x[1] is not None else np.nan for x in r_]

e2g = pd.read_csv(f'{B}/re2g_targets_clean/targets_clean.csv',
                  usecols=['cre_id', 'target_gene'])
e2g['target_gene'] = e2g.target_gene.astype(str).str.upper()
e2g_map = e2g.groupby('cre_id').target_gene.apply(set).to_dict()

TOPICS = {
    '身元と TCR 入口': ['LCK', 'PTPRC', 'PRKCQ', 'CCR7', 'CD7', 'TCF7', 'LEF1', 'SATB1',
                        'SELL', 'BCL11B', 'IKZF1', 'FOXO1', 'ETS1', 'LRRN3', 'MYC', 'ATM', 'G6PD'],
    '分化 TF': ['ZEB2', 'PRDM1', 'KLF6', 'KLF2', 'KLF4', 'KLF14', 'RORA', 'BHLHE40', 'RUNX3', 'TBX21'],
    'ISR/SG': ['HSPA5', 'ATF4', 'PPP1R15A', 'EIF2AK2', 'PABPC1', 'ATXN2L', 'UBAP2L', 'CSDE1',
               'DDX3X', 'YBX1', 'NUFIP2', 'DDIT4', 'XBP1', 'CAPRIN1', 'G3BP1', 'G3BP2', 'TIA1', 'TIAL1'],
    'NFkB/AP-1': ['NFKBIA', 'TNFAIP3', 'ZFP36', 'ZC3H12A', 'BIRC3', 'FOS', 'FOSB', 'JUN', 'JUNB',
                  'ATF3', 'DUSP1', 'NR4A2', 'EGR1'],
    '細胞傷害/アポトーシス': ['IL15', 'SRGN', 'GZMK', 'GZMB', 'IFNG', 'NKG7', 'PRF1', 'KLRG1',
                              'BCL2L11', 'PMAIP1', 'MCL1', 'BCL2', 'BAX'],
}
ALL = sorted({g for v in TOPICS.values() for g in v})

rows = []
for _, x in cd4.iterrows():
    genes = set()
    if x.near_gene: genes.add((x.near_gene, 'near'))
    for t in e2g_map.get(x.cre_id, ()):
        genes.add((t, 'e2g'))
    for gname, how in genes:
        if gname in ALL:
            rows.append(dict(gene=gname, link=how, cre_id=x.cre_id, cell_type=x.cell_type,
                             proximity=x.proximity, dist=x.near_dist if how == 'near' else np.nan,
                             rho_a=x.rho_aida, rho_s=x.rho_syn, direction=x.direction))
CG = pd.DataFrame(rows).drop_duplicates(['gene', 'cre_id', 'cell_type'])
CG.to_csv(f'{OUT}/cd4_canonA_update.csv', index=False)

w('\n' + '=' * 100)
w('CD4 の話題遺伝子 × 頑健制御素子 (canon166 = タグ A)。旧 4,598 定義の行はこの表で置き換える')
w('=' * 100)
for topic, gs in TOPICS.items():
    x = CG[CG.gene.isin(gs)]
    if not len(x): continue
    w(f'\n■ {topic}  ({x.cre_id.nunique()} tCRE / {len(x)} 行)')
    w(f'{"遺伝子":10s}{"細胞種":13s}{"素子":32s}{"位置":9s}{"向き":5s}{"rhoA":>7s}{"rhoS":>7s}{"距離":>10s}')
    for _, r in x.sort_values(['gene', 'cell_type']).iterrows():
        d = f'{int(r.dist):,}' if pd.notna(r.dist) else 'E2G'
        w(f'{r.gene:10s}{r.cell_type:13s}{r.cre_id:32s}{str(r.proximity)[:8]:9s}'
          f'{"上" if r.direction.startswith("up") else "下":5s}{r.rho_a:+7.2f}{r.rho_s:+7.2f}{d:>10s}')

old = pd.read_csv(f'{OUT}/cre_by_topic_gene_dedup.csv')
old_cd4 = old[old.cell_type.isin(['CD4_Memory', 'CD4_Naive']) & old.robust]
w('\n' + '=' * 100)
w('新旧比較 (頑健のみ)')
w('=' * 100)
w(f'  旧 (4,598 定義): CD4 頑健 {len(old_cd4)} 行 / {old_cd4.cre_id.nunique()} tCRE')
w(f'  新 (canon166) : CD4 頑健 {len(CG)} 行 / {CG.cre_id.nunique()} tCRE')
oldkey = set(zip(old_cd4.gene, old_cd4.cre_id, old_cd4.cell_type))
newkey = set(zip(CG.gene, CG.cre_id, CG.cell_type))
lost = oldkey - newkey; kept = oldkey & newkey; new = newkey - oldkey
w(f'  維持 {len(kept)} / 失われた {len(lost)} / 新規 {len(new)}')
if lost:
    w('\n  失われた行 (旧定義でのみ頑健だったもの):')
    for g, c, ct in sorted(lost):
        w(f'    {g:10s} {ct:12s} {c}')
key_genes = ['LCK', 'PTPRC', 'CCR7', 'CD7', 'PRKCQ', 'LEF1', 'ATM']
w('\n  ★柱 3 の名指し遺伝子の生存状況 (CD4 2 群):')
for g in key_genes:
    o = old_cd4[old_cd4.gene == g].cre_id.nunique()
    n = CG[CG.gene == g].cre_id.nunique()
    w(f'    {g:8s} 旧 {o} tCRE -> 新 {n} tCRE  {"[維持]" if n>0 else "[★消失]"}')
open(f'{OUT}/report_cd4_canonA.txt', 'w').write('\n'.join(L))
print(f'\n-> {OUT}/report_cd4_canonA.txt')
