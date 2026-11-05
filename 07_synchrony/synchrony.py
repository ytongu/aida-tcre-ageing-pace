#!/usr/bin/env python3
import pandas as pd, numpy as np, os
P='${PROJECT_DIR}/results/phase3b_ourcd4'
CLS='${PROJECT_DIR}/results/phase6_novel_enhancer/novel_enhancer_classification.csv'
OUT='${PAPER_DIR}/results/fixes/coordination_redo'
TV='yugene_diff_EN_tAge'; TS='_v4'

def load(fn, tag):
    keep=[]
    for ch in pd.read_csv(fn, usecols=['cre_id','cell_type','target_var','n_samples','mean_expression','spearman_rho','fdr'],
                          chunksize=3_000_000):
        ch=ch[ch.target_var==TV]
        if len(ch): keep.append(ch.drop(columns=['target_var']))
    d=pd.concat(keep,ignore_index=True)
    d=d.rename(columns={'mean_expression':f'expr_{tag}','spearman_rho':f'rho_{tag}','fdr':f'fdr_{tag}','n_samples':f'n_{tag}'})
    print(tag, d.shape, d.cell_type.nunique(), flush=True)
    return d

a=load(f'{P}/cre_corr_aida_subcluster_full_{TS}.csv','a')
s=load(f'{P}/cre_corr_synapse_subcluster_full_{TS}.csv','s')
m=a.merge(s[['cre_id','cell_type','n_s','expr_s','rho_s','fdr_s']],on=['cre_id','cell_type'],how='outer')
cls=pd.read_csv(CLS,usecols=['cre_id','proximity','enhancer_class'])
m=m.merge(cls,on='cre_id',how='left')
print('merged',m.shape,flush=True)
print(m.groupby('cell_type').size(),flush=True)
m.to_csv(f'{OUT}/tables/corr_merged_by_celltype.csv.gz',index=False)
print('WROTE',f'{OUT}/tables/corr_merged_by_celltype.csv.gz',flush=True)


import pickle, os, sys, time
import numpy as np, pandas as pd

OUT='${PAPER_DIR}/results/fixes/coordination_redo'
SCR='${SCRATCH}'
P='${PROJECT_DIR}/results/phase3b_ourcd4'
CTS=['NK','CD4_Naive','CD4_Memory','CD8_Naive','B_Naive','NKT','CD8_Effector','CD16_Mono']
MIN_CELLS=10
os.makedirs(f'{SCR}/mat',exist_ok=True)

rows=[]
for cohort,cache in [('AIDA',f'{P}/aggregation_cache_aida_subcluster.pkl'),
                     ('Synapse',f'{P}/aggregation_cache_synapse_subcluster.pkl')]:
    t0=time.time()
    print(f'[{cohort}] loading cache ...',flush=True)
    with open(cache,'rb') as f: agg=pickle.load(f)
    cres=agg['cres']; es=agg['expr_sum']; us=agg['umi_sum']; nc=agg['n_cells']
    print(f'  loaded {time.time()-t0:.0f}s  n_cre={len(cres)}  n_key={len(es)}',flush=True)
    ks=list(es.keys())[:3]
    print('  sample keys:',ks,flush=True)
    print('  cohort values:',sorted({k[0] for k in es})[:5],flush=True)
    print('  celltypes:',sorted({k[3] for k in es}),flush=True)
    np.save(f'{SCR}/mat/cres.npy',np.array(cres,dtype=object),allow_pickle=True)
    for CT in CTS:
        acc={}
        for key,cnt in es.items():
            if key[0]!=cohort or key[3]!=CT: continue
            n=nc.get(key,0); u=us.get(key,0)
            if n<MIN_CELLS or u<=0: continue
            d=key[1]
            cpm=(cnt.astype(np.float64)/u*1e6).astype(np.float32)
            if d not in acc: acc[d]=[cpm,1,int(n),int(u)]
            else:
                a=acc[d]; a[0]=a[0]+cpm; a[1]+=1; a[2]+=int(n); a[3]+=int(u)
        donors=sorted(acc)
        if not donors:
            print(f'  {cohort} {CT}: NO DONORS',flush=True); continue
        X=np.vstack([acc[d][0]/acc[d][1] for d in donors]).astype(np.float32)
        meta=pd.DataFrame({'donor':donors,
                           'n_batch':[acc[d][1] for d in donors],
                           'n_cells':[acc[d][2] for d in donors],
                           'umi':[acc[d][3] for d in donors]})
        np.save(f'{SCR}/mat/{cohort}_{CT}_X.npy',X)
        meta.to_csv(f'{SCR}/mat/{cohort}_{CT}_meta.csv',index=False)
        rows.append(dict(cohort=cohort,cell_type=CT,n_donor=len(donors),
                         n_key=int(meta.n_batch.sum()),
                         median_ncells=float(meta.n_cells.median()),
                         median_umi=float(meta.umi.median())))
        print(f'  {cohort} {CT}: {X.shape[0]} donors x {X.shape[1]} CRE  (keys {int(meta.n_batch.sum())})  {time.time()-t0:.0f}s',flush=True)
        del acc,X
    del agg,es,us,nc
    import gc; gc.collect()
pd.DataFrame(rows).to_csv(f'{OUT}/tables/extract_donor_counts.csv',index=False)
print('DONE',flush=True)


import os, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
OUT='${PAPER_DIR}/results/fixes/coordination_redo'
SCR='${SCRATCH}'
SYN='${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/subcluster/tage_all_combined_v2.csv'
AID='${DATA_DIR}/lnc_project_AIDA_dataset/by_subcluster/tage/tage_all_combined_v1.csv'
CD4='${PAPER_DIR}/results/cd4_redo/cd4_our_tage.csv'
CTS=['NK','CD4_Naive','CD4_Memory','CD8_Naive','B_Naive','NKT','CD8_Effector','CD16_Mono']
TV='yugene_diff_EN_tAge'

s=pd.read_csv(SYN); s=s[(s.model_type=='Chronoage')&(s['mode']=='EN')]
s=s[['Donor_id','CellType',TV]].rename(columns={'CellType':'cell_type'}); s['cohort']='Synapse'
a=pd.read_csv(AID); a=a[(a.model_type=='Chronoage')&(a['mode']=='EN')]
a=a[['Sample','Subcluster',TV]].rename(columns={'Sample':'Donor_id','Subcluster':'cell_type'}); a['cohort']='AIDA'
c=pd.read_csv(CD4)[['cohort','Donor_id','Group',TV]].rename(columns={'Group':'cell_type'})
base=pd.concat([s,a],ignore_index=True)
base=base[~base.cell_type.isin(['CD4_Naive','CD4_Memory'])]
tage=pd.concat([base,c],ignore_index=True)
tage=tage.groupby(['cohort','cell_type','Donor_id'],as_index=False)[TV].mean()
tage.to_csv(f'{OUT}/tables/tage_donor_level.csv',index=False)
print('tage donor-level',tage.shape,flush=True)

def rz(x):
    r=rankdata(x); return (r-r.mean())/(r.std()+1e-12)
def residualize(X,ncell,umi):
    Y=rankdata(X,axis=0).astype(np.float64)
    Y-=Y.mean(0); sd=Y.std(0); sd[sd<1e-12]=1e-12; Y/=sd
    C=np.column_stack([np.ones(len(ncell)),rz(np.log10(ncell+1)),rz(np.log10(umi/ncell+1))])
    B=np.linalg.lstsq(C,Y,rcond=None)[0]
    return (Y-C@B).astype(np.float32)

cres=np.load(f'{SCR}/mat/cres.npy',allow_pickle=True)
corr=pd.read_csv(f'{OUT}/tables/corr_merged_by_celltype.csv.gz')
rows=[]; gi_rows=[]
for cohort,tag in [('AIDA','a'),('Synapse','s')]:
    for CT in CTS:
        fx=f'{SCR}/mat/{cohort}_{CT}_X.npy'
        if not os.path.exists(fx): continue
        X=np.load(fx); meta=pd.read_csv(f'{SCR}/mat/{cohort}_{CT}_meta.csv')
        t=tage[(tage.cohort==cohort)&(tage.cell_type==CT)].set_index('Donor_id')[TV]
        keep=np.array([d in t.index for d in meta.donor])
        X=X[keep]; meta=meta[keep].reset_index(drop=True)
        y=t.loc[meta.donor].values
        R=residualize(X,meta.n_cells.values.astype(float),meta.umi.values.astype(float))

        C=np.column_stack([np.ones(len(meta)),rz(np.log10(meta.n_cells.values+1.0)),
                           rz(np.log10(meta.umi.values/meta.n_cells.values+1.0))])
        yz=rz(y); yr=yz-C@np.linalg.lstsq(C,yz,rcond=None)[0]
        yr=yr/ (np.linalg.norm(yr)+1e-12)
        nrm=np.linalg.norm(R,axis=0); nrm[nrm<1e-12]=1e-12
        g=(R.T@yr)/nrm
        can=corr[corr.cell_type==CT].set_index('cre_id')
        cser=can[f'rho_{tag}'].reindex(cres).values
        ns=can[f'n_{tag}'].max()
        ok=np.isfinite(cser)&np.isfinite(g)
        rho_agree=spearmanr(g[ok],cser[ok])[0]
        rows.append(dict(cohort=cohort,cell_type=CT,n_donor_cache=int(len(meta)),
                         n_donor_with_tage=int(keep.sum()),n_samples_canonical=int(ns),
                         n_cre_compared=int(ok.sum()),
                         spearman_g_vs_canonical=float(rho_agree),
                         mean_g=float(np.nanmean(g)),
                         detect_median=float(np.mean(X>0,axis=0).mean())))
        print(rows[-1],flush=True)
        np.save(f'{SCR}/mat/{cohort}_{CT}_g.npy',g)
        np.save(f'{SCR}/mat/{cohort}_{CT}_detect.npy',np.mean(X>0,axis=0).astype(np.float32))
        np.save(f'{SCR}/mat/{cohort}_{CT}_R.npy',R)
        meta.to_csv(f'{SCR}/mat/{cohort}_{CT}_meta_used.csv',index=False)
        np.save(f'{SCR}/mat/{cohort}_{CT}_yr.npy',yr)
        if CT=='NK':
            gi_rows.append(pd.DataFrame({'cohort':cohort,'cell_type':CT,'cre_id':cres,
                                         'g_recomputed':g,'rho_canonical':cser}))
        del X,R
pd.DataFrame(rows).to_csv(f'{OUT}/tables/gate_donor_counts.csv',index=False)
if gi_rows: pd.concat(gi_rows).to_csv(f'{OUT}/tables/gate_gi_vs_canonical_NK.csv.gz',index=False)
print('GATE DONE',flush=True)


import os, sys, json, zlib, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.spatial import cKDTree
OUT='${PAPER_DIR}/results/fixes/coordination_redo'
SCR='${SCRATCH}'
ENH={'unidirectional_enhancer','divergent_enhancer','convergent_other_distal'}
CTS=['NK','CD4_Naive','CD4_Memory','CD8_Naive','B_Naive','NKT','CD8_Effector','CD16_Mono']
K=733; NDRAW=20; NPERM_SEL=200; NPERM_NOISE=10
rng=np.random.default_rng(_v5)

cres=np.load(f'{SCR}/mat/cres.npy',allow_pickle=True); cres=np.asarray(cres,dtype=object)
pos={c:i for i,c in enumerate(cres)}
corr=pd.read_csv(f'{OUT}/tables/corr_merged_by_celltype.csv.gz')
corr['is_de']=(corr.proximity=='distal')&corr.enhancer_class.isin(ENH)

def load_ct(CT):
    d={}
    for coh in ['AIDA','Synapse']:
        R=np.load(f'{SCR}/mat/{coh}_{CT}_R.npy')
        g=np.load(f'{SCR}/mat/{coh}_{CT}_g.npy')
        det=np.load(f'{SCR}/mat/{coh}_{CT}_detect.npy')
        yr=np.load(f'{SCR}/mat/{coh}_{CT}_yr.npy')
        nrm=np.linalg.norm(R,axis=0)
        d[coh]=dict(R=R,g=g,det=det,yr=yr,nrm=nrm,n=R.shape[0])
    return d

def offdiag(C):
    iu=np.triu_indices_from(C,1); return C[iu]

def cmat(R,idx):
    S=R[:,idx].astype(np.float64)
    nr=np.linalg.norm(S,axis=0); nr[nr<1e-12]=np.nan
    S=S/nr
    C=S.T@S
    np.fill_diagonal(C,np.nan)
    return C

def pc1_corr(C):
    A=np.array(C,copy=True); np.fill_diagonal(A,1.0)
    A=np.nan_to_num(A,nan=0.0)
    w=np.linalg.eigvalsh(A)
    return float(w[-1]/A.shape[0])

def pc1_svd(R,idx):
    M=R[:,idx].astype(np.float64); M=M-M.mean(0)
    sv=np.linalg.svd(M,compute_uv=False)
    return float(sv[0]**2/ (sv**2).sum())

def metrics(d,idx,label,extra=None,do_null=True,seed=0):
    r=dict(label=label,n_cre=len(idx))
    oo={}
    for coh in ['AIDA','Synapse']:
        C=cmat(d[coh]['R'],idx); o=offdiag(C); oo[coh]=o
        g=d[coh]['g'][idx]
        gg=np.outer(g,g); ogg=offdiag(gg)
        P=(C-gg)/np.sqrt(np.outer(1-g**2,1-g**2))
        oP=offdiag(P)
        r[f'mean_r_{coh}']=float(np.nanmean(o))
        r[f'mean_absr_{coh}']=float(np.nanmean(np.abs(o)))
        r[f'mean_gg_{coh}']=float(np.nanmean(ogg))
        r[f'excess_r_{coh}']=float(np.nanmean(o)-np.nanmean(ogg))
        r[f'mean_absr_minus_absgg_{coh}']=float(np.nanmean(np.abs(o))-np.nanmean(np.abs(ogg)))
        r[f'mean_partial_r_{coh}']=float(np.nanmean(oP))
        r[f'mean_abs_partial_r_{coh}']=float(np.nanmean(np.abs(oP)))
        r[f'pc1_corr_{coh}']=pc1_corr(C)
        r[f'pc1_svd_{coh}']=pc1_svd(d[coh]['R'],idx)
        r[f'mean_g_{coh}']=float(np.nanmean(g))
        r[f'n_donor_{coh}']=int(d[coh]['n'])
        r[f'n_nan_pairs_{coh}']=int(np.isnan(o).sum())
        oo[coh+'_P']=oP
    ok=np.isfinite(oo['AIDA'])&np.isfinite(oo['Synapse'])
    r['repro_r']=float(spearmanr(oo['AIDA'][ok],oo['Synapse'][ok])[0])
    okP=np.isfinite(oo['AIDA_P'])&np.isfinite(oo['Synapse_P'])
    r['repro_partial_r']=float(spearmanr(oo['AIDA_P'][okP],oo['Synapse_P'][okP])[0])
    if do_null:
        rr=np.random.default_rng(1000+seed)
        f1=[];f2=[];f3=[];f4=[]
        for _ in range(NPERM_NOISE):
            perm={}
            for coh in ['AIDA','Synapse']:
                S=d[coh]['R'][:,idx].copy()
                for j in range(S.shape[1]): S[:,j]=S[rr.permutation(S.shape[0]),j]
                Cn=cmat(S,np.arange(S.shape[1])); perm[coh]=offdiag(Cn)
                if coh=='AIDA':
                    f1.append(np.nanmean(np.abs(perm[coh]))); f3.append(pc1_corr(Cn))
                else: f2.append(np.nanmean(np.abs(perm[coh])))
            m=np.isfinite(perm['AIDA'])&np.isfinite(perm['Synapse'])
            f4.append(spearmanr(perm['AIDA'][m],perm['Synapse'][m])[0])
        r['null_absr_AIDA']=float(np.mean(f1)); r['null_absr_Synapse']=float(np.mean(f2))
        r['null_pc1_corr_AIDA']=float(np.mean(f3)); r['null_repro_r']=float(np.mean(f4))
    if extra: r.update(extra)
    return r,oo

def match_sets(cand_idx,fg_idx,feat,k=10,ndraw=NDRAW,seed=0):
    rr=np.random.default_rng(5000+seed)
    tree=cKDTree(feat[cand_idx])
    out=[]
    for dw in range(ndraw):
        order=rr.permutation(len(fg_idx)); used=set(); sel=[]
        kq=min(len(cand_idx),max(60,int(1.5*len(fg_idx))))
        dist,nb=tree.query(feat[fg_idx],k=kq)
        for i in order:
            cands=[c for c in nb[i] if c not in used]
            if not cands: continue
            pick=rr.choice(cands[:k]) if len(cands)>=k else cands[0]
            used.add(int(pick)); sel.append(cand_idx[int(pick)])
        out.append(np.array(sorted(sel)))
    return out

summary=[]; setrows=[]; nullrows=[]; elemrows=[]
log=lambda *a: print(*a,flush=True)

CT='NK'; d=load_ct(CT)
sub=corr[corr.cell_type==CT].set_index('cre_id')
expr_a=sub['expr_a'].reindex(cres).values; fdr_a=sub['fdr_a'].reindex(cres).values
rho_a=sub['rho_a'].reindex(cres).values;  fdr_s=sub['fdr_s'].reindex(cres).values
rho_s=sub['rho_s'].reindex(cres).values;  isde=sub['is_de'].reindex(cres).fillna(False).values.astype(bool)
prox=sub['proximity'].reindex(cres).values
AE=pd.read_csv('${PAPER_DIR}/results/cd4_redo/aging_enhancers_all.csv')
fg733=np.array(sorted({pos[c] for c in AE[AE.cell_type==CT].cre_id if c in pos}))
log(f'版0 前景 (公表 733): {len(fg733)}')

poolmask=isde&(~np.isin(np.arange(len(cres)),fg733))&(np.nan_to_num(expr_a)>0)&(np.nan_to_num(fdr_a,nan=1.0)>=0.05)
pool0=np.where(poolmask)[0]
fgl=fg733[np.nan_to_num(expr_a[fg733])>0]
pa=np.log10(expr_a[pool0]); order=np.argsort(pa); pa=pa[order]; pc=pool0[order]
used=np.zeros(len(pc),bool); bg0=[]
for c in fgl:
    t=np.log10(expr_a[c]); i=np.searchsorted(pa,t); best=-1;bd=1e9
    for j in range(max(0,i-40),min(len(pc),i+40)):
        if used[j]: continue
        dd=abs(pa[j]-t)
        if dd<bd: bd=dd;best=j
    if best>=0: used[best]=True; bg0.append(pc[best])
bg0=np.array(sorted(bg0))
log(f'版0 背景: {len(bg0)}  (公表 733)')
log(f'  発現量中央値 fg={np.median(expr_a[fgl]):.2f} bg={np.median(expr_a[bg0]):.2f}')
log(f'  背景のうち Synapse FDR<0.05: {(np.nan_to_num(fdr_s[bg0],nan=1)<0.05).sum()} / 前景: {(np.nan_to_num(fdr_s[fgl],nan=1)<0.05).sum()}')
for nm,ii in [('v0_fg_aging733',fgl),('v0_bg_origmatch',bg0)]:
    m,_=metrics(d,ii,nm,extra=dict(version='v0_original',cell_type=CT,group=nm.split('_')[1]))
    summary.append(m); log(nm, {k2:round(v,4) for k2,v in m.items() if isinstance(v,float)})
    setrows.append(pd.DataFrame(dict(version='v0_original',cell_type=CT,group=nm,cre_id=cres[ii],
        rho_a=rho_a[ii],fdr_a=fdr_a[ii],rho_s=rho_s[ii],fdr_s=fdr_s[ii],expr_a=expr_a[ii],
        detect_A=d['AIDA']['det'][ii],detect_S=d['Synapse']['det'][ii],
        g_A=d['AIDA']['g'][ii],g_S=d['Synapse']['g'][ii],proximity=prox[ii])))

nrmA0=d['AIDA']['nrm']; nrmS0=d['Synapse']['nrm']
valid0=(nrmA0>0.1*np.sqrt(d['AIDA']['n']))&(nrmS0>0.1*np.sqrt(d['Synapse']['n']))&np.isfinite(rho_a)&np.isfinite(rho_s)
pool0v=np.where(isde&valid0&(np.nan_to_num(fdr_a,nan=1.0)>=0.05))[0]
lg0=np.log10(np.nan_to_num(expr_a,nan=0)+1e-3); dt0=d['AIDA']['det']
def _z(v,m): return (v-np.nanmean(v[m]))/(np.nanstd(v[m])+1e-12)
feat2=np.column_stack([_z(lg0,valid0),_z(dt0,valid0)])
feat1=np.column_stack([_z(lg0,valid0),np.zeros(len(lg0))])
log(f'版0b 前景 733 の検出率(AIDA)={dt0[fgl].mean():.3f} / 原著背景={dt0[bg0].mean():.3f}')
for tag,ft in [('bg_match_exprOnly',feat1),('bg_match_expr_detect',feat2)]:
    bgs=match_sets(pool0v,fgl,ft,ndraw=NDRAW,seed=zlib.crc32(tag.encode())%1000)
    acc=[]
    for bi,bg in enumerate(bgs):
        mb,_=metrics(d,bg,f'v0b_{tag}_{bi}',do_null=(bi==0),
                     extra=dict(version='v0b_matchvariants',cell_type=CT,group=tag,draw=bi))
        acc.append(mb)
        if bi==0:
            setrows.append(pd.DataFrame(dict(version='v0b_matchvariants',cell_type=CT,group=tag+'_draw0',
                cre_id=cres[bg],rho_a=rho_a[bg],fdr_a=fdr_a[bg],rho_s=rho_s[bg],fdr_s=fdr_s[bg],expr_a=expr_a[bg],
                detect_A=d['AIDA']['det'][bg],detect_S=d['Synapse']['det'][bg],
                g_A=d['AIDA']['g'][bg],g_S=d['Synapse']['g'][bg],proximity=prox[bg])))
    A=pd.DataFrame(acc); num=A.select_dtypes(include=[np.number]); mm=num.mean().to_dict(); sd=num.std().to_dict()
    mm.update(dict(label=f'v0b_{tag}_mean',version='v0b_matchvariants',cell_type=CT,group=tag,n_draw=len(acc)))
    for kk in ['mean_r_AIDA','mean_absr_AIDA','pc1_corr_AIDA','pc1_svd_AIDA','repro_r','repro_partial_r',
               'mean_absr_Synapse','excess_r_AIDA']:
        mm[kk+'_sd']=sd.get(kk,np.nan)
    summary.append(mm); A.to_csv(f'{OUT}/tables/v0b_{tag}_draws.csv',index=False)
    b0=bgs[0]
    log(f'  {tag}: n={len(b0)} log10expr fg={np.nanmedian(lg0[fgl]):.3f} bg={np.nanmedian(lg0[b0]):.3f} '
        f'検出率 fg={dt0[fgl].mean():.3f} bg={dt0[b0].mean():.3f}')
    log(f'  {tag}: mean_r_A={mm["mean_r_AIDA"]:.3f}+-{mm["mean_r_AIDA_sd"]:.3f} |r|A={mm["mean_absr_AIDA"]:.3f} '
        f'|r|S={mm["mean_absr_Synapse"]:.3f} PC1svd={mm["pc1_svd_AIDA"]:.3f} '
        f'repro={mm["repro_r"]:.3f}+-{mm["repro_r_sd"]:.3f} repro_partial={mm["repro_partial_r"]:.3f}')

def run_v1(CT,d):
    sub=corr[corr.cell_type==CT].set_index('cre_id')
    expr_a=sub['expr_a'].reindex(cres).values; fdr_a=np.nan_to_num(sub['fdr_a'].reindex(cres).values,nan=1.0)
    rho_a=sub['rho_a'].reindex(cres).values;   fdr_s=np.nan_to_num(sub['fdr_s'].reindex(cres).values,nan=1.0)
    rho_s=sub['rho_s'].reindex(cres).values;   isde=sub['is_de'].reindex(cres).fillna(False).values.astype(bool)
    prox=sub['proximity'].reindex(cres).values
    nrmA=d['AIDA']['nrm']; nrmS=d['Synapse']['nrm']
    valid=(nrmA>0.1*np.sqrt(d['AIDA']['n']))&(nrmS>0.1*np.sqrt(d['Synapse']['n']))&np.isfinite(rho_a)&np.isfinite(rho_s)
    hits=np.where(isde&valid&(fdr_a<0.05)&(rho_a>0))[0]
    pool=np.where(isde&valid&(fdr_a>=0.05))[0]
    log(f'[{CT}] v1 hits(up,distal-enh,AIDA FDR<0.05)={len(hits)}  pool(non-hit)={len(pool)}')
    if len(hits)<200: return
    k=min(K,len(hits))
    top=hits[np.argsort(fdr_a[hits],kind='stable')][:k]
    rnd=rng.choice(hits,size=k,replace=False)
    lg=np.log10(np.nan_to_num(expr_a,nan=0)+1e-3); dt=d['AIDA']['det']
    feat=np.column_stack([(lg-np.nanmean(lg[valid]))/np.nanstd(lg[valid]),
                          (dt-dt[valid].mean())/(dt[valid].std()+1e-12)])
    res={}
    for nm,fgi in [('fg_top','v1_fg_top'),('fg_rand','v1_fg_rand')]:
        ii=top if nm=='fg_top' else rnd
        m,oo=metrics(d,ii,f'{CT}_{fgi}',extra=dict(version='v1_AIDAonly',cell_type=CT,group=fgi))
        summary.append(m); res[nm]=(ii,m)
        setrows.append(pd.DataFrame(dict(version='v1_AIDAonly',cell_type=CT,group=fgi,cre_id=cres[ii],
            rho_a=rho_a[ii],fdr_a=fdr_a[ii],rho_s=rho_s[ii],fdr_s=fdr_s[ii],expr_a=expr_a[ii],
            detect_A=d['AIDA']['det'][ii],detect_S=d['Synapse']['det'][ii],
            g_A=d['AIDA']['g'][ii],g_S=d['Synapse']['g'][ii],proximity=prox[ii])))
        log(f'  {CT} {fgi}: mean_r_A={m["mean_r_AIDA"]:.3f} |r|A={m["mean_absr_AIDA"]:.3f} '
            f'(床 {m["null_absr_AIDA"]:.3f}) PC1={m["pc1_corr_AIDA"]:.3f} repro={m["repro_r"]:.3f} '
            f'repro_partial={m["repro_partial_r"]:.3f}')

        bgs=match_sets(pool,ii,feat,ndraw=NDRAW,seed=zlib.crc32((CT+fgi).encode())%1000)
        acc=[]
        for bi,bg in enumerate(bgs):
            mb,_=metrics(d,bg,f'{CT}_{fgi}_bg{bi}',do_null=(bi==0),
                         extra=dict(version='v1_AIDAonly',cell_type=CT,group=f'{fgi}_bg',draw=bi))
            acc.append(mb)
            if bi==0:
                setrows.append(pd.DataFrame(dict(version='v1_AIDAonly',cell_type=CT,group=f'{fgi}_bg_draw0',
                    cre_id=cres[bg],rho_a=rho_a[bg],fdr_a=fdr_a[bg],rho_s=rho_s[bg],fdr_s=fdr_s[bg],
                    expr_a=expr_a[bg],detect_A=d['AIDA']['det'][bg],detect_S=d['Synapse']['det'][bg],
                    g_A=d['AIDA']['g'][bg],g_S=d['Synapse']['g'][bg],proximity=prox[bg])))
        A=pd.DataFrame(acc); num=A.select_dtypes(include=[np.number])
        mm=num.mean().to_dict(); sd=num.std().to_dict()
        mm.update(dict(label=f'{CT}_{fgi}_bg_mean',version='v1_AIDAonly',cell_type=CT,
                       group=f'{fgi}_bg',n_draw=len(acc)))
        for kk in ['mean_r_AIDA','mean_absr_AIDA','pc1_corr_AIDA','repro_r','repro_partial_r',
                   'mean_r_Synapse','mean_absr_Synapse','mean_abs_partial_r_AIDA','excess_r_AIDA']:
            mm[kk+'_sd']=sd.get(kk,np.nan)
        summary.append(mm)
        A.to_csv(f'{OUT}/tables/bgdraws_v1_{CT}_{fgi}.csv',index=False)
        log(f'  {CT} {fgi} 背景{len(acc)}回平均: mean_r_A={mm["mean_r_AIDA"]:.3f}+-{mm["mean_r_AIDA_sd"]:.3f} '
            f'|r|A={mm["mean_absr_AIDA"]:.3f} PC1={mm["pc1_corr_AIDA"]:.3f} repro={mm["repro_r"]:.3f}+-{mm["repro_r_sd"]:.3f} '
            f'repro_partial={mm["repro_partial_r"]:.3f}')

        b0=bgs[0]
        log(f'    マッチ品質 draw0: log10expr fg={np.nanmedian(lg[ii]):.3f} bg={np.nanmedian(lg[b0]):.3f} | '
            f'検出率 fg={dt[ii].mean():.3f} bg={dt[b0].mean():.3f}')

        ref=np.concatenate([ii,b0]); C=cmat(d['AIDA']['R'],ref)
        mi=np.nanmean(C,axis=1); lab=np.array([1]*len(ii)+[0]*len(b0))
        obs=mi[lab==1].mean()-mi[lab==0].mean()
        rr2=np.random.default_rng(77); cnt=0; NP=10000
        for _ in range(NP):
            p=rr2.permutation(lab); cnt+= (mi[p==1].mean()-mi[p==0].mean())>=obs
        bs=[]; rr3=np.random.default_rng(78)
        for _ in range(2000):
            i1=rr3.integers(0,len(ii),len(ii)); i0=rr3.integers(len(ii),len(ref),len(b0))
            bs.append(mi[i1].mean()-mi[i0].mean())
        elemrows.append(dict(version='v1_AIDAonly',cell_type=CT,group=fgi,
            n_fg=len(ii),n_bg=len(b0),mean_mi_fg=float(mi[lab==1].mean()),mean_mi_bg=float(mi[lab==0].mean()),
            diff=float(obs),perm_p=float((cnt+1)/(NP+1)),
            boot_lo=float(np.percentile(bs,2.5)),boot_hi=float(np.percentile(bs,97.5))))
        log(f'    素子単位検定: fg平均r={mi[lab==1].mean():.4f} bg={mi[lab==0].mean():.4f} '
            f'差={obs:.4f} 置換p={(cnt+1)/(NP+1):.2e} boot95%=[{np.percentile(bs,2.5):.4f},{np.percentile(bs,97.5):.4f}]')
        pd.DataFrame(dict(cre_id=cres[ref],group=np.where(lab==1,'fg','bg'),mean_r_vs_ref=mi)).to_csv(
            f'{OUT}/tables/elementwise_meanr_v1_{CT}_{fgi}.csv',index=False)

    RA=d['AIDA']['R']; nrmA2=np.linalg.norm(RA,axis=0); nrmA2[nrmA2<1e-12]=np.nan
    cand=np.where(isde&valid)[0]
    rr4=np.random.default_rng(999)
    rec=[]
    for it in range(NPERM_SEL):
        yp=d['AIDA']['yr'][rr4.permutation(RA.shape[0])]
        gp=(RA[:,cand].T@yp)/nrmA2[cand]
        sel=cand[np.argsort(-gp)[:k]]
        CA=cmat(RA,sel); CS=cmat(d['Synapse']['R'],sel)
        oA=offdiag(CA); oS=offdiag(CS)
        gA=gp[np.argsort(-gp)[:k]]
        ggA=offdiag(np.outer(gA,gA))
        PA=(CA-np.outer(gA,gA))/np.sqrt(np.outer(1-gA**2,1-gA**2))
        gS=d['Synapse']['g'][sel]
        PS=(CS-np.outer(gS,gS))/np.sqrt(np.outer(1-gS**2,1-gS**2))
        m1=np.isfinite(oA)&np.isfinite(oS)
        oPA=offdiag(PA); oPS=offdiag(PS); m2=np.isfinite(oPA)&np.isfinite(oPS)
        rec.append(dict(mean_r_AIDA=np.nanmean(oA),mean_absr_AIDA=np.nanmean(np.abs(oA)),
                        mean_gg_AIDA=np.nanmean(ggA),excess_r_AIDA=np.nanmean(oA)-np.nanmean(ggA),
                        pc1_corr_AIDA=pc1_corr(CA),mean_r_Synapse=np.nanmean(oS),
                        repro_r=spearmanr(oA[m1],oS[m1])[0],
                        repro_partial_r=spearmanr(oPA[m2],oPS[m2])[0],
                        mean_abs_partial_r_AIDA=np.nanmean(np.abs(oPA))))
    N=pd.DataFrame(rec); N['cell_type']=CT; N['version']='v1_selection_null'
    N.to_csv(f'{OUT}/tables/selection_null_{CT}.csv',index=False)
    q=N.describe(percentiles=[0.5,0.95,0.99])
    log(f'  [{CT}] 選抜帰無 (tAge 置換で上位{k}選抜, {NPERM_SEL}回):')
    for c in ['mean_r_AIDA','mean_absr_AIDA','pc1_corr_AIDA','repro_r','repro_partial_r']:
        log(f'     {c}: 平均{N[c].mean():.4f} 95%点{N[c].quantile(0.95):.4f} 最大{N[c].max():.4f}')
    obs_top=res['fg_top'][1]; obs_rnd=res['fg_rand'][1]
    for nm,mo in [('fg_top',obs_top),('fg_rand',obs_rnd)]:
        for c in ['mean_r_AIDA','mean_absr_AIDA','pc1_corr_AIDA','repro_r','repro_partial_r']:
            nullrows.append(dict(cell_type=CT,group=nm,metric=c,observed=mo[c],
                null_mean=float(N[c].mean()),null_sd=float(N[c].std()),null_p95=float(N[c].quantile(0.95)),
                null_max=float(N[c].max()),
                p_emp=float((( N[c].values>=mo[c]).sum()+1)/(len(N)+1))))

for CT in CTS:
    dd=load_ct(CT); run_v1(CT,dd); del dd

CT='NK'; d=load_ct(CT)
sub=corr[corr.cell_type==CT].set_index('cre_id')
expr_a=sub['expr_a'].reindex(cres).values; fdr_a=np.nan_to_num(sub['fdr_a'].reindex(cres).values,nan=1.0)
rho_a=sub['rho_a'].reindex(cres).values;   fdr_s=np.nan_to_num(sub['fdr_s'].reindex(cres).values,nan=1.0)
rho_s=sub['rho_s'].reindex(cres).values;   isde=sub['is_de'].reindex(cres).fillna(False).values.astype(bool)
prox=sub['proximity'].reindex(cres).values
nrmA=d['AIDA']['nrm']; nrmS=d['Synapse']['nrm']
valid=(nrmA>0.1*np.sqrt(d['AIDA']['n']))&(nrmS>0.1*np.sqrt(d['Synapse']['n']))&np.isfinite(rho_a)&np.isfinite(rho_s)
rob=valid&(fdr_a<0.05)&(fdr_s<0.05)&(np.sign(rho_a)==np.sign(rho_s))
fg2=np.where(rob&isde&(rho_a>0))[0]
bg2={'robust_down_any':np.where(rob&(rho_a<0))[0],
     'robust_up_proximal':np.where(rob&(prox=='proximal')&(rho_a>0))[0],
     'synonly_sameSign':np.where(valid&isde&(fdr_a>=0.05)&(fdr_s<0.05)&(np.sign(rho_a)==np.sign(rho_s)))[0]}
log(f'\n版2 前景 (両コホート頑健×上昇×遠位エンハンサー様)={len(fg2)}')
lg=np.log10(np.nan_to_num(expr_a,nan=0)+1e-3); dt=d['AIDA']['det']
feat=np.column_stack([(lg-np.nanmean(lg[valid]))/np.nanstd(lg[valid]),(dt-dt[valid].mean())/(dt[valid].std()+1e-12)])
for bname,bidx in bg2.items():
    log(f'  背景 {bname}: n={len(bidx)}')
    if len(bidx)<50: continue
    m,_=metrics(d,bidx,f'v2_bg_{bname}',extra=dict(version='v2_symmetric',cell_type=CT,group=f'bg_{bname}'))
    summary.append(m)
    setrows.append(pd.DataFrame(dict(version='v2_symmetric',cell_type=CT,group=f'bg_{bname}',cre_id=cres[bidx],
        rho_a=rho_a[bidx],fdr_a=fdr_a[bidx],rho_s=rho_s[bidx],fdr_s=fdr_s[bidx],expr_a=expr_a[bidx],
        detect_A=d['AIDA']['det'][bidx],detect_S=d['Synapse']['det'][bidx],
        g_A=d['AIDA']['g'][bidx],g_S=d['Synapse']['g'][bidx],proximity=prox[bidx])))
    log(f'    bg: mean_r_A={m["mean_r_AIDA"]:.3f} |r|A={m["mean_absr_AIDA"]:.3f} (床 {m["null_absr_AIDA"]:.3f}) '
        f'PC1={m["pc1_corr_AIDA"]:.3f} repro={m["repro_r"]:.3f} repro_partial={m["repro_partial_r"]:.3f}')

    sels=match_sets(fg2,bidx,feat,ndraw=NDRAW,seed=zlib.crc32(bname.encode())%1000)
    acc=[]
    for si,ss in enumerate(sels):
        mf,_=metrics(d,ss,f'v2_fg_matched_to_{bname}_{si}',do_null=(si==0),
                     extra=dict(version='v2_symmetric',cell_type=CT,group=f'fg_matched_{bname}',draw=si))
        acc.append(mf)
    A=pd.DataFrame(acc); num=A.select_dtypes(include=[np.number]); mm=num.mean().to_dict(); sd=num.std().to_dict()
    mm.update(dict(label=f'v2_fg_matched_{bname}_mean',version='v2_symmetric',cell_type=CT,
                   group=f'fg_matched_{bname}',n_draw=len(acc)))
    for kk in ['mean_r_AIDA','mean_absr_AIDA','pc1_corr_AIDA','repro_r','repro_partial_r']: mm[kk+'_sd']=sd.get(kk,np.nan)
    summary.append(mm); A.to_csv(f'{OUT}/tables/v2_fgmatched_{bname}.csv',index=False)
    log(f'    fg(同数マッチ {len(sels[0])}, {len(acc)}回): mean_r_A={mm["mean_r_AIDA"]:.3f}+-{mm["mean_r_AIDA_sd"]:.3f} '
        f'|r|A={mm["mean_absr_AIDA"]:.3f} PC1={mm["pc1_corr_AIDA"]:.3f} repro={mm["repro_r"]:.3f}+-{mm["repro_r_sd"]:.3f} '
        f'repro_partial={mm["repro_partial_r"]:.3f}')

    ref=np.concatenate([sels[0],bidx]); C=cmat(d['AIDA']['R'],ref)
    mi=np.nanmean(C,axis=1); lab=np.array([1]*len(sels[0])+[0]*len(bidx))
    obs=mi[lab==1].mean()-mi[lab==0].mean(); rr2=np.random.default_rng(91); cnt=0; NP=10000
    for _ in range(NP):
        p=rr2.permutation(lab); cnt+=(mi[p==1].mean()-mi[p==0].mean())>=obs
    elemrows.append(dict(version='v2_symmetric',cell_type=CT,group=f'fg_vs_{bname}',n_fg=len(sels[0]),n_bg=len(bidx),
        mean_mi_fg=float(mi[lab==1].mean()),mean_mi_bg=float(mi[lab==0].mean()),diff=float(obs),
        perm_p=float((cnt+1)/(NP+1)),boot_lo=np.nan,boot_hi=np.nan))
    log(f'    素子単位検定 fg vs {bname}: 差={obs:.4f} 置換p={(cnt+1)/(NP+1):.2e}')

pd.DataFrame(summary).to_csv(f'{OUT}/tables/summary_all_versions.csv',index=False)
pd.concat(setrows,ignore_index=True).to_csv(f'{OUT}/tables/element_sets_all.csv.gz',index=False)
pd.DataFrame(nullrows).to_csv(f'{OUT}/tables/selection_null_vs_observed.csv',index=False)
pd.DataFrame(elemrows).to_csv(f'{OUT}/tables/elementwise_tests.csv',index=False)
log('ANALYSIS DONE')


import numpy as np, pandas as pd
OUT='${PAPER_DIR}/results/fixes/coordination_redo'
S=pd.read_csv(f'{OUT}/tables/summary_all_versions.csv')
log=lambda *a: print(*a,flush=True)
def get(ver,ct,grp,draw_agg=True):
    x=S[(S.version==ver)&(S.cell_type==ct)&(S.group==grp)]
    if 'n_draw' in x.columns and x.n_draw.notna().any(): x=x[x.n_draw.notna()]
    elif 'draw' in x.columns and x.draw.notna().any(): x=x[x.draw.isna()] if (x.draw.isna()).any() else x.head(1)
    return x.iloc[0]

fg=get('v0_original','NK','fg'); bg=get('v0_original','NK','bg')
pub=dict(mean_absr_AIDA=(0.241,0.066),mean_absr_Synapse=(0.209,0.105),
         pc1_svd_AIDA=(0.286,0.058),repro_r=(0.445,0.222))
rows=[]
for k,(pf,pb) in pub.items():
    rows.append(dict(metric=k,published_fg=pf,recomputed_fg=round(float(fg[k]),4),
                     published_bg=pb,recomputed_bg=round(float(bg[k]),4)))
T1=pd.DataFrame(rows); T1.to_csv(f'{OUT}/tables/T1_reproduce_published.csv',index=False)
log('=== 表1 公表値の検算 (NK, 原著設計) ===\n'+T1.to_string(index=False))

rows=[]
for nm,x in [('前景 老化エンハンサー733',fg),('背景 発現量マッチ733',bg)]:
    rows.append(dict(group=nm,
      mean_r_AIDA=round(float(x.mean_r_AIDA),4), mean_absr_AIDA=round(float(x.mean_absr_AIDA),4),
      noise_floor_absr_AIDA=round(float(x.null_absr_AIDA),4),
      mean_g_AIDA=round(float(x.mean_g_AIDA),4), mean_gigj_AIDA=round(float(x.mean_gg_AIDA),4),
      excess_signed_r=round(float(x.excess_r_AIDA),4),
      excess_absr=round(float(x.mean_absr_minus_absgg_AIDA),4),
      mean_partial_r_AIDA=round(float(x.mean_partial_r_AIDA),4),
      repro_r=round(float(x.repro_r),4), repro_partial_r=round(float(x.repro_partial_r),4)))
T2=pd.DataFrame(rows); T2.to_csv(f'{OUT}/tables/T2_selection_decomposition.csv',index=False)
log('\n=== 表2 選抜由来の分離 (NK) ===\n'+T2.to_string(index=False))
log(f"  監査の見積もり 0.162 / 0.061 = 2.7 倍  →  実測 "
    f"{float(fg.mean_absr_minus_absgg_AIDA):.4f} / {float(bg.mean_absr_minus_absgg_AIDA):.4f} = "
    f"{float(fg.mean_absr_minus_absgg_AIDA)/float(bg.mean_absr_minus_absgg_AIDA):.2f} 倍")
log(f"  signed r で見ると {float(fg.excess_r_AIDA):.4f} / {float(bg.excess_r_AIDA):.4f} = "
    f"{float(fg.excess_r_AIDA)/float(bg.excess_r_AIDA):.2f} 倍")

rows=[dict(background='原著 (発現量のみ1:1最近傍, 1回)',n=int(bg.n_cre),
           mean_r_AIDA=round(float(bg.mean_r_AIDA),4),mean_absr_AIDA=round(float(bg.mean_absr_AIDA),4),
           pc1_svd_AIDA=round(float(bg.pc1_svd_AIDA),4),repro_r=round(float(bg.repro_r),4),sd=np.nan)]
for tag in ['bg_match_exprOnly','bg_match_expr_detect']:
    x=get('v0b_matchvariants','NK',tag)
    rows.append(dict(background=tag+' (20回平均)',n=int(x.n_cre),
        mean_r_AIDA=round(float(x.mean_r_AIDA),4),mean_absr_AIDA=round(float(x.mean_absr_AIDA),4),
        pc1_svd_AIDA=round(float(x.pc1_svd_AIDA),4),repro_r=round(float(x.repro_r),4),
        sd=round(float(x.repro_r_sd),4)))
rows.append(dict(background='★前景 (公表733)',n=int(fg.n_cre),mean_r_AIDA=round(float(fg.mean_r_AIDA),4),
    mean_absr_AIDA=round(float(fg.mean_absr_AIDA),4),pc1_svd_AIDA=round(float(fg.pc1_svd_AIDA),4),
    repro_r=round(float(fg.repro_r),4),sd=np.nan))
T3=pd.DataFrame(rows); T3.to_csv(f'{OUT}/tables/T3_matching_variants.csv',index=False)
log('\n=== 表3 マッチ変数を変えた背景 (公表733前景) ===\n'+T3.to_string(index=False))

rows=[]
for ct in S[S.version=='v1_AIDAonly'].cell_type.dropna().unique():
    for grp in ['v1_fg_top','v1_fg_rand']:
        try:
            f=get('v1_AIDAonly',ct,grp); b=get('v1_AIDAonly',ct,grp+'_bg')
        except Exception: continue
        rows.append(dict(cell_type=ct,foreground=grp,n=int(f.n_cre),
            fg_mean_r=round(float(f.mean_r_AIDA),4),bg_mean_r=round(float(b.mean_r_AIDA),4),
            fg_excess_r=round(float(f.excess_r_AIDA),4),bg_excess_r=round(float(b.excess_r_AIDA),4),
            fg_pc1=round(float(f.pc1_corr_AIDA),4),bg_pc1=round(float(b.pc1_corr_AIDA),4),
            fg_repro=round(float(f.repro_r),4),bg_repro=round(float(b.repro_r),4),
            bg_repro_sd=round(float(b.repro_r_sd),4),
            repro_gap=round(float(f.repro_r)-float(b.repro_r),4),
            fg_repro_partial=round(float(f.repro_partial_r),4),
            bg_repro_partial=round(float(b.repro_partial_r),4),
            repro_partial_gap=round(float(f.repro_partial_r)-float(b.repro_partial_r),4)))
T4=pd.DataFrame(rows).sort_values(['foreground','repro_gap'],ascending=[True,False])
T4.to_csv(f'{OUT}/tables/T4_v1_all_celltypes.csv',index=False)
log('\n=== 表4 版1 (AIDA 単独選抜) 全細胞種 ===\n'+T4.to_string(index=False))

rows=[]
for b in ['robust_down_any','robust_up_proximal','synonly_sameSign']:
    try:
        bb=get('v2_symmetric','NK','bg_'+b); ff=get('v2_symmetric','NK','fg_matched_'+b)
    except Exception: continue
    rows.append(dict(background=b,n=int(bb.n_cre),
        fg_mean_r=round(float(ff.mean_r_AIDA),4),bg_mean_r=round(float(bb.mean_r_AIDA),4),
        fg_absr=round(float(ff.mean_absr_AIDA),4),bg_absr=round(float(bb.mean_absr_AIDA),4),
        fg_pc1=round(float(ff.pc1_corr_AIDA),4),bg_pc1=round(float(bb.pc1_corr_AIDA),4),
        fg_repro=round(float(ff.repro_r),4),bg_repro=round(float(bb.repro_r),4),
        gap=round(float(ff.repro_r)-float(bb.repro_r),4),
        fg_repro_partial=round(float(ff.repro_partial_r),4),bg_repro_partial=round(float(bb.repro_partial_r),4)))
T5=pd.DataFrame(rows); T5.to_csv(f'{OUT}/tables/T5_v2_symmetric_NK.csv',index=False)
log('\n=== 表5 版2 対称条件 (NK, 前景は同数+発現量/検出率マッチ) ===\n'+T5.to_string(index=False))

N=pd.read_csv(f'{OUT}/tables/selection_null_vs_observed.csv')
N.to_csv(f'{OUT}/tables/T6_selection_null_vs_observed.csv',index=False)
log('\n=== 表6 選抜帰無 (tAge 置換で同じ上位733選抜, 200回) vs 実測 ===\n'
    +N[N.metric.isin(['mean_r_AIDA','repro_r','repro_partial_r'])].to_string(index=False))

E=pd.read_csv(f'{OUT}/tables/elementwise_tests.csv')
log('\n=== 表7 素子単位検定 (ペア単位 MWU の置き換え) ===\n'+E.to_string(index=False))

ES=pd.read_csv(f'{OUT}/tables/element_sets_all.csv.gz')
P=ES.groupby(['version','cell_type','group']).agg(n=('cre_id','size'),
    expr_med=('expr_a','median'),detA=('detect_A','mean'),detS=('detect_S','mean'),
    gA=('g_A','mean'),gS=('g_S','mean'),
    frac_syn_sig=('fdr_s',lambda v:(v<0.05).mean())).reset_index()
P.to_csv(f'{OUT}/tables/T8_group_composition.csv',index=False)
log('\n=== 表8 群の発現量・検出率・Synapse 有意割合 (NK 抜粋) ===\n'
    +P[P.cell_type=='NK'].round(4).to_string(index=False))
log('\nFINAL TABLES DONE')


import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
OUT='${PAPER_DIR}/results/fixes/coordination_redo'
SCR='${SCRATCH}'
ENH={'unidirectional_enhancer','divergent_enhancer','convergent_other_distal'}
CT='NK'; CAL=0.25; NDRAW=20
cres=np.asarray(np.load(f'{SCR}/mat/cres.npy',allow_pickle=True),dtype=object)
corr=pd.read_csv(f'{OUT}/tables/corr_merged_by_celltype.csv.gz'); corr=corr[corr.cell_type==CT].set_index('cre_id')
expr_a=corr['expr_a'].reindex(cres).values; fdr_a=np.nan_to_num(corr['fdr_a'].reindex(cres).values,nan=1.0)
rho_a=corr['rho_a'].reindex(cres).values; fdr_s=np.nan_to_num(corr['fdr_s'].reindex(cres).values,nan=1.0)
rho_s=corr['rho_s'].reindex(cres).values; prox=corr['proximity'].reindex(cres).values
isde=((corr['proximity']=='distal')&corr['enhancer_class'].isin(ENH)).reindex(cres).fillna(False).values.astype(bool)
d={}
for coh in ['AIDA','Synapse']:
    d[coh]=dict(R=np.load(f'{SCR}/mat/{coh}_{CT}_R.npy'),g=np.load(f'{SCR}/mat/{coh}_{CT}_g.npy'),
                det=np.load(f'{SCR}/mat/{coh}_{CT}_detect.npy'))
    d[coh]['nrm']=np.linalg.norm(d[coh]['R'],axis=0); d[coh]['n']=d[coh]['R'].shape[0]
valid=(d['AIDA']['nrm']>0.1*np.sqrt(d['AIDA']['n']))&(d['Synapse']['nrm']>0.1*np.sqrt(d['Synapse']['n']))&np.isfinite(rho_a)&np.isfinite(rho_s)
rob=valid&(fdr_a<0.05)&(fdr_s<0.05)&(np.sign(rho_a)==np.sign(rho_s))
fg=np.where(rob&isde&(rho_a>0))[0]
BG={'robust_down_any':np.where(rob&(rho_a<0))[0],
    'robust_up_proximal':np.where(rob&(prox=='proximal')&(rho_a>0))[0],
    'robust_down_proximal':np.where(rob&(prox=='proximal')&(rho_a<0))[0]}
lg=np.log10(np.nan_to_num(expr_a,nan=0)+1e-3); dt=d['AIDA']['det']
z=lambda v:(v-np.nanmean(v[valid]))/(np.nanstd(v[valid])+1e-12)
F=np.column_stack([z(lg),z(dt)])
def offd(C):
    iu=np.triu_indices_from(C,1); return C[iu]
def cmat(R,idx):
    S=R[:,idx].astype(np.float64); nr=np.linalg.norm(S,axis=0); nr[nr<1e-12]=np.nan
    S=S/nr; C=S.T@S; np.fill_diagonal(C,np.nan); return C
def pc1c(C):
    A=np.array(C,copy=True); np.fill_diagonal(A,1.0); A=np.nan_to_num(A)
    return float(np.linalg.eigvalsh(A)[-1]/A.shape[0])
def met(idx):
    o={}
    for coh in ['AIDA','Synapse']:
        C=cmat(d[coh]['R'],idx); o[coh]=offd(C); g=d[coh]['g'][idx]
        P=(C-np.outer(g,g))/np.sqrt(np.outer(1-g**2,1-g**2)); o[coh+'P']=offd(P)
        if coh=='AIDA': pcA=pc1c(C); mrA=np.nanmean(o[coh])
    m1=np.isfinite(o['AIDA'])&np.isfinite(o['Synapse']); m2=np.isfinite(o['AIDAP'])&np.isfinite(o['SynapseP'])
    return dict(n=len(idx),mean_r_AIDA=mrA,pc1_AIDA=pcA,
        repro_r=spearmanr(o['AIDA'][m1],o['Synapse'][m1])[0],
        repro_partial_r=spearmanr(o['AIDAP'][m2],o['SynapseP'][m2])[0],
        med_expr=float(np.nanmedian(expr_a[idx])),mean_det=float(dt[idx].mean()))
rows=[]
tf=cKDTree(F[fg])
for bn,bi in BG.items():
    dist,nb=tf.query(F[bi],k=min(len(fg),200))
    ok=dist<=CAL
    for dw in range(NDRAW):
        rr=np.random.default_rng(4000+dw)
        order=rr.permutation(len(bi)); usedf=set(); pf=[]; pb=[]
        for i in order:
            cand=[int(c) for c,g_ in zip(nb[i],ok[i]) if g_ and int(c) not in usedf]
            if not cand: continue
            pick=int(rr.choice(cand[:10])) if len(cand)>=10 else cand[0]
            usedf.add(pick); pf.append(fg[pick]); pb.append(bi[i])
        if len(pf)<30: continue
        a=met(np.array(pf)); b=met(np.array(pb))
        rows.append(dict(background=bn,draw=dw,side='fg_matched',**a))
        rows.append(dict(background=bn,draw=dw,side='bg_matched',**b))
D=pd.DataFrame(rows); D.to_csv(f'{OUT}/tables/v2_draws_raw_commonsupport.csv',index=False)
S=D.groupby(['background','side']).agg(n=('n','mean'),
    mean_r=('mean_r_AIDA','mean'),mean_r_sd=('mean_r_AIDA','std'),
    pc1=('pc1_AIDA','mean'),pc1_sd=('pc1_AIDA','std'),
    repro=('repro_r','mean'),repro_sd=('repro_r','std'),
    repro_partial=('repro_partial_r','mean'),repro_partial_sd=('repro_partial_r','std'),
    med_expr=('med_expr','mean'),mean_det=('mean_det','mean')).round(4).reset_index()
S.to_csv(f'{OUT}/tables/T12_v2_commonsupport_20draws.csv',index=False)
print(S.to_string(index=False))

print('\n--- 同一 draw 内の対応差 (前景 - 背景) ---')
dif=[]
for bn in BG:
    x=D[D.background==bn].pivot(index='draw',columns='side')
    for m in ['mean_r_AIDA','repro_r','repro_partial_r','pc1_AIDA']:
        v=x[m]['fg_matched']-x[m]['bg_matched']
        dif.append(dict(background=bn,metric=m,mean_diff=round(v.mean(),4),sd=round(v.std(),4),
                        n_draw=len(v),frac_positive=round((v>0).mean(),3)))
        print(f'  {bn:24s} {m:16s} 差={v.mean():+.4f} (SD {v.std():.4f}, {int((v>0).sum())}/{len(v)} 回で前景>背景)')
pd.DataFrame(dif).to_csv(f'{OUT}/tables/T13_v2_paired_differences.csv',index=False)
print('DONE')
