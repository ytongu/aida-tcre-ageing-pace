#!/usr/bin/env python3
import os, sys, glob, gc
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import sparse, io as sio
import anndata as ad

TS = datetime.now().strftime('%Y%m%d_%H%M%S')
BASE = '${PIPELINE_DIR}'
NAS  = '${DATA_DIR}/lnc_project_synapse_dataset/syn49637038'

CELL_META = sorted(glob.glob(f'{BASE}/results/04_gene_expression/cell_metadata_with_donor_*.csv'))[-1]
CR_LOC = f'{NAS}/02_cellranger'
CR_NIG = f'{NAS}/02_cellranger_nig'
OUT_DIR = f'{BASE}/results/05_metacell'
LOG = f'{BASE}/logs/34_metacell_gene_{TS}.log'
MIN_CELLS = 10

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG), exist_ok=True)

def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def cr_path(sid):
    for base in (CR_LOC, CR_NIG):
        p = f'{base}/{sid}/outs/filtered_feature_bc_matrix'
        if os.path.isfile(f'{p}/matrix.mtx.gz'):
            return p
    return None

log(f"loading cell_metadata: {CELL_META}")
cm = pd.read_csv(CELL_META, index_col=0)
ok = cm[cm['patient_status']=='ok'].copy()
ok['barcode'] = [idx[len(s)+1:] for idx, s in zip(ok.index, ok['sample'].astype(str))]
log(f"  ok cells: {len(ok):,}")

ref_p = cr_path(ok['sample'].iloc[0])
features = pd.read_csv(f'{ref_p}/features.tsv.gz', sep='\t', header=None,
                       names=['gene_id','gene_symbol','feature_type'])
log(f"reference features: {len(features):,} genes ({ref_p})")
n_gene = len(features)

def make_accumulator():
    return {'sums': {}, 'meta': {}}
acc_ct  = make_accumulator()
acc_sub = make_accumulator()

samples = sorted(ok['sample'].unique())
log(f"samples to process: {len(samples)}")

processed = 0
dropped_no_cr = 0
for sid in samples:
    p = cr_path(sid)
    if p is None:
        n = (ok['sample']==sid).sum()
        log(f"  SKIP {sid} (no CR matrix, dropped {n} ok cells)")
        dropped_no_cr += n
        continue
    M = sio.mmread(f'{p}/matrix.mtx.gz').tocsc()
    bc = pd.read_csv(f'{p}/barcodes.tsv.gz', sep='\t', header=None)[0].tolist()
    if M.shape[1] != len(bc):
        log(f"  WARN {sid}: M cols ({M.shape[1]}) != bc ({len(bc)})")
    if M.shape[0] != n_gene:
        log(f"  WARN {sid}: M rows ({M.shape[0]}) != features ({n_gene})")
        del M; gc.collect(); continue
    bc_to_col = {b: i for i, b in enumerate(bc)}

    sub_ok = ok[ok['sample']==sid]
    for level, acc, group_col in [('celltype', acc_ct, 'cell_type_aida'),
                                  ('subcluster', acc_sub, 'subcluster')]:
        for (donor, batch, group), sub in sub_ok.groupby(['Donor_id','batch', group_col], observed=True):
            if pd.isna(donor) or pd.isna(group):
                continue
            cols = [bc_to_col[b] for b in sub['barcode'] if b in bc_to_col]
            if not cols:
                continue
            sub_M = M[:, cols]
            summed = np.asarray(sub_M.sum(axis=1)).ravel()
            key = (donor, batch, group)
            if key in acc['sums']:
                acc['sums'][key] = acc['sums'][key] + summed
                acc['meta'][key]['n_cells'] += len(cols)
            else:
                acc['sums'][key] = summed
                acc['meta'][key] = {
                    'Donor_id': donor, 'Batch': batch, group_col: group,
                    'n_cells': len(cols),
                    'Age': sub['Age'].iloc[0],
                    'Sex': sub['Sex'].iloc[0],
                    'Age_group': sub['Age_group'].iloc[0],
                }

    del M; gc.collect()
    processed += 1
    if processed % 10 == 0:
        log(f"  processed {processed}/{len(samples)}; ct={len(acc_ct['sums'])} sub={len(acc_sub['sums'])}")

log(f"done loop. processed={processed}/{len(samples)} dropped_no_cr={dropped_no_cr}")

def build_h5ad(acc, level_col, out_path):
    keys = [k for k, m in acc['meta'].items() if m['n_cells'] >= MIN_CELLS]
    log(f"  {level_col}: {len(acc['sums'])} candidate, {len(keys)} pass MIN_CELLS={MIN_CELLS}")
    if not keys:
        return
    obs_rows = [acc['meta'][k] for k in keys]
    obs = pd.DataFrame(obs_rows)
    obs.index = [f"{m['Donor_id']}.{m['Batch']}.{m[level_col]}" for m in obs_rows]
    obs.index.name = 'metacell_id'
    X = np.vstack([acc['sums'][k] for k in keys]).astype(np.float32)
    var = features.copy()
    var.index = var['gene_id'].astype(str)
    a = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)
    a.obs['Coverage'] = np.asarray(a.X.sum(axis=1)).ravel().astype('int64')
    a.write(out_path)
    log(f"  wrote {out_path} shape={a.shape} ({os.path.getsize(out_path)/1e6:.1f}MB)")

OUT_CT = f'{OUT_DIR}/metacells_gene_celltype_{TS}.h5ad'
OUT_SUB = f'{OUT_DIR}/metacells_gene_subcluster_{TS}.h5ad'
build_h5ad(acc_ct, 'cell_type_aida', OUT_CT)
build_h5ad(acc_sub, 'subcluster', OUT_SUB)

for level, path in [('celltype', OUT_CT), ('subcluster', OUT_SUB)]:
    if os.path.isfile(path):
        a = ad.read_h5ad(path, backed='r')
        col = 'cell_type_aida' if level=='celltype' else 'subcluster'
        log(f"\n[{level}] metacells per group:\n{a.obs[col].value_counts().to_string()}")
        npd = a.obs.groupby('Donor_id')['Batch'].nunique()
        log(f"\n[{level}] timepoints: 1tp={int((npd==1).sum())} 2tp={int((npd==2).sum())} 3tp={int((npd==3).sum())}")
log("DONE")


import os, sys, glob, gc
from datetime import datetime
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

TS = datetime.now().strftime('%Y%m%d_%H%M%S')
BASE = '${PIPELINE_DIR}'
MC_DIR = f'{BASE}/results/05_metacell'
OUT_BASE = f'{BASE}/results/06_tage'
LOG = f'{BASE}/logs/35a_export_{TS}.log'

os.makedirs(OUT_BASE, exist_ok=True)
os.makedirs(os.path.dirname(LOG), exist_ok=True)

def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def export_level(level, group_col):
    pat = f'{MC_DIR}/metacells_gene_{level}_*.h5ad'
    files = sorted(glob.glob(pat))
    if not files:
        log(f"  no h5ad for {level}: {pat}")
        return
    h5ad_path = files[-1]
    log(f"\n[{level}] loading {h5ad_path}")
    a = sc.read_h5ad(h5ad_path)
    log(f"  shape: {a.shape}  obs cols: {list(a.obs.columns)}")

    out_dir = f'{OUT_BASE}/{level}'
    os.makedirs(out_dir, exist_ok=True)

    sym = a.var['gene_symbol'].astype(str).values
    keep = sym != a.var['gene_id'].astype(str).values
    log(f"  genes with HGNC symbol: {keep.sum():,} / {len(sym):,}")

    X = a.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X[:, keep].astype(np.int64).T.tocsr()
    sym_kept = sym[keep]

    uniq, inv = np.unique(sym_kept, return_inverse=True)
    if len(uniq) < len(sym_kept):
        log(f"  collapsing {len(sym_kept) - len(uniq):,} duplicate symbols (sum)")

        n_uniq = len(uniq)

        ind = sp.csr_matrix(
            (np.ones(len(sym_kept), dtype=np.int8),
             (inv, np.arange(len(sym_kept)))),
            shape=(n_uniq, len(sym_kept))
        )
        X_collapsed = ind @ X
        X = X_collapsed
        sym_kept = uniq
    log(f"  final expr shape: {X.shape}  (genes × metacells)")

    expr_csv = f'{out_dir}/Exprs_metacells_all.csv'
    log(f"  writing expr csv -> {expr_csv}  ({X.shape[0]:,} × {X.shape[1]:,}) ...")
    df_e = pd.DataFrame(
        X.toarray() if sp.issparse(X) else X,
        index=pd.Index(sym_kept, name='gene'),
        columns=a.obs.index,
    )
    df_e.to_csv(expr_csv)
    log(f"  done expr csv ({os.path.getsize(expr_csv)/1e6:.1f}MB)")
    del df_e, X; gc.collect()

    meta = a.obs.copy()
    meta['CellType'] = meta[group_col].astype(str)
    meta['Treatment'] = 'all'
    meta['Sample'] = meta['Donor_id'].astype(str) + '.' + meta['Batch'].astype(str)
    keep_cols = ['Sample', 'Donor_id', 'Batch', 'CellType', 'Age', 'Sex', 'Age_group', 'n_cells', 'Coverage', 'Treatment']
    meta_csv = f'{out_dir}/Metadata_metacells_all.csv'
    meta[keep_cols].to_csv(meta_csv)
    log(f"  meta csv -> {meta_csv}  ({len(meta):,} rows, {os.path.getsize(meta_csv)/1e3:.1f}KB)")

    with open(f'{out_dir}/EXPORT_TS.txt', 'w') as f:
        f.write(f'{TS}\n{h5ad_path}\n')

    del a; gc.collect()

log("=== export start ===")
export_level('celltype', 'cell_type_aida')
export_level('subcluster', 'subcluster')
log("=== export done ===")


import h5py, numpy as np, pandas as pd, glob
from scipy.sparse import csr_matrix
W='${SCRATCH}'
CH=200_000
A=pd.read_parquet(f'{W}/aida_cd4_final.parquet')[['barcode_full','our_cd4']]
S=pd.read_parquet(f'{W}/synapse_cd4_final.parquet')[['barcode_full','our_cd4']]
m=pd.read_csv('${PROJECT_DIR}/results/phase3b_cre_corr/barcode_metadata_latest.tsv',sep='\t',usecols=['barcode_full','cohort','Donor_id','CellType'])
m=m[m.CellType=='CD4_T_cell']
lab=pd.concat([A,S]).merge(m[['barcode_full','cohort','Donor_id']],on='barcode_full',how='inner')
lab=lab[lab.our_cd4.isin(['Naive','Memory'])].copy()
lab['grp']=lab.Donor_id+'|CD4_'+lab.our_cd4
print(f'CD4: {len(lab):,} (AIDA {int((lab.cohort=="AIDA").sum()):,} / Synapse {int((lab.cohort!="AIDA").sum()):,})',flush=True)
def agg(files,b2g,groups):
    g2i={g:i for i,g in enumerate(groups)}; ACC=None; genes=None
    for fp in files:
        with h5py.File(fp,'r') as f:
            oi=f['obs'].attrs.get('_index','_index'); vi=f['var'].attrs.get('_index','_index')
            bc=np.array([x.decode() if isinstance(x,bytes) else x for x in f['obs'][oi][:]])
            gs=[x.decode() if isinstance(x,bytes) else x for x in f['var'][vi][:]]
            if genes is None: genes=gs; ACC=np.zeros((len(groups),len(genes)),np.float32)
            gi=np.array([g2i.get(b2g.get(b,''),-1) for b in bc],np.int32)
            ip=f['X']['indptr']; dat=f['X']['data']; idx=f['X']['indices']; n=len(bc)
            for s in range(0,n,CH):
                e=min(s+CH,n); p=ip[s:e+1]
                if p[-1]==p[0]: continue
                M=csr_matrix((dat[p[0]:p[-1]],idx[p[0]:p[-1]],p-p[0]),shape=(e-s,len(genes)))
                gg=gi[s:e]; ok=gg>=0
                if not ok.any(): continue
                nn=int(ok.sum())
                Sm=csr_matrix((np.ones(nn),(gg[ok],np.arange(nn))),shape=(len(groups),nn))
                ACC+=np.asarray((Sm@M[ok]).todense(),np.float32)
            print(f'  {fp.split("/")[-1]} done',flush=True)
    return ACC,genes
la=lab[lab.cohort=='AIDA']; b2g=dict(zip(la.barcode_full,la.grp)); groups=sorted(la.grp.unique())
Ea,ga=agg(['${DATA_DIR}/lnc_project_AIDA_dataset/all_cells_clustered/all_cells_clustered.h5ad'],b2g,groups)
dfa=pd.DataFrame(Ea.T,index=ga,columns=groups); print(f'AIDA CD4 mc: {dfa.shape[1]}',flush=True)
ls=lab[lab.cohort!='AIDA']; b2g=dict(zip(ls.barcode_full,ls.grp)); groups=sorted(ls.grp.unique())
Es,gs=agg(sorted(glob.glob('${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/04_gene_expression/by_subcluster/CD4_*.h5ad')),b2g,groups)
dfs=pd.DataFrame(Es.T,index=gs,columns=groups); print(f'Synapse CD4 mc: {dfs.shape[1]}',flush=True)
common=dfa.index.intersection(dfs.index)
E=pd.concat([dfa.loc[common],dfs.loc[common]],axis=1)
E=E[~E.index.str.startswith('ENSG')].groupby(level=0).sum()
E.to_parquet(f'{W}/cd4_gene_mc.parquet')
print(f'-> cd4_gene_mc.parquet: {E.shape[0]:,} genes x {E.shape[1]:,} mc',flush=True)


import os, glob, h5py
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUT = f'{D}/results/gene_mc_canonical'
os.makedirs(OUT, exist_ok=True)
CH = 200_000
SUBS = ['B_Memory','B_Naive','CD14_Mono','CD16_Mono','CD4_Memory','CD4_Naive',
        'CD8_EM','CD8_Effector','CD8_Naive','NK','NKT']
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

can = pd.read_parquet(f'{D}/results/canonical_labels/barcode_subset_canonical.parquet',
                      columns=['cohort','barcode_full','Donor_id','Batch','subset_canonical'])
can = can[can.subset_canonical.isin(SUBS)].copy()
can['grp'] = can.Donor_id + '|' + can.subset_canonical
w(f'正典ラベルの細胞 {len(can):,}（AIDA {int((can.cohort=="AIDA").sum()):,} / '
  f'Synapse {int((can.cohort!="AIDA").sum()):,}）')

def agg(files, b2g, groups, tag):
    g2i = {g: i for i, g in enumerate(groups)}
    ACC = None; genes = None
    for fp in files:
        with h5py.File(fp, 'r') as f:
            oi = f['obs'].attrs.get('_index', '_index'); vi = f['var'].attrs.get('_index', '_index')
            bc = np.array([x.decode() if isinstance(x, bytes) else x for x in f['obs'][oi][:]])
            gs = [x.decode() if isinstance(x, bytes) else x for x in f['var'][vi][:]]
            if genes is None:
                genes = gs; ACC = np.zeros((len(groups), len(genes)), np.float32)
            gi = np.array([g2i.get(b2g.get(b, ''), -1) for b in bc], np.int32)
            ip = f['X']['indptr']; dat = f['X']['data']; idx = f['X']['indices']; n = len(bc)
            hit = int((gi >= 0).sum())
            for s in range(0, n, CH):
                e = min(s + CH, n); p = ip[s:e + 1]
                if p[-1] == p[0]: continue
                M = csr_matrix((dat[p[0]:p[-1]], idx[p[0]:p[-1]], p - p[0]),
                               shape=(e - s, len(genes)))
                gg = gi[s:e]; ok = gg >= 0
                if not ok.any(): continue
                nn = int(ok.sum())
                Sm = csr_matrix((np.ones(nn), (gg[ok], np.arange(nn))), shape=(len(groups), nn))
                ACC += np.asarray((Sm @ M[ok]).todense(), np.float32)
            w(f'  {tag} {os.path.basename(fp)}: 細胞 {n:,} / 採用 {hit:,}')
    return ACC, genes

la = can[can.cohort == 'AIDA']
b2g = dict(zip(la.barcode_full, la.grp)); groups_a = sorted(la.grp.unique())
w(f'\nAIDA メタセル {len(groups_a):,}')
Ea, ga = agg(['${DATA_DIR}/lnc_project_AIDA_dataset/all_cells_clustered/all_cells_clustered.h5ad'],
             b2g, groups_a, 'AIDA')
dfa = pd.DataFrame(Ea.T, index=ga, columns=groups_a)

ls_ = can[can.cohort != 'AIDA']
b2g = dict(zip(ls_.barcode_full, ls_.grp)); groups_s = sorted(ls_.grp.unique())
w(f'\nTerekhova メタセル {len(groups_s):,}')
files = [f'${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/04_gene_expression/'
         f'by_subcluster/{s}.h5ad' for s in SUBS]
files = [f for f in files if os.path.exists(f)]
Es, gs = agg(files, b2g, groups_s, 'Terekhova')
dfs = pd.DataFrame(Es.T, index=gs, columns=groups_s)

common = dfa.index.intersection(dfs.index)
E = pd.concat([dfa.loc[common], dfs.loc[common]], axis=1)
E = E[~E.index.str.startswith('ENSG')].groupby(level=0).sum()
w(f'\n遺伝子 {E.shape[0]:,} × メタセル {E.shape[1]:,}')

nc = can.groupby('grp').size().rename('n_cells')
meta = pd.DataFrame({'ID': E.columns})
meta['Sample'] = [g.split('|')[0] for g in meta.ID]
meta['Subcluster'] = [g.split('|')[1] for g in meta.ID]
meta['n_cells'] = meta.ID.map(nc)
meta['Coverage'] = E.sum(0).reindex(meta.ID).values.astype(np.int64)
key = can.drop_duplicates('grp').set_index('grp')
meta['cohort'] = meta.ID.map(key.cohort)
E.to_csv(f'{OUT}/Exprs_metacells_canonical.csv')
meta.to_csv(f'{OUT}/Metadata_metacells_canonical.csv', index=False)

old = pd.read_csv('${DATA_DIR}/lnc_project_AIDA_dataset/by_subcluster/'
                  'Metadata_metacells_by_subcluster.csv')
o = old.groupby('Subcluster').n_cells.sum().rename('old')
n = meta[meta.cohort == 'AIDA'].groupby('Subcluster').n_cells.sum().rename('new')
cmp = pd.concat([o, n], axis=1); cmp['ratio'] = cmp.old / cmp.new
cmp.to_csv(f'{OUT}/compare_vs_old.csv')
w('\n=== AIDA 細胞数: 旧（元ラベル）対 新（正典ラベル）===')
for s, r in cmp.iterrows():
    flag = '  ★変化' if abs(r.ratio - 1) > 0.02 else ''
    w(f'  {s:14s} 旧 {r.old:>8,.0f}  新 {r.new:>8,.0f}  比 {r.ratio:5.2f}{flag}')
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')
