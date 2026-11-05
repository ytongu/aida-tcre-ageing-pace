#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import os
import glob
import gc
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=150, fontsize=12)
plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'Arial'})

NAS_CR_LOCAL = "${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/02_cellranger"
NAS_CR_NIG = "${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/02_cellranger_nig"
OUTDIR = os.path.expanduser("~/projects/lnc_project_scafe/results/04_gene_expression")
TMPDIR = os.path.expanduser("~/projects/lnc_project_scafe/results/04_gene_expression/tmp_samples")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(TMPDIR, exist_ok=True)
os.makedirs(f"{OUTDIR}/by_cluster", exist_ok=True)
os.makedirs(f"{OUTDIR}/by_subcluster", exist_ok=True)
os.makedirs(f"{OUTDIR}/figures", exist_ok=True)

TS = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
MAX_CELLS_PER_SAMPLE = 3000

print("=" * 60)
print("Phase 1: Individual sample QC + normalize")
print("=" * 60)

EXCLUDE_SAMPLES = {'AS062-2_new'}

sample_dirs = []
for base in [NAS_CR_LOCAL, NAS_CR_NIG]:
    for d in sorted(glob.glob(f"{base}/*/outs/filtered_feature_bc_matrix")):
        sample_id = d.split("/")[-3]
        if sample_id in EXCLUDE_SAMPLES:
            continue
        if any(s[0] == sample_id for s in sample_dirs):
            continue

        if not os.path.isfile(f"{d}/matrix.mtx.gz"):
            continue
        sample_dirs.append((sample_id, d))

print(f"Found {len(sample_dirs)} samples (excluded: {EXCLUDE_SAMPLES})")

sample_cells = {}
total_cells = 0
for i, (sample_id, matrix_path) in enumerate(sample_dirs):
    out_path = f"{TMPDIR}/{sample_id}.h5ad"

    if os.path.exists(out_path):
        adata = sc.read_h5ad(out_path, backed='r')
        n = adata.n_obs
        adata.file.close()
        sample_cells[sample_id] = n
        total_cells += n
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(sample_dirs)}] {sample_id}: {n} cells (cached)")
        continue

    try:
        adata = sc.read_10x_mtx(matrix_path, var_names='gene_symbols', cache=False)
        adata.X = adata.X.astype(np.float32)
        adata.obs['sample'] = sample_id
        adata.obs['batch'] = sample_id.split('-')[0]
        adata.obs_names = [f"{sample_id}_{bc}" for bc in adata.obs_names]

        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, inplace=True)
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_cells(adata, max_genes=6000)
        adata = adata[adata.obs['pct_counts_mt'] < 20, :].copy()
        sc.pp.filter_genes(adata, min_cells=3)

        adata.raw = adata.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        adata.write(out_path)
        sample_cells[sample_id] = adata.n_obs
        total_cells += adata.n_obs

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(sample_dirs)}] {sample_id}: {adata.n_obs} cells")

        del adata
        gc.collect()
    except Exception as e:
        print(f"  ERROR {sample_id}: {e}")

print(f"\nPhase 1 complete: {len(sample_cells)} samples, {total_cells:,} cells total")

print("\n" + "=" * 60)
print("Phase 2: Subsample + PCA + Harmony + UMAP + Leiden")
print("=" * 60)

print("Loading subsampled cells...")
adatas_sub = []
for sample_id in sorted(sample_cells.keys()):
    adata = sc.read_h5ad(f"{TMPDIR}/{sample_id}.h5ad")
    n_cells = adata.n_obs
    if n_cells > MAX_CELLS_PER_SAMPLE:
        sc.pp.subsample(adata, n_obs=MAX_CELLS_PER_SAMPLE, random_state=42)
    adatas_sub.append(adata)
    del adata

adata_sub = sc.concat(adatas_sub, join='outer')
adata_sub.var_names_make_unique()
del adatas_sub
gc.collect()

print(f"Subsample: {adata_sub.n_obs:,} cells x {adata_sub.n_vars:,} genes")

sc.pp.highly_variable_genes(adata_sub, n_top_genes=3000, batch_key='batch')
hvg_genes = adata_sub.var_names[adata_sub.var['highly_variable']].tolist()
print(f"HVGs: {len(hvg_genes)}")

adata_hvg = adata_sub[:, adata_sub.var['highly_variable']].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=50, svd_solver='arpack')
print("PCA done")

try:
    import harmonypy
    sc.external.pp.harmony_integrate(adata_hvg, key='batch')
    use_rep = 'X_pca_harmony'
    print("Harmony done")
except Exception as e:
    print(f"Harmony failed ({e}), using PCA")
    use_rep = 'X_pca'

adata_sub.obsm['X_pca'] = adata_hvg.obsm['X_pca'].astype(np.float32).copy()
if use_rep == 'X_pca_harmony':
    adata_sub.obsm['X_pca_harmony'] = adata_hvg.obsm['X_pca_harmony'].astype(np.float32).copy()
del adata_hvg
gc.collect()

sc.pp.neighbors(adata_sub, use_rep=use_rep, n_neighbors=30, n_pcs=30)
sc.tl.umap(adata_sub)
sc.tl.leiden(adata_sub, resolution=0.8, key_added='leiden')
print(f"Leiden clusters: {adata_sub.obs['leiden'].nunique()}")

print("\n" + "=" * 60)
print("Phase 3: Cell type annotation (AIDA-compatible)")
print("=" * 60)

markers = {
    'CD4_T_cell': ['CD3D', 'CD3E', 'CD4', 'IL7R', 'TCF7', 'LEF1'],
    'CD8_T_cell': ['CD3D', 'CD3E', 'CD8A', 'CD8B', 'GZMK', 'GZMB'],
    'NK_cell': ['NKG7', 'GNLY', 'KLRD1', 'KLRF1', 'NCAM1'],
    'B_cell': ['CD79A', 'CD79B', 'MS4A1', 'CD19', 'BANK1'],
    'Monocyte': ['CD14', 'LYZ', 'S100A8', 'S100A9', 'FCGR3A', 'CST3'],
    'CD4_Naive': ['CCR7', 'SELL', 'TCF7', 'LEF1', 'CD4'],
    'CD4_Memory': ['S100A4', 'IL7R', 'CD69', 'CD4'],
    'CD8_Naive': ['CCR7', 'SELL', 'TCF7', 'CD8A'],
    'CD8_Effector': ['GZMB', 'PRF1', 'NKG7', 'CD8A', 'GNLY'],
    'CD8_EM': ['GZMK', 'EOMES', 'CD8A'],
    'NKT': ['NKG7', 'CD3D', 'KLRD1'],
    'CD14_Mono': ['CD14', 'LYZ', 'S100A8', 'S100A9'],
    'CD16_Mono': ['FCGR3A', 'MS4A7', 'CDKN1C'],
    'B_Naive': ['MS4A1', 'IGHD', 'TCL1A', 'FCER2'],
    'B_Memory': ['MS4A1', 'CD27', 'IGHG1'],
}

for ct, genes in markers.items():
    available = [g for g in genes if g in adata_sub.var_names]
    if available:
        sc.tl.score_genes(adata_sub, gene_list=available, score_name=f'score_{ct}')

major_types = ['CD4_T_cell', 'CD8_T_cell', 'NK_cell', 'B_cell', 'Monocyte']
score_cols = [f'score_{ct}' for ct in major_types if f'score_{ct}' in adata_sub.obs.columns]
score_df = adata_sub.obs[score_cols]
adata_sub.obs['cell_type'] = score_df.idxmax(axis=1).str.replace('score_', '')
adata_sub.obs.loc[score_df.max(axis=1) < 0.1, 'cell_type'] = 'Other'

adata_sub.obs['subcluster'] = 'Other'
for major in major_types:
    mask = adata_sub.obs['cell_type'] == major
    if not mask.any():
        continue
    if major == 'CD4_T_cell':
        subs = ['CD4_Naive', 'CD4_Memory']
    elif major == 'CD8_T_cell':
        subs = ['CD8_Naive', 'CD8_Effector', 'CD8_EM']
    elif major == 'NK_cell':
        if 'score_NKT' in adata_sub.obs.columns:
            nkt = mask & (adata_sub.obs['score_NKT'] > adata_sub.obs.get('score_NK_cell', 0))
            adata_sub.obs.loc[nkt, 'subcluster'] = 'NKT'
            adata_sub.obs.loc[mask & ~nkt, 'subcluster'] = 'NK'
        else:
            adata_sub.obs.loc[mask, 'subcluster'] = 'NK'
        continue
    elif major == 'Monocyte':
        subs = ['CD14_Mono', 'CD16_Mono']
    elif major == 'B_cell':
        subs = ['B_Naive', 'B_Memory']
    else:
        continue
    sub_scores = [f'score_{s}' for s in subs if f'score_{s}' in adata_sub.obs.columns]
    if sub_scores:
        sub_df = adata_sub.obs.loc[mask, sub_scores]
        adata_sub.obs.loc[mask, 'subcluster'] = sub_df.idxmax(axis=1).str.replace('score_', '')

print("Cell type distribution (subsample):")
print(adata_sub.obs['cell_type'].value_counts())
print("\nSubcluster distribution (subsample):")
print(adata_sub.obs['subcluster'].value_counts())

adata_sub.write(f"{OUTDIR}/reference_subsample.h5ad")
print(f"Reference saved: {adata_sub.n_obs:,} cells")

print("\n" + "=" * 60)
print("Phase 4: Transfer labels to all cells (batch processing)")
print("=" * 60)

all_obs = []
for i, sample_id in enumerate(sorted(sample_cells.keys())):
    adata = sc.read_h5ad(f"{TMPDIR}/{sample_id}.h5ad")

    for ct, genes in markers.items():
        available = [g for g in genes if g in adata.var_names]
        if available:
            sc.tl.score_genes(adata, gene_list=available, score_name=f'score_{ct}')

    score_cols_local = [f'score_{ct}' for ct in major_types if f'score_{ct}' in adata.obs.columns]
    if score_cols_local:
        sdf = adata.obs[score_cols_local]
        adata.obs['cell_type'] = sdf.idxmax(axis=1).str.replace('score_', '')
        adata.obs.loc[sdf.max(axis=1) < 0.1, 'cell_type'] = 'Other'

    adata.obs['subcluster'] = 'Other'
    for major in major_types:
        mask = adata.obs['cell_type'] == major
        if not mask.any():
            continue
        if major == 'CD4_T_cell':
            subs = ['CD4_Naive', 'CD4_Memory']
        elif major == 'CD8_T_cell':
            subs = ['CD8_Naive', 'CD8_Effector', 'CD8_EM']
        elif major == 'NK_cell':
            if 'score_NKT' in adata.obs.columns:
                nkt = mask & (adata.obs['score_NKT'] > adata.obs.get('score_NK_cell', 0))
                adata.obs.loc[nkt, 'subcluster'] = 'NKT'
                adata.obs.loc[mask & ~nkt, 'subcluster'] = 'NK'
            else:
                adata.obs.loc[mask, 'subcluster'] = 'NK'
            continue
        elif major == 'Monocyte':
            subs = ['CD14_Mono', 'CD16_Mono']
        elif major == 'B_cell':
            subs = ['B_Naive', 'B_Memory']
        else:
            continue
        sub_scores = [f'score_{s}' for s in subs if f'score_{s}' in adata.obs.columns]
        if sub_scores:
            sub_df = adata.obs.loc[mask, sub_scores]
            adata.obs.loc[mask, 'subcluster'] = sub_df.idxmax(axis=1).str.replace('score_', '')

    all_obs.append(adata.obs[['sample', 'batch', 'cell_type', 'subcluster',
                               'n_genes_by_counts', 'total_counts', 'pct_counts_mt']].copy())

    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(sample_cells)}] {sample_id}: {adata.n_obs} cells")

    del adata
    gc.collect()

print("Combining metadata...")
all_meta = pd.concat(all_obs)
print(f"Total: {len(all_meta):,} cells")

print("\n" + "=" * 60)
print("Phase 5: Save results")
print("=" * 60)

all_meta.to_csv(f"{OUTDIR}/cell_metadata.csv")
print(f"Metadata saved: {len(all_meta):,} cells")

print("\nCell type distribution (all cells):")
print(all_meta['cell_type'].value_counts())
print("\nSubcluster distribution (all cells):")
print(all_meta['subcluster'].value_counts())

print("\nSaving by_cluster h5ads...")
for ct in all_meta['cell_type'].unique():
    ct_cells = all_meta[all_meta['cell_type'] == ct].index.tolist()
    ct_adatas = []

    ct_by_sample = all_meta[all_meta['cell_type'] == ct].groupby('sample')
    for sample_id, group in ct_by_sample:
        adata = sc.read_h5ad(f"{TMPDIR}/{sample_id}.h5ad")
        common = list(set(group.index) & set(adata.obs_names))
        if common:
            ct_adatas.append(adata[common].copy())
        del adata
    if ct_adatas:
        ct_adata = sc.concat(ct_adatas, join='outer')
        ct_adata.obs['cell_type'] = ct
        safe_name = ct.replace(' ', '_').replace('/', '_')
        ct_adata.write(f"{OUTDIR}/by_cluster/{safe_name}.h5ad")
        print(f"  {ct}: {ct_adata.n_obs:,} cells")
        del ct_adata, ct_adatas
        gc.collect()

print("\nSaving by_subcluster h5ads...")
for sc_name in all_meta['subcluster'].unique():
    sc_cells = all_meta[all_meta['subcluster'] == sc_name]
    sc_adatas = []
    sc_by_sample = sc_cells.groupby('sample')
    for sample_id, group in sc_by_sample:
        adata = sc.read_h5ad(f"{TMPDIR}/{sample_id}.h5ad")
        common = list(set(group.index) & set(adata.obs_names))
        if common:
            sc_adatas.append(adata[common].copy())
        del adata
    if sc_adatas:
        sc_adata = sc.concat(sc_adatas, join='outer')
        sc_adata.obs['subcluster'] = sc_name
        safe_name = sc_name.replace(' ', '_').replace('/', '_')
        sc_adata.write(f"{OUTDIR}/by_subcluster/{safe_name}.h5ad")
        print(f"  {sc_name}: {sc_adata.n_obs:,} cells")
        del sc_adata, sc_adatas
        gc.collect()

print("\nGenerating figures from reference subsample...")
adata_ref = sc.read_h5ad(f"{OUTDIR}/reference_subsample.h5ad")
fig_dir = f"{OUTDIR}/figures"
sc.settings.figdir = fig_dir

sc.pl.umap(adata_ref, color='cell_type', title='Cell Type (subsample)',
           save=f'_cell_type_{TS}.pdf', show=False)
sc.pl.umap(adata_ref, color='subcluster', title='Subcluster (subsample)',
           save=f'_subcluster_{TS}.pdf', show=False)
sc.pl.umap(adata_ref, color='batch', title='Batch (subsample)',
           save=f'_batch_{TS}.pdf', show=False)

marker_genes = ['CD3D', 'CD4', 'CD8A', 'NKG7', 'GNLY', 'MS4A1', 'CD79A',
                'CD14', 'LYZ', 'FCGR3A', 'CCR7', 'GZMB', 'PRF1']
available = [g for g in marker_genes if g in adata_ref.var_names]
if available:
    sc.pl.dotplot(adata_ref, var_names=available, groupby='cell_type',
                  save=f'_markers_celltype_{TS}.pdf', show=False)
    sc.pl.dotplot(adata_ref, var_names=available, groupby='subcluster',
                  save=f'_markers_subcluster_{TS}.pdf', show=False)

print(f"\nDONE: {len(all_meta):,} cells, {all_meta['cell_type'].nunique()} cell types, {all_meta['subcluster'].nunique()} subclusters")


import scanpy as sc
import pandas as pd
import os
import glob
import sys

OUTDIR = os.path.expanduser("~/projects/lnc_project_scafe/results/04_gene_expression")
SRC_META = f"{OUTDIR}/cell_metadata.csv"
SRC_DIR = f"{OUTDIR}/by_cluster"
DST_DIR = f"{OUTDIR}/by_cluster_aida_aligned"
os.makedirs(DST_DIR, exist_ok=True)

TS = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

CELL_TYPE_MAP = {
    'CD4_T_cell': 'CD4',
    'CD8_T_cell': 'CD8',
    'NK_cell':    'NK',
    'B_cell':     'B',
    'Monocyte':   'Monocyte',
    'Other':      'unassigned',
}

def main():
    if not os.path.exists(SRC_META):
        sys.exit(f"ERROR: {SRC_META} not found. annot28 完走前に実行された?")

    print(f"=== AIDA 整合化 ({TS}) ===")
    print(f"Source meta: {SRC_META}")

    meta = pd.read_csv(SRC_META, index_col=0)
    print(f"Loaded metadata: {len(meta):,} cells, columns={list(meta.columns)}")

    if 'cell_type' not in meta.columns:
        sys.exit("ERROR: cell_type 列が無い")

    meta['cell_type_aida'] = meta['cell_type'].map(CELL_TYPE_MAP).fillna('unassigned')
    dst_meta = f"{OUTDIR}/cell_metadata_aida_aligned_{TS}.csv"
    meta.to_csv(dst_meta)
    print(f"Saved: {dst_meta}")
    print("\ncell_type_aida 分布:")
    print(meta['cell_type_aida'].value_counts())

    print(f"\n=== by_cluster h5ad rename → {DST_DIR}/ ===")
    for src_h5ad in sorted(glob.glob(f"{SRC_DIR}/*.h5ad")):
        src_name = os.path.basename(src_h5ad).replace('.h5ad', '')
        aida_name = CELL_TYPE_MAP.get(src_name, src_name)

        adata = sc.read_h5ad(src_h5ad)
        if 'cell_type' in adata.obs.columns:
            mapped = adata.obs['cell_type'].astype(str).map(CELL_TYPE_MAP)
            mapped = mapped.fillna('unassigned')
            adata.obs['cell_type_aida'] = mapped.astype('category')
        else:
            adata.obs['cell_type_aida'] = aida_name

        dst_h5ad = f"{DST_DIR}/{aida_name}.h5ad"
        adata.write(dst_h5ad)
        print(f"  {src_name} → {aida_name}: {adata.n_obs:,} cells")
        del adata

    print(f"\nDONE: AIDA aligned outputs at {DST_DIR}/")
    print(f"      Metadata: {dst_meta}")

if __name__ == '__main__':
    main()


import os, glob, sys
from datetime import datetime
import pandas as pd

TS = datetime.now().strftime('%Y%m%d_%H%M%S')
BASE = '${PIPELINE_DIR}'
NAS  = '${DATA_DIR}/lnc_project_synapse_dataset/syn49637038'

CELL_META = sorted(glob.glob(f'{BASE}/results/04_gene_expression/cell_metadata_aida_aligned_*.csv'))[-1]
DEMUX_DIR = f'{NAS}/02_metadata/demultiplexing'
SAMPLE_META = f'{NAS}/02_metadata/complete_sample_metadata.csv'
OUT = f'{BASE}/results/04_gene_expression/cell_metadata_with_donor_{TS}.csv'
LOG = f'{BASE}/logs/32_add_donor_{TS}.log'

def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

os.makedirs(os.path.dirname(LOG), exist_ok=True)

log(f"loading cell_metadata: {CELL_META}")
cm = pd.read_csv(CELL_META, index_col=0)
log(f"  shape: {cm.shape}")

log("building barcode -> Tube_id map from demux csv ...")
bc_map = {}
demux_loaded = set()
for batch_dir in sorted(os.listdir(DEMUX_DIR)):
    bd = f'{DEMUX_DIR}/{batch_dir}'
    if not os.path.isdir(bd):
        continue
    for csv in glob.glob(f'{bd}/ALAW-*_assignment_cell_patient.csv'):
        sid = os.path.basename(csv).replace('ALAW-', '').replace('_assignment_cell_patient.csv', '')
        df = pd.read_csv(csv)
        if not {'barcode', 'patient'}.issubset(df.columns):
            log(f"  WARN: {csv} missing barcode/patient columns")
            continue
        demux_loaded.add(sid)
        for bc, pat in zip(df['barcode'].astype(str), df['patient'].astype(str)):
            bc_map[(sid, bc)] = pat
log(f"  demux loaded for {len(demux_loaded)} samples, {len(bc_map):,} (sample,barcode) entries")

log(f"loading sample metadata: {SAMPLE_META}")
sm = pd.read_csv(SAMPLE_META)
sm_map = {}
for _, r in sm.iterrows():
    sm_map[(r['Batch'], r['Tube_id'])] = (r['Donor_id'], r['Age'], r['Sex'], r['Age_group'])
log(f"  {len(sm_map)} (Batch, Tube_id) entries (167 unique Donor_id, 14 Batch)")

log("annotating cells ...")
tube_ids, donor_ids, ages, sexes, ag, status = [], [], [], [], [], []
n_cells = len(cm)
n_ok = n_covid = n_doublet = n_unassigned = n_no_demux = n_no_meta = 0
for idx, sample, batch in zip(cm.index, cm['sample'].astype(str), cm['batch'].astype(str)):

    bc = idx[len(sample)+1:]
    if sample not in demux_loaded:
        st = 'no_demux'
        tube_ids.append(None); donor_ids.append(None); ages.append(None); sexes.append(None); ag.append(None)
        n_no_demux += 1
    else:
        pat = bc_map.get((sample, bc))
        if pat is None:

            st = 'unassigned'; tube_ids.append(None); donor_ids.append(None); ages.append(None); sexes.append(None); ag.append(None)
            n_unassigned += 1
        elif pat == 'covid':
            st = 'covid'; tube_ids.append(pat); donor_ids.append(None); ages.append(None); sexes.append(None); ag.append(None)
            n_covid += 1
        elif pat == 'doublet':
            st = 'doublet'; tube_ids.append(None); donor_ids.append(None); ages.append(None); sexes.append(None); ag.append(None)
            n_doublet += 1
        elif pat == 'unassigned':
            st = 'unassigned'; tube_ids.append(None); donor_ids.append(None); ages.append(None); sexes.append(None); ag.append(None)
            n_unassigned += 1
        else:
            meta = sm_map.get((batch, pat))
            if meta is None:
                st = 'no_meta'; tube_ids.append(pat); donor_ids.append(None); ages.append(None); sexes.append(None); ag.append(None)
                n_no_meta += 1
            else:
                d, a, s, agroup = meta
                st = 'ok'; tube_ids.append(pat); donor_ids.append(d); ages.append(a); sexes.append(s); ag.append(agroup)
                n_ok += 1
    status.append(st)

cm['Tube_id']  = tube_ids
cm['Donor_id'] = donor_ids
cm['Age']      = ages
cm['Sex']      = sexes
cm['Age_group'] = ag
cm['patient_status'] = status

log(f"status counts:")
log(f"  ok={n_ok:,}  covid={n_covid:,}  doublet={n_doublet:,}  unassigned={n_unassigned:,}  no_demux={n_no_demux:,}  no_meta={n_no_meta:,}")
log(f"  total={n_cells:,}  ok pct={n_ok/n_cells*100:.1f}%")

cm.to_csv(OUT)
log(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")

ok = cm[cm['patient_status']=='ok']
log(f"\nOK cells by Donor count:  donors={ok['Donor_id'].nunique()}  (Batch,Donor) groups={ok.groupby(['batch','Donor_id']).ngroups}")
n_per_donor = ok.groupby('Donor_id')['batch'].nunique()
log(f"  donors with 1 timepoint: {(n_per_donor==1).sum()}")
log(f"  donors with 2 timepoints: {(n_per_donor==2).sum()}")
log(f"  donors with 3 timepoints: {(n_per_donor==3).sum()}")
log("DONE")


import h5py, numpy as np, pandas as pd
W='${SCRATCH}'

m=pd.read_csv('${PROJECT_DIR}/results/phase3b_cre_corr/'
              'barcode_metadata_latest.tsv',sep='\t',usecols=['cohort','barcode_full','CellType','Donor_id'])
p3_cd4=set(m[(m.cohort=='AIDA')&(m.CellType=='CD4_T_cell')].barcode_full)
print(f'phase3b AIDA CD4: {len(p3_cd4):,}',flush=True)

f=h5py.File('${DATA_DIR}/lnc_project_AIDA_dataset/all_cells_clustered/all_cells_clustered.h5ad','r')
oi=f['obs'].attrs.get('_index','_index'); vi=f['var'].attrs.get('_index','_index')
bc=np.array([x.decode() if isinstance(x,bytes) else x for x in f['obs'][oi][:]])
genes=np.array([x.decode() if isinstance(x,bytes) else x for x in f['var'][vi][:]])
g2i={g:i for i,g in enumerate(genes)}

sel=np.isin(bc,list(p3_cd4))
print(f'  all_cells 内で見つかった: {sel.sum():,}',flush=True)

le=f['obs']['leiden']; lc=[x.decode() if isinstance(x,bytes) else x for x in le['categories'][:]]
leiden=np.array([lc[c] for c in le['codes'][:]])[sel]
idx=np.where(sel)[0]
MK=['CCR7','SELL','LEF1','TCF7','IL7R','S100A4','GZMK','GZMA','CCL5','KLRB1']
mk=[g for g in MK if g in g2i]
cols={g:np.zeros(len(idx),np.float32) for g in mk}; tot=np.zeros(len(idx),np.float32)
ip=f['X']['indptr']; dat=f['X']['data']; ind=f['X']['indices']
for jj,r in enumerate(idx):
    a,b=ip[r],ip[r+1]
    if b<=a: continue
    gi=ind[a:b]; dv=dat[a:b]; tot[jj]=dv.sum()
    for g in mk:
        h=gi==g2i[g]
        if h.any(): cols[g][jj]=dv[h][0]
f.close()
cpm={g:np.log1p(cols[g]/np.maximum(tot,1)*1e4) for g in mk}
bcs=bc[sel]
df=pd.DataFrame({'barcode_full':bcs,'leiden':leiden,**cpm})
g=df.groupby('leiden').agg(n=('leiden','size'),**{k:(k,'median') for k in mk})
g['割合%']=(100*g.n/g.n.sum()).round(1)
def rule(r):
    if r[mk].max()<0.3: return 'Exclude'
    if r['CCR7']>0.5 and r['SELL']>0.5: return 'Naive'
    return 'Memory'
g['grp']=g.apply(rule,axis=1)
pd.set_option('display.width',200,'display.max_columns',20)
print('\n=== AIDA CD4 (phase3b 細胞集合) クラスタ別マーカー ===')
print(g[['n','割合%','CCR7','SELL','TCF7','IL7R','GZMA','grp']].sort_values('CCR7',ascending=False).round(2).to_string())
df['our_cd4']=df.leiden.map(g.grp.to_dict())
df[['barcode_full','leiden','our_cd4']].to_parquet(f'{W}/aida_cd4_labels_p3.parquet')
print('\n付与:'); 
for a in ['Naive','Memory','Exclude']:
    n=(df.our_cd4==a).sum(); print(f'  {a}: {n:,} ({100*n/len(df):.0f}%)')
print(f'→ aida_cd4_labels_p3.parquet ({len(df):,})')


import scanpy as sc, numpy as np, pandas as pd, anndata as ad
W='${SCRATCH}'
print('読み込み...',flush=True)
As=[]
for state in ['Naive','Memory']:
    a=sc.read_h5ad(f'${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/'
                   f'04_gene_expression/by_subcluster/CD4_{state}.h5ad')
    a.obs['old_sub']=f'CD4_{state}'; As.append(a)
A=ad.concat(As); del As
print(f'Synapse CD4: {A.n_obs:,} 細胞',flush=True)
counts=A.X.copy()
tot=np.asarray(counts.sum(1)).ravel()

A.X=counts.copy(); sc.pp.normalize_total(A,target_sum=1e4); sc.pp.log1p(A)
sc.pp.highly_variable_genes(A,n_top_genes=2000)
Ah=A[:,A.var.highly_variable].copy()
sc.pp.scale(Ah,max_value=10); sc.tl.pca(Ah,n_comps=50)
print('近傍+Leiden...',flush=True)
sc.pp.neighbors(Ah,n_neighbors=15,n_pcs=50); sc.tl.leiden(Ah,resolution=1.0)
A.obs['leiden']=Ah.obs.leiden.values
print(f'クラスタ数: {A.obs.leiden.nunique()}',flush=True)

MK=['CCR7','SELL','LEF1','TCF7','IL7R','S100A4','GZMK','GZMA','CCL5','KLRB1']
vi={g:i for i,g in enumerate(A.var_names)}
mk={}
for g in MK:
    if g in vi:
        c=np.asarray(counts[:,vi[g]].todense()).ravel()
        mk[g]=np.log1p(c/np.maximum(tot,1)*1e4)
    else: print(f'  欠落: {g}')
naive_g=[g for g in ['CCR7','SELL','LEF1','TCF7'] if g in mk]
mem_g=[g for g in ['IL7R','S100A4','GZMK','GZMA','CCL5','KLRB1'] if g in mk]
df=pd.DataFrame({'leiden':A.obs.leiden.values,**mk})
df['naive']=df[naive_g].mean(1); df['mem']=df[mem_g].mean(1); df['score']=df.naive-df.mem
g=df.groupby('leiden').agg(n=('score','size'),score=('score','median'),
    CCR7=('CCR7','median'),SELL=('SELL','median'),LEF1=('LEF1','median'),
    TCF7=('TCF7','median'),IL7R=('IL7R','median'),S100A4=('S100A4','median'),
    GZMK=('GZMK','median'),GZMA=('GZMA','median')).sort_values('score',ascending=False)
g['割合%']=(100*g.n/g.n.sum()).round(1)
pd.set_option('display.width',200,'display.max_columns',20)
print('\n=== Synapse CD4 Leiden クラスタ別マーカー (AIDA と同一式) ===')
print(g.round(2).to_string())

out=A.obs[['old_sub','leiden']].copy(); out['barcode_full']=A.obs_names
out['Donor_id']=A.obs['sample'].values if 'sample' in A.obs else pd.Series(A.obs_names).str.rsplit('_',n=1).str[0].values
out.to_parquet(f'{W}/synapse_cd4_clusters.parquet')
g.to_csv(f'{W}/synapse_cd4_cluster_markers.csv')
print(f'\n保存: synapse_cd4_clusters.parquet / synapse_cd4_cluster_markers.csv')


import h5py, numpy as np, pandas as pd, glob
W='${SCRATCH}'
FILT=['CD14','LYZ','KLRD1','GNLY']
def marker_cpm(fp, want_bc, mk):
    f=h5py.File(fp,'r')
    oi=f['obs'].attrs.get('_index','_index'); vi=f['var'].attrs.get('_index','_index')
    bc=np.array([x.decode() if isinstance(x,bytes) else x for x in f['obs'][oi][:]])
    genes=np.array([x.decode() if isinstance(x,bytes) else x for x in f['var'][vi][:]])
    g2i={g:i for i,g in enumerate(genes)}
    sel=np.isin(bc,list(want_bc)); idx=np.where(sel)[0]
    have=[g for g in mk if g in g2i]
    cols={g:np.zeros(len(idx),np.float32) for g in have}; tot=np.zeros(len(idx),np.float32)
    ip=f['X']['indptr']; dat=f['X']['data']; ind=f['X']['indices']
    for jj,r in enumerate(idx):
        a,b=ip[r],ip[r+1]
        if b<=a: continue
        gi=ind[a:b]; dv=dat[a:b]; tot[jj]=dv.sum()
        for g in have:
            h=gi==g2i[g]
            if h.any(): cols[g][jj]=dv[h][0]
    f.close()
    out=pd.DataFrame({g:np.log1p(cols[g]/np.maximum(tot,1)*1e4) for g in have})
    out['barcode_full']=bc[sel]
    return out

A=pd.read_parquet(f'{W}/aida_cd4_labels_p3.parquet')
S=pd.read_parquet(f'{W}/synapse_cd4_labels.parquet')
print('汚染マーカー抽出中...',flush=True)
ca=marker_cpm('${DATA_DIR}/lnc_project_AIDA_dataset/all_cells_clustered/all_cells_clustered.h5ad',
              set(A.barcode_full),FILT)
cs=[]
for fp in sorted(glob.glob('${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/04_gene_expression/by_subcluster/CD4_*.h5ad')):
    cs.append(marker_cpm(fp,set(S.barcode_full),FILT)); print(f'  {fp.split("/")[-1]}',flush=True)
cs=pd.concat(cs,ignore_index=True)

def flag(c):
    bad=(c.get('CD14',0)>1)|(c.get('LYZ',0)>1.5)|((c.get('KLRD1',0)>1)&(c.get('GNLY',0)>1))
    return bad
A=A.merge(ca,on='barcode_full',how='left'); A['bad']=flag(A[[c for c in ['CD14','LYZ','KLRD1','GNLY'] if c in A]].fillna(0))
S=S.merge(cs,on='barcode_full',how='left'); S['bad']=flag(S[[c for c in ['CD14','LYZ','KLRD1','GNLY'] if c in S]].fillna(0))
for nm,d in [('AIDA',A),('Synapse',S)]:
    print(f'\n{nm}: {len(d):,} 細胞')
    for g in ['Naive','Memory']:
        sub=d[d.our_cd4==g]; print(f'  {g}: 除外 {int(sub.bad.sum()):,}/{len(sub):,} ({100*sub.bad.mean():.1f}%)')

A.loc[A.bad,'our_cd4']='Exclude'; S.loc[S.bad,'our_cd4']='Exclude'
A[['barcode_full','leiden','our_cd4']].to_parquet(f'{W}/aida_cd4_final.parquet')
S[['barcode_full','leiden','our_cd4']].to_parquet(f'{W}/synapse_cd4_final.parquet')
print('\n=== フィルタ後 確定 ===')
for nm,d in [('AIDA',A),('Synapse',S)]:
    print(f'{nm}:',dict(d.our_cd4.value_counts()))
print('→ aida_cd4_final.parquet / synapse_cd4_final.parquet')


import pickle
import numpy as np, pandas as pd

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
CANON = f'{D}/results/canonical_labels/barcode_subset_canonical.parquet'

def canonical_counts(cohort='AIDA'):
    c = pd.read_parquet(CANON, columns=['cohort', 'Donor_id', 'subset_canonical'])
    c = c[c.cohort == cohort]
    o = c.groupby(['Donor_id', 'subset_canonical']).size().rename('n_can').reset_index()
    o.columns = ['donor', 'subset', 'n_can']
    return o

def verify(df, donor_col='donor', subset_col='subset', n_col='n_cells', cohort='AIDA',
           strict=True, verbose=True):
    d = df.rename(columns={donor_col: 'donor', subset_col: 'subset', n_col: 'n'})[['donor', 'subset', 'n']]
    m = d.merge(canonical_counts(cohort), on=['donor', 'subset'], how='inner')
    bad = []
    for s, g in m.groupby('subset'):
        if len(g) < 20: continue
        r = np.corrcoef(g.n, g.n_can)[0, 1]; ratio = g.n.sum() / g.n_can.sum()
        ok = abs(ratio - 1) < 0.02 and r > 0.99
        if verbose:
            print(f'  {s:14s} n={len(g):3d}  合計比 {ratio:5.2f}  相関 {r:6.3f}'
                  f'{"" if ok else "  ★不一致"}')
        if not ok: bad.append(s)
    if bad:
        msg = ('細胞ラベルが正典と一致しない: ' + ', '.join(bad) +
               '\n  → canonical_labels/barcode_subset_canonical.parquet の'
               ' subset_canonical を使うこと（README_WARNING_subcluster.md 参照）')
        if strict: raise AssertionError(msg)
        print('WARNING: ' + msg)
    return not bad

if __name__ == '__main__':
    print('=== barcode_metadata_latest.tsv の subcluster を照合（不一致が出るのが正しい）===')
    bm = pd.read_csv(f'{B}/phase3b_cre_corr/barcode_metadata_latest.tsv', sep='\t',
                     usecols=['cohort', 'Donor_id', 'subcluster'])
    bm = bm[bm.cohort == 'AIDA']
    cnt = bm.groupby(['Donor_id', 'subcluster']).size().rename('n_cells').reset_index()
    cnt.columns = ['donor', 'subset', 'n_cells']
    try:
        verify(cnt)
        print('\n(不一致が出なかった。想定外)')
    except AssertionError as e:
        print(f'\n想定どおり検出: {e}')


import pickle, os, gzip
import numpy as np, pandas as pd
from scipy.stats import spearmanr

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUR = f'{B}/phase3b_ourcd4'
OUT = f'{D}/results/paper_additions'
os.makedirs(OUT, exist_ok=True)
TV = 'yugene_diff_EN_tAge'
GTF = '${DATA_DIR}/adachi/reference/GRCh38.113.gtf.gz'
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

ann = pd.read_csv(f'{B}/phase6_novel_enhancer/novel_enhancer_classification.csv',
                  usecols=['cre_id', 'proximity']).drop_duplicates('cre_id')

ts = pd.read_csv('${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/06_tage/'
                 'subcluster/tage_all_combined_v2.csv')
ts = ts[(ts.model_type == 'Chronoage') & (ts['mode'] == 'EN')]
base = (ts[['Donor_id', 'Batch', 'Age']].drop_duplicates()
        .sort_values(['Donor_id', 'Age']).groupby('Donor_id').first().reset_index())
bkey = set(zip(base.Donor_id, base.Batch))
ts_b = ts[[(x, y) in bkey for x, y in zip(ts.Donor_id, ts.Batch)]]
meta_syn = ts_b[['Donor_id', 'Batch', 'CellType', 'Age', TV]].rename(columns={TV: 'pace'})
meta_syn['cohort'] = 'Synapse'
ta = pd.read_csv('${DATA_DIR}/lnc_project_AIDA_dataset/by_subcluster/tage/'
                 'tage_all_combined_v1.csv')
ta = ta[(ta.model_type == 'Chronoage') & (ta['mode'] == 'EN')]
meta_aida = ta[['Sample', 'Subcluster', 'age', TV]].rename(
    columns={'Sample': 'Donor_id', 'Subcluster': 'CellType', 'age': 'Age', TV: 'pace'})
meta_aida['Batch'] = meta_aida.Donor_id
meta_aida['cohort'] = 'AIDA'
META = pd.concat([meta_aida, meta_syn], ignore_index=True)

rob = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv',
                  usecols=['cre_id', 'cell_type', 'direction', 'program'])

FR_ROWS = []
STRAT_ROWS = []
for coh, pkl in [('AIDA', f'{OUR}/aggregation_cache_aida_subcluster.pkl'),
                 ('Synapse', f'{OUR}/aggregation_cache_synapse_subcluster.pkl')]:
    w(f'\n===== {coh} キャッシュ =====')
    d = pickle.load(open(pkl, 'rb'))
    cres = np.array(d['cres']); keys = list(d['expr_sum'].keys())
    pmap = dict(zip(ann.cre_id, ann.proximity))
    prox_arr = np.array([pmap.get(c, '') for c in cres])
    is_dist = prox_arr == 'distal'
    cre_idx = {c: i for i, c in enumerate(cres)}
    mcm = pd.DataFrame(keys, columns=['cohort_k', 'Donor_id', 'Batch', 'CellType'])
    mcm['n_cells'] = [d['n_cells'][k] for k in keys]
    mcm = mcm.merge(META[META.cohort == coh], on=['Donor_id', 'Batch', 'CellType'], how='inner')
    mcm = mcm[mcm.n_cells >= 10].dropna(subset=['pace'])

    for i in mcm.index:
        v = np.asarray(d['expr_sum'][keys[i]], dtype=np.float64)
        tot = v.sum()
        if tot < 1: continue
        r = mcm.loc[i]
        FR_ROWS.append(dict(cohort=coh, Donor_id=r.Donor_id, CellType=r.CellType,
                            Age=r.Age, pace=r.pace, dist_frac=v[is_dist].sum() / tot))

    for ct in sorted(mcm.CellType.unique()):
        sub = mcm[mcm.CellType == ct]
        rr = rob[rob.cell_type == ct]
        if len(rr) < 30: continue
        ix = np.array([cre_idx[c] for c in rr.cre_id if c in cre_idx], dtype=int)
        rr = rr[rr.cre_id.isin(set(cres[ix]))].reset_index(drop=True)
        for lab, lo, hi in [('young(<45)', 0, 45), ('mid(45-64)', 45, 65), ('old(65+)', 65, 200)]:
            s2 = sub[(sub.Age >= lo) & (sub.Age < hi)]
            if len(s2) < 30: continue
            M = np.vstack([d['expr_sum'][keys[i]] for i in s2.index]).astype(np.float64)
            tot = M.sum(1, keepdims=True); tot[tot == 0] = 1
            M = np.log2(M / tot * 1e6 + 1)[:, ix]
            from scipy.stats import rankdata
            rp = rankdata(s2.pace.values)
            rM = np.apply_along_axis(rankdata, 0, M)
            rpc = rp - rp.mean(); rMc = rM - rM.mean(0)
            den = np.sqrt((rMc**2).sum(0) * (rpc**2).sum()); den[den == 0] = np.nan
            rho = (rMc * rpc[:, None]).sum(0) / den
            for prog, sgn in [('up_distal', 1), ('down_proximal', -1)]:
                m_ = (rr.program == prog).values
                if m_.sum() < 20: continue
                ok = np.isfinite(rho[m_])
                keep = 100 * (np.sign(rho[m_][ok]) == sgn).mean()
                STRAT_ROWS.append(dict(cohort=coh, CellType=ct, stratum=lab, program=prog,
                                       n_donor=len(s2), n_cre=int(m_.sum()), sign_keep=keep))
    del d

FR = pd.DataFrame(FR_ROWS)
FR.to_csv(f'{OUT}/distal_fraction_per_metacell.csv', index=False)
w('\n' + '=' * 92)
w('[1] 最少仮定の 1 数値 — 遠位読み割合と老化速度（横断、細胞種内 Spearman）')
w('=' * 92)
res1 = []
for (coh, ct), g in FR.groupby(['cohort', 'CellType']):
    if len(g) < 60: continue
    r1 = spearmanr(g.dist_frac, g.pace); r2 = spearmanr(g.dist_frac, g.Age)
    res1.append(dict(cohort=coh, CellType=ct, n=len(g),
                     rho_pace=r1.statistic, p_pace=r1.pvalue, rho_age=r2.statistic))
R1 = pd.DataFrame(res1)
R1.to_csv(f'{OUT}/distal_fraction_correlations.csv', index=False)
for coh in ['AIDA', 'Synapse']:
    x = R1[R1.cohort == coh]
    w(f'\n{coh}: 細胞種 {len(x)} / 老化速度と正 {int((x.rho_pace>0).sum())}/{len(x)} / '
      f'rho 中央 {x.rho_pace.median():+.3f}（暦年齢 rho 中央 {x.rho_age.median():+.3f}）')
    for _, r in x.sort_values('rho_pace', ascending=False).head(4).iterrows():
        w(f'  {r.CellType:13s} rho(速度) {r.rho_pace:+.3f} (p={r.p_pace:.2g})  rho(年齢) {r.rho_age:+.3f}  n={r.n}')
don = FR.groupby(['cohort', 'Donor_id']).agg(frac=('dist_frac', 'mean'),
                                             pace=('pace', 'mean'), Age=('Age', 'first')).reset_index()
don.to_csv(f'{OUT}/distal_fraction_per_donor.csv', index=False)
for coh in ['AIDA', 'Synapse']:
    x = don[don.cohort == coh]
    r = spearmanr(x.frac, x.pace)
    w(f'\n{coh} ドナー 1 値（全細胞種平均, n={len(x)}）: rho(速度) {r.statistic:+.3f} (p={r.pvalue:.2g})')

ST = pd.DataFrame(STRAT_ROWS)
ST.to_csv(f'{OUT}/strata_sign_preservation.csv', index=False)
w('\n' + '=' * 92)
w('[2] 年齢層別 — 頑健集合の符号は高齢層でも保たれるか（層内で相関を再計算）')
w('=' * 92)
for prog in ['up_distal', 'down_proximal']:
    x = ST[ST.program == prog]
    w(f'\n{prog}:')
    piv = x.pivot_table(index=['cohort', 'CellType'], columns='stratum',
                        values='sign_keep', aggfunc='first')
    w(piv.round(1).to_string())
w('\n  ★70/65 歳以上の列が 50% を大きく超えていれば、二プログラムは高齢でも走っている。')

w('\n' + '=' * 92)
w('[3] 三つの独立な見方すべてに顔を出す遺伝子')
w('=' * 92)
axis = pd.read_csv(f'{D}/results/pca_tage/axis_genes_shared_fast.csv')
axis_g = set(axis[axis.n_ct >= 4].gene.dropna())
w(f'PCA 軸（fast, 4 細胞種以上）: {len(axis_g)} 遺伝子')

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
def near(cid):
    try:
        ch, s, e, st = cid.rsplit('_', 3); mid = (int(s) + int(e)) // 2
    except Exception: return None
    a = tssd.get(ch)
    if not a: return None
    i = int(np.searchsorted(POS[ch], mid)); best, bd = None, None
    for j in (i - 1, i):
        if 0 <= j < len(a):
            dd = abs(a[j][0] - mid)
            if bd is None or dd < bd: bd, best = dd, a[j][1]
    return best

rob_up = rob[rob.direction == 'up']
rob_g = set(pd.Series([near(c) for c in rob_up.cre_id.unique()]).dropna())
w(f'横断の頑健 上昇 tCRE の最近接遺伝子: {len(rob_g)}')

mv = pd.read_csv(f'{D}/results/synchrony_elements/within_donor_cre_rho.csv.gz')
mv = mv[mv.rho.abs() >= 0.35]
mv_g = set(pd.Series([near(c) for c in mv[mv.rho > 0].cre_id.unique()]).dropna())
w(f'縦断で個人内 |rho|>=0.35 上昇の tCRE の最近接遺伝子: {len(mv_g)}')

tri = axis_g & rob_g & mv_g
w(f'\n★三者すべてに現れる遺伝子: {len(tri)} 個')
w('  ' + ', '.join(sorted(tri)))
pd.DataFrame(dict(gene=sorted(tri))).to_csv(f'{OUT}/triple_overlap_genes.csv', index=False)
for pair, gs in [('軸∩頑健', axis_g & rob_g), ('軸∩縦断', axis_g & mv_g), ('頑健∩縦断', rob_g & mv_g)]:
    w(f'  {pair}: {len(gs)}')

w('\n' + '=' * 92)
w('[4] 縦断 CD4 — 訪問単位 tAge（新資源）で Δ(tAge) を差し替え')
w('=' * 92)
sc = pd.read_csv(f'{D}/results/fixes/longitudinal_fix/metacell_scores.csv')
nt = pd.read_csv(f'{D}/results/cd4_visit_tage/cd4_our_tage_visit.csv')
nt = nt[nt.Donor_id.str.match(r'^[A-Z]\d')]
nt['tage_new_r'] = nt.yugene_diff_EN_tAge - nt.groupby('Batch').yugene_diff_EN_tAge.transform('mean')
db = ts[['Donor_id', 'Batch', 'Age']].drop_duplicates().sort_values(['Donor_id', 'Age'])
db['visit'] = db.groupby('Donor_id').cumcount() + 1
nt = nt.merge(db[['Donor_id', 'Batch', 'visit']], on=['Donor_id', 'Batch'], how='left')
for ct in ['CD4_Memory', 'CD4_Naive']:
    s_ = sc[(sc.CellType == ct) & sc.visit.isin([1, 2])]
    pv = s_.pivot_table(index='Donor_id', columns='visit', values=['up_r', 'dn_r', 'tage_res_r'])
    n_ = nt[(nt.CellType == ct) & nt.visit.isin([1, 2])]
    npv = n_.pivot_table(index='Donor_id', columns='visit', values='tage_new_r')
    npv.columns = pd.MultiIndex.from_tuples([('new', c) for c in npv.columns])
    m = pv.join(npv, how='inner').dropna()
    if not len(m): w(f'{ct}: 対応なし'); continue
    dU = m[('up_r', 2)] - m[('up_r', 1)]
    dDn = m[('dn_r', 2)] - m[('dn_r', 1)]
    dT_old = m[('tage_res_r', 2)] - m[('tage_res_r', 1)]
    dT_new = m[('new', 2)] - m[('new', 1)]
    ru_o = spearmanr(dU, dT_old); rd_o = spearmanr(dDn, dT_old)
    ru_n = spearmanr(dU, dT_new); rd_n = spearmanr(dDn, dT_new)
    rc = spearmanr(dT_old, dT_new)
    w(f'\n{ct} (n={len(m)}):')
    w(f'  Δ(旧 tAge) と Δ(新 tAge) の相関: {rc.statistic:+.3f}')
    w(f'  Δ上昇 × ΔtAge : 旧 {ru_o.statistic:+.3f} (p={ru_o.pvalue:.3g}) -> 新 {ru_n.statistic:+.3f} (p={ru_n.pvalue:.3g})')
    w(f'  Δ低下 × ΔtAge : 旧 {rd_o.statistic:+.3f} (p={rd_o.pvalue:.3g}) -> 新 {rd_n.statistic:+.3f} (p={rd_n.pvalue:.3g})')
w('\n  ★新版でも符号が保たれれば、CD4 の縦断連動は tAge の作り方に依らない。')
open(f'{OUT}/report.txt', 'w').write('\n'.join(L))
print(f'\n-> {OUT}')
