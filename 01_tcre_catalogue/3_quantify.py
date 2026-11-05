#!/usr/bin/env python3
import os, sys, glob, gzip
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

TS = datetime.now().strftime('%Y%m%d_%H%M%S')
PROJ = Path('${HOME}/projects/integrated_aida_synapse')
OUT_DIR = PROJ / 'results' / 'phase3b_cre_corr'
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / f'step1_log_{TS}.txt'

def log(msg):
    line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

AIDA_H5AD = '${DATA_DIR}/lnc_project_AIDA_dataset/all_cells_clustered/all_cells_clustered.h5ad'

def build_aida():
    log('=== AIDA: load h5ad and extract per-cell metadata ===')
    ad = sc.read_h5ad(AIDA_H5AD, backed='r')
    log(f'AIDA h5ad: {ad.shape}')

    obs = ad.obs[['sample', 'cell_type_corrected', 'subcluster']].copy()
    obs.index.name = 'barcode_full'

    obs['Donor_id'] = obs['sample']
    obs['cohort'] = 'AIDA'

    log(f'AIDA cells total: {len(obs)}')
    log(f'AIDA cell_type counts:\n{obs["cell_type_corrected"].value_counts()}')
    log(f'AIDA subcluster counts:\n{obs["subcluster"].value_counts()}')
    obs = obs.reset_index().rename(columns={
        'cell_type_corrected': 'CellType',
        'sample': 'sample',
    })
    obs['Batch'] = obs['sample']
    return obs[['cohort', 'sample', 'barcode_full', 'Donor_id', 'Batch', 'CellType', 'subcluster']]

SYN_DEMUX_DIR = '${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/02_metadata/demultiplexing'
SYN_META = '${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/02_metadata/complete_sample_metadata.csv'
SYN_H5AD_BY_CT = '${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/04_gene_expression/by_cluster'
SYN_METACELL = '${PIPELINE_DIR}/results/05_metacell/metacells_gene_celltype.h5ad'

NON_DONOR_LABELS = {'doublet', 'unassigned', 'ambiguous', 'covid', 'NA', 'nan', ''}

def build_synapse_assignment():
    log('=== Synapse: read all 106 assignment CSVs ===')
    rows = []
    files = sorted(glob.glob(f'{SYN_DEMUX_DIR}/*/ALAW-*_assignment_cell_patient.csv'))
    log(f'  assignment csvs: {len(files)}')
    for f in files:

        bn = os.path.basename(f)
        sample = bn.replace('ALAW-', '').replace('_assignment_cell_patient.csv', '')
        df = pd.read_csv(f)

        df = df.rename(columns={df.columns[0]: 'idx'}) if df.columns[0].startswith('Unnamed') or df.columns[0] == '' else df
        if 'patient' not in df.columns:
            log(f'  WARN: {f} missing patient col, cols={list(df.columns)}')
            continue
        df = df[['barcode', 'patient']].copy()
        df['sample'] = sample
        df['Batch'] = sample.split('-')[0]
        rows.append(df)
    asg = pd.concat(rows, ignore_index=True)
    log(f'  total assignments: {len(asg):,}')
    log(f'  unique patients: {asg["patient"].nunique()} (incl. non-donor)')
    log(f'  patient label samples: {sorted(asg["patient"].unique())[:30]}')

    n_before = len(asg)
    asg = asg[~asg['patient'].astype(str).str.lower().isin([s.lower() for s in NON_DONOR_LABELS])]
    asg = asg.dropna(subset=['patient'])
    log(f'  drop non-donor: {n_before - len(asg):,} dropped, {len(asg):,} remain')
    return asg

def build_synapse():
    log('=== Synapse: build full per-cell metadata ===')

    asg = build_synapse_assignment()

    asg['barcode_full'] = asg['sample'] + '_' + asg['barcode']

    log('Loading complete_sample_metadata.csv ...')
    meta = pd.read_csv(SYN_META)
    log(f'  {len(meta)} rows, cols: {list(meta.columns)}')

    meta = meta[['Batch', 'Tube_id', 'Donor_id', 'Age', 'Sex']].rename(columns={'Tube_id': 'patient'})
    asg = asg.merge(meta[['Batch', 'patient', 'Donor_id']], on=['Batch', 'patient'], how='left')
    n_no_donor = asg['Donor_id'].isna().sum()
    log(f'  cells with no Donor_id after Tube_id join: {n_no_donor:,} / {len(asg):,}')
    if n_no_donor > 0:
        log(f'  unmatched (Batch, patient) examples:')
        unm = asg[asg['Donor_id'].isna()][['Batch', 'patient']].drop_duplicates().head(10)
        for _, r in unm.iterrows():
            log(f'    {r.Batch} / {r.patient}')
    asg = asg.dropna(subset=['Donor_id'])
    log(f'  Synapse cells after donor join: {len(asg):,} ({asg["Donor_id"].nunique()} donors)')

    log('Loading Synapse by_cluster h5ads (obs only) ...')
    obs_chunks = []
    for f in sorted(glob.glob(f'{SYN_H5AD_BY_CT}/*.h5ad')):
        ad = sc.read_h5ad(f, backed='r')
        ob = ad.obs[['sample', 'cell_type']].copy()
        ob.index.name = 'barcode_full'
        log(f'  {os.path.basename(f)}: {len(ob):,} cells')
        obs_chunks.append(ob.reset_index())
    obs_ct = pd.concat(obs_chunks, ignore_index=True).drop_duplicates(subset=['barcode_full'])
    log(f'  total cells from by_cluster: {len(obs_ct):,}')

    log('Loading Synapse by_subcluster h5ads (obs only) ...')
    SYN_H5AD_BY_SC = '${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/04_gene_expression/by_subcluster'
    sc_chunks = []
    for f in sorted(glob.glob(f'{SYN_H5AD_BY_SC}/*.h5ad')):
        ad = sc.read_h5ad(f, backed='r')

        if 'subcluster' in ad.obs.columns:
            sub_col = 'subcluster'
        else:

            sub_col = None
        ob = pd.DataFrame({'barcode_full': ad.obs.index})
        if sub_col:
            ob['subcluster'] = ad.obs[sub_col].values
        else:
            ob['subcluster'] = os.path.basename(f).replace('.h5ad', '')
        sc_chunks.append(ob)
    obs_sc = pd.concat(sc_chunks, ignore_index=True).drop_duplicates(subset=['barcode_full'])
    log(f'  total cells from by_subcluster: {len(obs_sc):,}')

    full = asg[['cohort_dummy', 'sample', 'Batch', 'barcode_full', 'Donor_id', 'patient']].assign(cohort='Synapse') if 'cohort_dummy' in asg.columns else asg.assign(cohort='Synapse')
    full = full.merge(obs_ct[['barcode_full', 'cell_type']], on='barcode_full', how='inner')
    full = full.merge(obs_sc[['barcode_full', 'subcluster']], on='barcode_full', how='inner')
    full = full.rename(columns={'cell_type': 'CellType'})
    log(f'  Synapse final cells: {len(full):,}')
    log(f'  cell_type counts:\n{full["CellType"].value_counts()}')
    log(f'  subcluster counts:\n{full["subcluster"].value_counts()}')
    return full[['cohort', 'sample', 'barcode_full', 'Donor_id', 'Batch', 'CellType', 'subcluster']]

def validate_synapse(meta_full):
    log('\n=== VALIDATE: Synapse (Donor_id, Batch, CellType) n_cells vs metacell h5ad ===')
    ad = sc.read_h5ad(SYN_METACELL, backed='r')
    mc = ad.obs[['Donor_id', 'Batch', 'cell_type_aida', 'n_cells']].copy()
    mc = mc.rename(columns={'cell_type_aida': 'CellType_mc'})
    log(f'  metacell h5ad rows: {len(mc):,}')

    ct_map = {
        'B_cell': 'B', 'CD4_T_cell': 'CD4', 'CD8_T_cell': 'CD8',
        'NK_cell': 'NK', 'Monocyte': 'Monocyte', 'Other': 'unassigned',
    }
    cnt_ours = (meta_full
                .assign(CellType_mc=lambda x: x['CellType'].map(ct_map))
                .groupby(['Donor_id', 'Batch', 'CellType_mc'])
                .size().reset_index(name='n_cells_ours'))

    cmp = mc.merge(cnt_ours, on=['Donor_id', 'Batch', 'CellType_mc'], how='outer', indicator=True)
    log(f'  merged rows: {len(cmp):,}')
    log(f'  match status:\n{cmp["_merge"].value_counts()}')
    cmp_both = cmp[cmp['_merge'] == 'both'].copy()
    cmp_both['diff'] = cmp_both['n_cells_ours'].fillna(0) - cmp_both['n_cells'].fillna(0)
    log(f'  identical: {(cmp_both["diff"] == 0).sum()}/{len(cmp_both)}')
    log(f'  diff stats:\n{cmp_both["diff"].describe()}')
    if (cmp_both['diff'] != 0).any():
        log('  first 10 mismatches:')
        log(cmp_both[cmp_both['diff'] != 0].head(10).to_string())

    return cmp

def main():
    log(f'TS = {TS}')

    aida = build_aida()
    syn = build_synapse()

    full = pd.concat([aida, syn], ignore_index=True)
    log(f'\n=== TOTAL combined cells: {len(full):,} ===')
    log(f'  cohort: {full["cohort"].value_counts().to_dict()}')
    log(f'  CellType x cohort:\n{full.groupby(["cohort", "CellType"]).size().unstack().fillna(0)}')

    cmp = validate_synapse(syn)
    cmp_path = OUT_DIR / f'synapse_validation_{TS}.tsv'
    cmp.to_csv(cmp_path, sep='\t', index=False)
    log(f'  saved validation: {cmp_path}')

    out_path = OUT_DIR / f'barcode_metadata_{TS}.tsv'
    full.to_csv(out_path, sep='\t', index=False)
    log(f'\nSaved barcode metadata: {out_path}')

    latest = OUT_DIR / 'barcode_metadata_latest.tsv'
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_path.name)
    log(f'Symlink: {latest} -> {out_path.name}')

if __name__ == '__main__':
    main()


import os, sys, glob, pickle, gzip
from pathlib import Path
import pathlib
import pandas as pd
import numpy as np
from scipy.io import mmread
from scipy import sparse
from scipy.stats import spearmanr, rankdata
from statsmodels.stats.multitest import multipletests
from datetime import datetime
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

NWORKERS = int(os.environ.get('PHASE3B_WORKERS', 8))

TS = datetime.now().strftime('%Y%m%d_%H%M%S')
PROJ = Path('${HOME}/projects/integrated_aida_synapse')
OUT_DIR = PROJ / 'results' / 'phase3b_ourcd4'
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / f'step2_log_{TS}.txt'

SC_BASE = Path('${DATA_DIR}/integrated_aida_synapse_hg38_440/output/sc_count')
RUN_TAG = 'integrated_aida_synapse_hg38_440'
ANNOT_BED = Path(f'${DATA_DIR}/integrated_aida_synapse_hg38_440/output/annotate/{RUN_TAG}/bed/{RUN_TAG}.CRE.annot.bed.gz')
DIRECT_LOG = Path(f'${DATA_DIR}/integrated_aida_synapse_hg38_440/output/directionality/{RUN_TAG}/log/{RUN_TAG}.directionality.log.tsv.gz')

AIDA_TAGE = '${DATA_DIR}/lnc_project_AIDA_dataset/scafe_aggregate_recount_hg38/input/tage_all_combined_v1.csv'
SYN_TAGE_CT = '${PIPELINE_DIR}/results/06_tage/celltype/tage_all_combined_v3.csv'
SYN_TAGE_SC = '${PIPELINE_DIR}/results/06_tage/subcluster/tage_all_combined_v2.csv'

BARCODE_META = pathlib.Path('${SCRATCH}')

MIN_CELLS = 10
MIN_SAMPLES = 10
MIN_SAMPLES_PER_COHORT = 5
FDR_THRESH = 0.05
TARGET_VARS = ['age', 'tAge_EN', 'tAge_minus_age', 'yugene_diff_EN_tAge']

def log(msg):
    line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def load_tage():
    log('Loading AIDA tage...')
    a = pd.read_csv(AIDA_TAGE)
    a = a[(a['model_type'] == 'Chronoage') & (a['mode'] == 'EN')].copy()
    a = a.rename(columns={'Sample': 'Donor_id', 'CellType': 'CellType_raw'})
    a['Batch'] = a['Donor_id']
    a['cohort'] = 'AIDA'

    a['CellType'] = a['CellType_raw']
    a = a.rename(columns={'age': 'age', 'sex': 'Sex'})
    a['tAge_EN'] = a['age'] + a['scaled_diff_EN_tAge']
    a['tAge_minus_age'] = a['scaled_diff_EN_tAge']
    log(f'  AIDA Chronoage+EN: {len(a)} rows, {a["Donor_id"].nunique()} donors')

    log('Loading Synapse celltype tage...')
    s = pd.read_csv(SYN_TAGE_CT)
    s = s[(s['model_type'] == 'Chronoage') & (s['mode'] == 'EN')].copy()
    s = s.rename(columns={'CellType': 'CellType_raw', 'Age': 'age'})
    s['cohort'] = 'Synapse'

    ct_map = {'B': 'B_cell', 'CD4': 'CD4_T_cell', 'CD8': 'CD8_T_cell',
              'NK': 'NK_cell', 'Monocyte': 'Monocyte', 'unassigned': 'Other'}
    s['CellType'] = s['CellType_raw'].map(ct_map)
    s['tAge_EN'] = s['age'] + s['scaled_diff_EN_tAge']
    s['tAge_minus_age'] = s['scaled_diff_EN_tAge']
    log(f'  Synapse Chronoage+EN: {len(s)} rows, {s["Donor_id"].nunique()} donors')

    log('Synapse baseline filter (min Age Batch per donor)...')
    n_before = len(s)
    pair_age = s[['Donor_id', 'Batch', 'age']].drop_duplicates()
    keep = pair_age.sort_values(['Donor_id', 'age']).groupby('Donor_id').head(1)
    s = s.merge(keep[['Donor_id', 'Batch']], on=['Donor_id', 'Batch'])
    log(f'  before={n_before}, after baseline={len(s)} ({s["Donor_id"].nunique()} donors)')

    keep_cols = ['cohort', 'Donor_id', 'Batch', 'CellType',
                 'age', 'Sex', 'tAge_EN', 'tAge_minus_age', 'yugene_diff_EN_tAge', 'n_cells', 'Coverage']
    out = pd.concat([a[keep_cols], s[keep_cols]], ignore_index=True)
    log(f'Combined tAge metacell rows: {len(out)} ({out["Donor_id"].nunique()} donors)')
    return out

def load_tage_subcluster():
    log('Loading Synapse subcluster tage...')
    s = pd.read_csv(SYN_TAGE_SC)
    s = s[(s['model_type'] == 'Chronoage') & (s['mode'] == 'EN')].copy()
    s = s.rename(columns={'CellType': 'subcluster_raw', 'Age': 'age'})
    s['cohort'] = 'Synapse'
    s['CellType'] = s['subcluster_raw']
    s['tAge_EN'] = s['age'] + s['scaled_diff_EN_tAge']
    s['tAge_minus_age'] = s['scaled_diff_EN_tAge']

    pair_age = s[['Donor_id', 'Batch', 'age']].drop_duplicates()
    keep = pair_age.sort_values(['Donor_id', 'age']).groupby('Donor_id').head(1)
    s = s.merge(keep[['Donor_id', 'Batch']], on=['Donor_id', 'Batch'])
    log(f'  Synapse subcluster baseline: {len(s)} rows, {s["Donor_id"].nunique()} donors')

    aida_sc_path = '${DATA_DIR}/lnc_project_AIDA_dataset/by_subcluster/tage/tage_all_combined_v1.csv'
    log(f'AIDA subcluster tage: {aida_sc_path}')
    a = pd.read_csv(aida_sc_path)
    a = a[(a['model_type'] == 'Chronoage') & (a['mode'] == 'EN')].copy()
    a = a.rename(columns={'Sample': 'Donor_id', 'Subcluster': 'subcluster_raw'})
    a['Batch'] = a['Donor_id']
    a['cohort'] = 'AIDA'
    a['CellType'] = a['subcluster_raw']
    a = a.rename(columns={'sex': 'Sex'})
    a['tAge_EN'] = a['age'] + a['scaled_diff_EN_tAge']
    a['tAge_minus_age'] = a['scaled_diff_EN_tAge']
    log(f'  AIDA subcluster: {len(a)} rows, {a["Donor_id"].nunique()} donors')

    keep_cols = ['cohort', 'Donor_id', 'Batch', 'CellType',
                 'age', 'Sex', 'tAge_EN', 'tAge_minus_age', 'yugene_diff_EN_tAge', 'n_cells', 'Coverage']
    combined = pd.concat([a[keep_cols], s[keep_cols]], ignore_index=True)

    our = pd.read_csv('${SCRATCH}')
    log(f'CD4 自前 tAge 差し替え: {len(our)} メタセル ({our.Group.value_counts().to_dict()})')

    before = len(combined)
    combined = combined[~combined.CellType.isin(['CD4_Naive', 'CD4_Memory'])].copy()
    log(f'  既存 CD4 sub 行を除去: {before} -> {len(combined)}')

    o = pd.DataFrame({
        'cohort': our['cohort'].values,
        'Donor_id': our['Donor_id'].values,
        'Batch': our['Batch'].values,
        'CellType': our['Group'].values,
        'age': our['age'].values,
        'Sex': our['Sex'].values,
        'tAge_EN': np.nan, 'tAge_minus_age': np.nan,
        'yugene_diff_EN_tAge': our['yugene_diff_EN_tAge'].values,
        'n_cells': 100, 'Coverage': np.nan,
    })
    combined = pd.concat([combined, o[keep_cols]], ignore_index=True)
    log(f'  差し替え後: {len(combined)} 行, CD4 sub = {int(combined.CellType.isin(["CD4_Naive","CD4_Memory"]).sum())}')
    return combined

def _worker_aggregate(args):
    sample, bc2meta, n_cre, sc_base, cohort = args
    mtx_dir = Path(sc_base) / sample / sample / 'matrix'
    if not (mtx_dir / 'matrix.mtx').exists():
        return sample, None
    try:
        m = mmread(str(mtx_dir / 'matrix.mtx')).tocsc()
        bcs = pd.read_csv(mtx_dir / 'barcodes.tsv', header=None)[0].tolist()
    except Exception as e:
        return sample, None
    if m.shape[0] != n_cre:
        return sample, None

    groups = {}
    for col_idx, bc in enumerate(bcs):
        meta = bc2meta.get(bc)
        if meta is None: continue
        groups.setdefault(meta, []).append(col_idx)

    out = {}
    for meta_key, cols in groups.items():
        sub = m[:, cols]
        counts = np.asarray(sub.sum(axis=1)).flatten().astype(np.float64)
        umi = int(sub.sum())
        out[meta_key] = (counts, umi, len(cols))
    return sample, out

def aggregate_cohort(barcode_meta, cohort, level):
    cache_path = OUT_DIR / f'aggregation_cache_{cohort.lower()}_{level}.pkl'
    if cache_path.exists():
        log(f'Loading cache: {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    log(f'\n=== AGGREGATE {cohort} {level} ===')
    bm = barcode_meta[barcode_meta['cohort'] == cohort].copy()
    ct_col = 'CellType' if level == 'celltype' else 'subcluster'
    log(f'  cells: {len(bm):,}')

    sample_groups = {}
    for sample, gp in bm.groupby('sample'):
        d = {}
        for _, r in gp.iterrows():

            raw_bc = r['barcode_full'][len(sample) + 1:] if r['barcode_full'].startswith(sample + '_') else r['barcode_full']
            d[raw_bc] = (r['Donor_id'], r['Batch'], r[ct_col])
        sample_groups[sample] = d
    log(f'  samples in metadata: {len(sample_groups)}')

    samples = sorted(sample_groups.keys())

    first_sample = samples[0]
    genes_path = SC_BASE / first_sample / first_sample / 'matrix' / 'genes.tsv'
    cres = pd.read_csv(genes_path, header=None, sep='\t')[0].tolist()
    log(f'  CREs from {first_sample}: {len(cres):,}')

    n_cre = len(cres)

    expr_sum = {}
    umi_sum = {}
    n_cells_meta = {}

    log(f'  parallel workers: {NWORKERS}')
    args = [(sample, sample_groups[sample], n_cre, str(SC_BASE), cohort) for sample in samples]
    with Pool(NWORKERS) as p:
        for i, result in enumerate(p.imap_unordered(_worker_aggregate, args, chunksize=4)):
            sample, partials = result
            if partials is None:
                log(f'  MISS {sample}'); continue
            for key3, (counts, umi, ncells) in partials.items():

                key = (cohort, key3[0], key3[1], key3[2])
                if key not in expr_sum:
                    expr_sum[key] = counts
                    umi_sum[key] = umi
                    n_cells_meta[key] = ncells
                else:
                    expr_sum[key] = expr_sum[key] + counts
                    umi_sum[key] += umi
                    n_cells_meta[key] += ncells
            if (i + 1) % 50 == 0:
                log(f'  processed {i+1}/{len(samples)}, {len(expr_sum)} metacells so far')

    log(f'  final metacells: {len(expr_sum)}')

    out = {
        'cres': cres,
        'expr_sum': expr_sum,
        'umi_sum': umi_sum,
        'n_cells': n_cells_meta,
    }
    log(f'Saving cache: {cache_path}')
    with open(cache_path, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    return out

def build_expr_matrix(agg, mc_meta, level):
    cres = agg['cres']
    n_cre = len(cres)

    keys_with_tage = set(zip(mc_meta['cohort'], mc_meta['Donor_id'], mc_meta['Batch'], mc_meta['CellType']))
    keys_in_agg = set(agg['expr_sum'].keys())
    common = keys_with_tage & keys_in_agg
    log(f'  metacells in tage: {len(keys_with_tage)}, in agg: {len(keys_in_agg)}, common: {len(common)}')

    mc = mc_meta[
        mc_meta.apply(lambda r: (r['cohort'], r['Donor_id'], r['Batch'], r['CellType']) in common, axis=1)
        & (mc_meta['CellType'] != 'Other')
    ].copy().reset_index(drop=True)
    log(f'  metacells after Other-drop: {len(mc)}')

    mc['agg_n_cells'] = mc.apply(
        lambda r: agg['n_cells'].get((r['cohort'], r['Donor_id'], r['Batch'], r['CellType']), 0), axis=1)
    n_b = len(mc)
    mc = mc[mc['agg_n_cells'] >= MIN_CELLS].copy().reset_index(drop=True)
    log(f'  metacells with n_cells≥{MIN_CELLS}: {len(mc)} (dropped {n_b - len(mc)})')

    expr_cpm = np.zeros((n_cre, len(mc)), dtype=np.float32)
    for j, r in mc.iterrows():
        key = (r['cohort'], r['Donor_id'], r['Batch'], r['CellType'])
        cnt = agg['expr_sum'][key].astype(np.float64)
        umi = agg['umi_sum'][key]
        if umi > 0:
            expr_cpm[:, j] = (cnt / umi * 1e6).astype(np.float32)

    return expr_cpm, mc, cres

def spearman_per_celltype(expr_cpm, mc, target_var):
    from scipy.stats import t as tdist
    rows = []
    for ct, gp in mc.groupby('CellType'):
        idx = gp.index.values
        y = gp[target_var].values.astype(np.float64)
        ok = ~np.isnan(y)
        if ok.sum() < MIN_SAMPLES:
            continue
        idx_ok = idx[ok]
        y_ok = y[ok]
        y_ranked = rankdata(y_ok, method='average')
        n = len(y_ok)
        if n < MIN_SAMPLES:
            continue
        x_sub = expr_cpm[:, idx_ok].astype(np.float64)
        x_ranked = rankdata(x_sub, axis=1, method='average')
        x_mean = x_ranked.mean(axis=1, keepdims=True)
        x_std = x_ranked.std(axis=1, ddof=0)
        y_mean = y_ranked.mean()
        y_std = y_ranked.std(ddof=0)
        cov = ((x_ranked - x_mean) * (y_ranked - y_mean)).mean(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = np.where((x_std > 0) & (y_std > 0), cov / (x_std * y_std + 1e-12), np.nan)
            t = rho * np.sqrt((n - 2) / np.clip(1 - rho ** 2, 1e-12, None))
        p = 2 * tdist.sf(np.abs(t), n - 2)
        mean_expr = x_sub.mean(axis=1)
        rows.append(pd.DataFrame({
            'cell_type': ct, 'target_var': target_var,
            'cre_idx': np.arange(expr_cpm.shape[0]),
            'n_samples': n,
            'mean_expression': mean_expr.astype(np.float32),
            'spearman_rho': rho.astype(np.float32),
            'spearman_pval': p.astype(np.float32),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def partial_spearman_per_celltype(expr_cpm, mc, target_var, covar='cohort_dummy'):
    rows = []
    for ct, gp in mc.groupby('CellType'):
        idx = gp.index.values
        y = gp[target_var].values.astype(np.float64)
        cohort_d = gp['cohort_dummy'].values.astype(np.float64)
        ok = ~np.isnan(y) & ~np.isnan(cohort_d)
        if ok.sum() < MIN_SAMPLES:
            continue

        c0 = (cohort_d[ok] == 0).sum()
        c1 = (cohort_d[ok] == 1).sum()
        if c0 < MIN_SAMPLES_PER_COHORT or c1 < MIN_SAMPLES_PER_COHORT:
            continue
        idx_ok = idx[ok]
        y_ok = y[ok]
        cd_ok = cohort_d[ok]
        n = ok.sum()

        y_r = rankdata(y_ok)

        X = np.column_stack([np.ones(n), cd_ok])
        beta_y, *_ = np.linalg.lstsq(X, y_r, rcond=None)
        y_res = y_r - X @ beta_y

        x_sub = expr_cpm[:, idx_ok].astype(np.float64)
        n_cre = x_sub.shape[0]

        x_r = rankdata(x_sub, axis=1, method='average')

        XtX_inv = np.linalg.inv(X.T @ X)
        beta_x = (XtX_inv @ X.T @ x_r.T).T
        x_res = x_r - beta_x @ X.T

        x_mean = x_res.mean(axis=1, keepdims=True)
        y_mean_r = y_res.mean()
        x_std = x_res.std(axis=1, ddof=0)
        y_std_r = y_res.std(ddof=0)
        cov = ((x_res - x_mean) * (y_res - y_mean_r)).mean(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho_all = np.where((x_std > 0) & (y_std_r > 0),
                                cov / (x_std * y_std_r + 1e-12), np.nan)

        from scipy.stats import t as tdist
        df_t = n - 3
        with np.errstate(divide='ignore', invalid='ignore'):
            t = rho_all * np.sqrt(df_t / np.clip(1 - rho_all ** 2, 1e-12, None))
        p = 2 * tdist.sf(np.abs(t), df_t)
        mean_expr = x_sub.mean(axis=1)
        rows.append(pd.DataFrame({
            'cell_type': ct, 'target_var': target_var,
            'cre_idx': np.arange(n_cre),
            'n_samples': n,
            'n_aida': c0,
            'n_synapse': c1,
            'mean_expression': mean_expr.astype(np.float32),
            'spearman_rho': rho_all.astype(np.float32),
            'spearman_pval': p.astype(np.float32),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def apply_fdr(df):
    if len(df) == 0:
        return df
    df = df.copy()
    df['fdr'] = np.nan
    for (ct, tv), grp in df.groupby(['cell_type', 'target_var']):
        pvals = grp['spearman_pval'].values
        ok = ~np.isnan(pvals)
        fdr = np.full(len(pvals), np.nan)
        if ok.sum() > 0:
            _, fdr_ok, _, _ = multipletests(pvals[ok], method='fdr_bh')
            fdr[ok] = fdr_ok
        df.loc[grp.index, 'fdr'] = fdr
    return df

def load_directionality_classification():
    log('Loading CRE annot + directionality...')

    annot = pd.read_csv(ANNOT_BED, sep='\t', header=None, compression='gzip',
                        usecols=[3, 4],
                        names=['cre_id', 'classification_score'])

    df = pd.read_csv(ANNOT_BED, sep='\t', header=None, compression='gzip')
    log(f'  annot bed shape: {df.shape}')

    df['cre_id'] = df.iloc[:, 3]
    df['name_prefix'] = df['cre_id'].str.split(':').str[0]
    log(f'  cre_id name prefixes: {df["name_prefix"].value_counts().head(10).to_dict()}')

    log(f'  first 3 rows: {df.iloc[:3].to_dict()}')

    dlog = pd.read_csv(DIRECT_LOG, sep='\t', compression='gzip')
    log(f'  directionality log shape: {dlog.shape}, cols: {list(dlog.columns)[:10]}')
    return df, dlog

def run_level(level, mc_meta_full):
    log(f'\n{"="*60}\nLEVEL: {level}\n{"="*60}')

    bm = pd.read_csv(BARCODE_META, sep='\t')
    log(f'barcode metadata: {len(bm):,} cells')

    agg_a = aggregate_cohort(bm, 'AIDA', level)
    agg_s = aggregate_cohort(bm, 'Synapse', level)

    if level == 'celltype':
        mc_meta = mc_meta_full
    else:
        mc_meta = mc_meta_full

    mc_a = mc_meta[mc_meta['cohort'] == 'AIDA'].copy()
    mc_s = mc_meta[mc_meta['cohort'] == 'Synapse'].copy()

    log('\n--- AIDA expr matrix ---')
    expr_a, mc_a_filt, cres = build_expr_matrix(agg_a, mc_a, level)
    log(f'  AIDA: expr_cpm shape={expr_a.shape}, mc rows={len(mc_a_filt)}')

    log('\n--- Synapse expr matrix ---')

    expr_s, mc_s_filt, cres_s = build_expr_matrix(agg_s, mc_s, level)
    log(f'  Synapse: expr_cpm shape={expr_s.shape}, mc rows={len(mc_s_filt)}')
    assert cres == cres_s, 'CRE list mismatch between cohorts!'

    for cohort_name, expr, mcdf in [('aida', expr_a, mc_a_filt), ('synapse', expr_s, mc_s_filt)]:
        log(f'\n--- correlate {cohort_name} {level} ---')
        results = []
        for tv in TARGET_VARS:
            log(f'  {tv} ...')
            r = spearman_per_celltype(expr, mcdf, tv)
            results.append(r)
        df = pd.concat(results, ignore_index=True)
        df['cre_id'] = df['cre_idx'].apply(lambda i: cres[i])
        df = apply_fdr(df)

        out_full = OUT_DIR / f'cre_corr_{cohort_name}_{level}_full_{TS}.csv'
        out_sig = OUT_DIR / f'cre_corr_{cohort_name}_{level}_significant_{TS}.csv'
        cols = ['cre_id', 'cell_type', 'target_var', 'n_samples', 'mean_expression', 'spearman_rho', 'spearman_pval', 'fdr']
        df[cols].to_csv(out_full, index=False)
        df[df['fdr'] < FDR_THRESH][cols].to_csv(out_sig, index=False)
        log(f'  saved: {out_full.name}, {out_sig.name} ({(df["fdr"] < FDR_THRESH).sum()} sig)')

    log(f'\n--- correlate pooled {level} (cohort dummy) ---')
    expr_pool = np.concatenate([expr_a, expr_s], axis=1)
    mc_pool = pd.concat([mc_a_filt, mc_s_filt], ignore_index=True)
    mc_pool['cohort_dummy'] = (mc_pool['cohort'] == 'Synapse').astype(int)
    log(f'  pooled expr shape: {expr_pool.shape}, mc: {len(mc_pool)}')

    results = []
    for tv in TARGET_VARS:
        log(f'  {tv} ...')
        r = partial_spearman_per_celltype(expr_pool, mc_pool, tv, 'cohort_dummy')
        results.append(r)
    df = pd.concat(results, ignore_index=True)
    df['cre_id'] = df['cre_idx'].apply(lambda i: cres[i])
    df = apply_fdr(df)
    out_full = OUT_DIR / f'cre_corr_pooled_{level}_full_{TS}.csv'
    out_sig = OUT_DIR / f'cre_corr_pooled_{level}_significant_{TS}.csv'
    cols = ['cre_id', 'cell_type', 'target_var', 'n_samples', 'n_aida', 'n_synapse',
            'mean_expression', 'spearman_rho', 'spearman_pval', 'fdr']
    df[cols].to_csv(out_full, index=False)
    df[df['fdr'] < FDR_THRESH][cols].to_csv(out_sig, index=False)
    log(f'  saved: {out_full.name}, {out_sig.name} ({(df["fdr"] < FDR_THRESH).sum()} sig)')

def main():
    log(f'TS = {TS}')
    mc_meta_ct = load_tage()
    run_level('celltype', mc_meta_ct)

    log('\n========== SUBCLUSTER ==========')
    mc_meta_sc = load_tage_subcluster()
    run_level('subcluster', mc_meta_sc)

    log('\nALL DONE')

if __name__ == '__main__':
    main()


import pickle, os
import numpy as np, pandas as pd

B = '${PROJECT_DIR}/results'
D = f'{B}/PAPER_CRE_MECHANISM'
OUT = f'{D}/results/shared_assets'
os.makedirs(OUT, exist_ok=True)
TV = 'yugene_diff_EN_tAge'
rng = np.random.default_rng(0)
def w(s):
    print(s, flush=True)

import h5py
f = h5py.File('${DATA_DIR}/lnc_project_AIDA_dataset/all_cells_clustered/all_cells_clustered.h5ad', 'r')
um = f['obsm']['X_umap'][:]
obs = f['obs']
sub = pd.Categorical.from_codes(obs['subcluster']['codes'][:],
      [c.decode() for c in obs['subcluster']['categories'][:]])
samp = pd.Categorical.from_codes(obs['sample']['codes'][:],
      [c.decode() for c in obs['sample']['categories'][:]])
pd.DataFrame(dict(umap1=um[:, 0].astype(np.float32), umap2=um[:, 1].astype(np.float32),
                  Subcluster=sub.astype(str), Sample=samp.astype(str))
             ).to_parquet(f'{OUT}/umap_cells.parquet')
f.close()
w(f'[1] UMAP {um.shape[0]:,} 細胞 保存')

rob = pd.read_csv(f'{D}/results/fixes/cd4_donor_filter/robust_set_canon166.csv',
                  usecols=['cre_id', 'cell_type', 'program'])
rob = rob[rob.program.isin(['up_distal', 'down_proximal'])]
PROG = rob.drop_duplicates('cre_id').set_index('cre_id').program.to_dict()
jo = pd.read_csv(f'{D}/results/gene_to_cre/cre_gene_joined.csv',
                 usecols=['cre_id', 'cell_type', 'mean_expression', 'proximity', 'robust', 'near_gene'])
fg = rob.merge(jo[['cre_id', 'cell_type', 'mean_expression', 'proximity']],
               on=['cre_id', 'cell_type'], how='left').dropna(subset=['mean_expression'])
BG = {}
for (ct, prox), g in fg.groupby(['cell_type', 'proximity']):
    pool = jo[(jo.cell_type == ct) & (jo.proximity == prox) & (~jo.robust)
              & (~jo.cre_id.isin(set(rob.cre_id)))].sort_values('mean_expression')
    if not len(pool): continue
    pv = pool.mean_expression.values; ids = pool.cre_id.values
    used = set()
    for _, r in g.iterrows():
        i = int(np.searchsorted(pv, r.mean_expression))
        for off in range(400):
            hit = False
            for j in (i + off, i - off):
                if 0 <= j < len(ids) and ids[j] not in used:
                    used.add(ids[j]); BG[ids[j]] = r.program; hit = True; break
            if hit: break
info = pd.DataFrame(dict(cre_id=list(PROG) + list(BG),
                         role=['robust'] * len(PROG) + ['background'] * len(BG),
                         program=[PROG[c] for c in PROG] + [BG[c] for c in BG]))
gm = jo.drop_duplicates('cre_id').set_index('cre_id')
info['near_gene'] = info.cre_id.map(gm.near_gene)
info['proximity'] = info.cre_id.map(gm.proximity)
ctof = rob.drop_duplicates('cre_id').set_index('cre_id').cell_type
info['cell_type_hit'] = info.cre_id.map(ctof)
info.to_csv(f'{OUT}/tcre_info.csv', index=False)
w(f'対象 tCRE {len(info):,}（頑健 {len(PROG):,} / 背景 {len(BG):,}）')

ta = pd.read_csv(f'{D}/results/tage_v110_check/aida/'
                 'tage_all_combined_v5.csv')
ta = ta[(ta.model_type == 'Chronoage') & (ta['mode'] == 'EN')]
ta[['Sample', 'Subcluster', 'age', 'sex', 'country', TV]].rename(
    columns={TV: 'pace'}).to_csv(f'{OUT}/aida_tage.csv', index=False)

ts = pd.read_csv(f'{D}/results/tage_v110_check/'
                 'tage_all_combined_v4.csv')
ts = ts[(ts.model_type == 'Chronoage') & (ts['mode'] == 'EN')]
vis = ts[['Donor_id', 'Batch', 'Age']].drop_duplicates().sort_values(['Donor_id', 'Age'])
vis['visit'] = vis.groupby('Donor_id').cumcount() + 1
ts = ts.merge(vis[['Donor_id', 'Batch', 'visit']], on=['Donor_id', 'Batch'])
cols = ['Donor_id', 'Batch', 'CellType', 'Age', 'visit', TV]
if 'Sex' in ts.columns: cols.append('Sex')
ts[cols].rename(columns={TV: 'pace'}).to_csv(f'{OUT}/synapse_tage.csv', index=False)
w('tAge テーブル 2 本保存（tAge 1.1.0。0.1.0 原本は読まない）')

AGEK = ta.drop_duplicates('Sample').set_index('Sample').age.to_dict()
SAGE = {(r.Donor_id, r.Batch): r.Age for _, r in vis.iterrows()}
SVIS = {(r.Donor_id, r.Batch): r.visit for _, r in vis.iterrows()}
for coh, pkl in [('aida', f'{B}/phase3b_ourcd4/aggregation_cache_aida_subcluster.pkl'),
                 ('synapse', f'{B}/phase3b_ourcd4/aggregation_cache_synapse_subcluster.pkl')]:
    w(f'[{coh}] キャッシュを開く')
    d = pickle.load(open(pkl, 'rb'))
    cres = np.array(d['cres']); keys = list(d['expr_sum'].keys())
    idx = {c: i for i, c in enumerate(cres)}
    tgt = [c for c in info.cre_id if c in idx]
    tid = np.array([idx[c] for c in tgt])
    meta = pd.DataFrame(keys, columns=['cohort_k', 'Donor_id', 'Batch', 'CellType'])
    meta['n_cells'] = [d['n_cells'][k] for k in keys]
    if coh == 'aida':
        meta['age'] = meta.Donor_id.map(AGEK); meta['visit'] = 1
    else:
        meta['age'] = [SAGE.get((a, b), np.nan) for a, b in zip(meta.Donor_id, meta.Batch)]
        meta['visit'] = [SVIS.get((a, b), np.nan) for a, b in zip(meta.Donor_id, meta.Batch)]
    keep = (meta.n_cells >= 10) & meta.age.notna()
    kk = [k for k, ok in zip(keys, keep) if ok]
    meta = meta[keep].reset_index(drop=True)
    sub_k = kk[::max(1, len(kk) // 300)]
    det = np.zeros(len(cres))
    for k in sub_k:
        det += (d['expr_sum'][k] > 0)
    det /= len(sub_k)
    ref = np.nonzero(det >= 0.8)[0]
    if len(ref) > 20000:
        ref = rng.choice(ref, 20000, replace=False)
    M = np.zeros((len(kk), len(tgt)), np.float32)
    for i, k in enumerate(kk):
        v = d['expr_sum'][k]
        M[i] = np.log(v[tid] + 0.5) - np.log(v[ref] + 0.5).mean()
    del d
    pd.DataFrame(M, columns=tgt).to_parquet(f'{OUT}/{coh}_clr.parquet')
    meta.to_csv(f'{OUT}/{coh}_meta.csv', index=False)
    w(f'[{coh}] CLR {M.shape} 保存（行=メタセル {coh}_meta.csv と同順、列=tCRE）')
w('完了')


import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 11

TS = datetime.now().strftime('%Y%m%d_%H%M%S')
PROJ = Path('${HOME}/projects/integrated_aida_synapse')
OUT_DIR = PROJ / 'results' / 'phase6_novel_enhancer'
PLOTS = OUT_DIR / 'plots'
PLOTS.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / f'log_{TS}.txt'

PHASE3B_TS = '_v1'
PHASE4_TS = '_v2'

CRE_INFO = '${DATA_DIR}/integrated_aida_synapse_hg38_440/output/annotate/integrated_aida_synapse_hg38_440/log/integrated_aida_synapse_hg38_440.CRE.info.tsv.gz'
DIRECT_CSV = PROJ / 'results' / 'phase4_overlap' / f'directionality_classification_{PHASE4_TS}.csv'
POOLED_CT = PROJ / 'results' / 'phase3b_cre_corr' / f'cre_corr_pooled_celltype_full_{PHASE3B_TS}.csv'
REP_CSV = PROJ / 'results' / 'phase4_overlap' / f'replication_robust_{PHASE4_TS}.csv'

PRIMARY_TV = 'yugene_diff_EN_tAge'

def log(msg):
    line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def main():
    log(f'TS = {TS}')

    log(f'Loading CRE.info: {CRE_INFO}')
    info = pd.read_csv(CRE_INFO, sep='\t', compression='gzip',
                       usecols=['CREID','CREName','typeStr','proximity','class',
                                'geneNameStr','geneClassStr','gene_promoter','best_prob'])
    info = info.rename(columns={'CREID': 'cre_id'})
    log(f'  {len(info):,} CREs total')
    log(f'  proximity: {info["proximity"].value_counts().to_dict()}')
    log(f'  typeStr: {info["typeStr"].value_counts().to_dict()}')

    DIRECT_LOG_PATH = '${DATA_DIR}/integrated_aida_synapse_hg38_440/output/directionality/integrated_aida_synapse_hg38_440/log/integrated_aida_synapse_hg38_440.directionality.log.tsv.gz'
    log(f'Loading directionality LOG: {DIRECT_LOG_PATH}')
    dlog = pd.read_csv(DIRECT_LOG_PATH, sep='\t', compression='gzip',
                       usecols=['CREID','directionality','orientation','bias_strand','fwd_count','rev_count'])
    dlog = dlog.rename(columns={'CREID': 'cre_id'})
    log(f'  {len(dlog):,} CREs with directionality, range [{dlog["directionality"].min():.3f}, {dlog["directionality"].max():.3f}]')

    cre_full = info.merge(dlog, on='cre_id', how='left')
    log(f'merged: {len(cre_full):,}, with directionality non-null: {cre_full["directionality"].notna().sum():,}')

    def classify(row):
        prox = row['proximity']
        d = row['directionality']
        if prox == 'distal':
            if pd.isna(d): return 'distal_no_score'
            if d >= 0.5: return 'divergent_enhancer'
            if d >= 0: return 'unidirectional_enhancer'
            return 'convergent_other_distal'
        if prox == 'proximal':
            if pd.isna(d): return 'proximal_no_score'
            if d >= 0.5: return 'divergent_promoter'
            if d >= 0: return 'unidirectional_promoter'
            return 'convergent_other_proximal'
        return 'other'
    cre_full['enhancer_class'] = cre_full.apply(classify, axis=1)
    log(f'\n6-class enhancer classification (FIXED):')
    log(cre_full['enhancer_class'].value_counts().to_string())

    cre_full['is_novel_enh'] = (
        (cre_full['proximity'] == 'distal') &
        (cre_full['typeStr'] == 'unanno_tss') &
        (cre_full['gene_promoter'] == 'none') &
        (cre_full['enhancer_class'].isin(['divergent_enhancer', 'unidirectional_enhancer']))
    )
    cre_full['is_known_enh'] = (
        (cre_full['proximity'] == 'distal') &
        (cre_full['typeStr'] != 'unanno_tss') &
        (cre_full['enhancer_class'].isin(['divergent_enhancer', 'unidirectional_enhancer']))
    )
    cre_full['cre_class_summary'] = np.select(
        [cre_full['is_novel_enh'], cre_full['is_known_enh']],
        ['novel_enhancer', 'known_enhancer'],
        default='other_or_promoter'
    )
    log(f'\nCRE class summary:')
    log(cre_full['cre_class_summary'].value_counts().to_string())
    log(f'\nNovel enhancer breakdown:')
    log(cre_full[cre_full['is_novel_enh']]['enhancer_class'].value_counts().to_string())

    cre_full[['cre_id','typeStr','proximity','class','gene_promoter','enhancer_class','directionality',
              'cre_class_summary','geneNameStr','geneClassStr']].to_csv(
        OUT_DIR / f'novel_enhancer_classification_{TS}.csv', index=False)
    log(f'Saved classification: novel_enhancer_classification_{TS}.csv')

    log('\nLoading pooled sig...')
    pooled = pd.read_csv(POOLED_CT, usecols=['cre_id','cell_type','target_var','spearman_rho','fdr'])
    pooled_sig = pooled[pooled['fdr'] < 0.05]
    log(f'  pooled sig: {len(pooled_sig):,}')

    novel_pool = pooled_sig.merge(
        cre_full[cre_full['is_novel_enh']][['cre_id','enhancer_class','geneNameStr','class']],
        on='cre_id'
    )
    log(f'novel × pooled sig: {len(novel_pool):,}')
    novel_pool.to_csv(OUT_DIR / f'novel_enhancer_aging_pooled_{TS}.csv', index=False)

    log(f'\nNovel enhancer aging-associated (pooled sig) breakdown:')
    bt = novel_pool.groupby(['cell_type','target_var']).size().unstack(fill_value=0)
    log(bt.to_string())

    log('\nLoading replication set...')
    rep = pd.read_csv(REP_CSV)
    rep_ct = rep[rep['level'] == 'celltype']
    novel_rep = rep_ct.merge(
        cre_full[cre_full['is_novel_enh']][['cre_id','enhancer_class','geneNameStr','class']],
        on='cre_id'
    )
    log(f'novel × replication: {len(novel_rep):,}')
    novel_rep.to_csv(OUT_DIR / f'novel_enhancer_aging_replication_{TS}.csv', index=False)

    log(f'\nNovel enhancer aging-associated (replication) breakdown:')
    bt2 = novel_rep.groupby(['cell_type','target_var']).size().unstack(fill_value=0)
    log(bt2.to_string())

    if 'rho_aida' in novel_rep.columns and 'rho_synapse' in novel_rep.columns:
        novel_rep['rho_mean'] = (novel_rep['rho_aida'] + novel_rep['rho_synapse']) / 2
        top = novel_rep[(novel_rep['target_var'] == PRIMARY_TV) & (novel_rep['rho_mean'].abs() >= 0.25)].copy()
        top = top.sort_values('rho_mean', key=lambda x: x.abs(), ascending=False)
        cols_keep = ['cre_id','cell_type','target_var','rho_aida','rho_synapse','rho_mean',
                     'fdr_aida','fdr_synapse','geneNameStr','class']

        if 'enhancer_class' in top.columns:
            cols_keep.append('enhancer_class')
        top[cols_keep].to_csv(OUT_DIR / f'novel_enhancer_top_{TS}.csv', index=False)
        log(f'\nTop novel enhancers (replication, |ρ_mean|≥0.25, primary tv): {len(top):,}')
        log('Top 10:')
        log(top.head(10)[cols_keep].to_string())

    log('\n=== Plots ===')

    log('Plot 1: novel vs known breakdown (overall counts)')
    cnt = cre_full['cre_class_summary'].value_counts().reindex(
        ['novel_enhancer','known_enhancer','other_or_promoter']).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ['#ff7f0e', '#2ca02c', '#cccccc']
    ax.barh(cnt.index, cnt.values, color=colors, edgecolor='black')
    for i, v in enumerate(cnt.values):
        ax.text(v + cnt.max()*0.01, i, f'{int(v):,}', va='center')
    ax.set_xlabel('# CREs')
    ax.set_title(f'CRE class breakdown (total {len(cre_full):,} integrated tCREs)')
    plt.tight_layout()
    plt.savefig(PLOTS / f'novel_vs_known_breakdown_{TS}.pdf')
    plt.savefig(PLOTS / f'novel_vs_known_breakdown_{TS}.png', dpi=300)
    plt.close()

    log('Plot 2: aging-association rate by class')
    classes = ['novel_enhancer','known_enhancer','other_or_promoter']
    rates = []
    sig_set = set(pooled_sig['cre_id'])
    for c in classes:
        sub = cre_full[cre_full['cre_class_summary'] == c]
        n_total = len(sub)
        n_sig = len(set(sub['cre_id']) & sig_set)
        rates.append({'class': c, 'n_total': n_total, 'n_pooled_sig_any': n_sig,
                      'pct_sig_any': 100*n_sig/max(1,n_total)})
    rates_df = pd.DataFrame(rates)
    log(rates_df.to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rates_df['class'], rates_df['pct_sig_any'], color=['#ff7f0e','#2ca02c','#cccccc'], edgecolor='black')
    for i, r in enumerate(rates_df.itertuples()):
        ax.text(i, r.pct_sig_any + 0.5, f'{r.pct_sig_any:.1f}%\n({r.n_pooled_sig_any:,}/{r.n_total:,})', ha='center', fontsize=9)
    ax.set_ylabel('% CREs aging-associated (pooled FDR<0.05 in any cell_type × target_var)')
    ax.set_title('Aging-association enrichment per CRE class')
    plt.tight_layout()
    plt.savefig(PLOTS / f'aging_assoc_rate_by_class_{TS}.pdf')
    plt.savefig(PLOTS / f'aging_assoc_rate_by_class_{TS}.png', dpi=300)
    plt.close()
    rates_df.to_csv(OUT_DIR / f'aging_assoc_rate_by_class_{TS}.csv', index=False)

    log('Plot 3: novel enhancer volcano (pooled, primary tv)')
    novel_pool_p = novel_pool[novel_pool['target_var'] == PRIMARY_TV].copy()
    novel_pool_p['neg_log10_fdr'] = -np.log10(novel_pool_p['fdr'].clip(lower=1e-300))
    cells = sorted(novel_pool_p['cell_type'].unique())
    cells = [c for c in cells if c != 'Other']
    fig, axs = plt.subplots(1, len(cells), figsize=(4*len(cells), 4), sharey=True, sharex=True)
    for ax, ct in zip(axs, cells):
        sub = novel_pool_p[novel_pool_p['cell_type'] == ct]
        if len(sub) == 0: continue
        col = sub['enhancer_class'].map({'divergent_enhancer':'#d62728','unidirectional_enhancer':'#1f77b4'}).fillna('gray')
        ax.scatter(sub['spearman_rho'], sub['neg_log10_fdr'], s=2, alpha=0.4, c=col, rasterized=True)

        top = sub.assign(absrho=sub['spearman_rho'].abs()).nlargest(5, 'absrho')
        for _, r in top.iterrows():
            label = (str(r.geneNameStr).split(';')[0] if pd.notna(r.geneNameStr) else r.cre_id[:20])
            ax.text(r.spearman_rho, r.neg_log10_fdr, label, fontsize=7)
        ax.scatter(top['spearman_rho'], top['neg_log10_fdr'], s=40, c='black', edgecolor='red', linewidth=1, zorder=3)
        ax.axvline(0, color='black', lw=0.4)
        ax.set_title(f'{ct}\nn={len(sub):,}', fontsize=10)
        ax.set_xlabel('Spearman ρ')
        if ax is axs[0]:
            ax.set_ylabel('-log10(FDR)')
    plt.suptitle(f'Novel enhancer volcano — {PRIMARY_TV} (pooled sig)\nred=divergent, blue=unidirectional')
    plt.tight_layout()
    plt.savefig(PLOTS / f'novel_enhancer_volcano_{TS}.pdf')
    plt.savefig(PLOTS / f'novel_enhancer_volcano_{TS}.png', dpi=200)
    plt.close()

    log('Plot 4: top 30 novel enhancer celltype heatmap')
    if len(novel_pool_p) > 0:

        agg = novel_pool_p.groupby('cre_id').agg(max_abs_rho=('spearman_rho', lambda x: x.abs().max()),
                                                  n_cell=('cell_type','nunique')).reset_index()
        top30 = agg.nlargest(30, 'max_abs_rho')['cre_id'].tolist()
        sub = novel_pool_p[novel_pool_p['cre_id'].isin(top30)]
        pivot = sub.pivot_table(index='cre_id', columns='cell_type', values='spearman_rho', aggfunc='mean')

        gene_map = sub.groupby('cre_id')['geneNameStr'].first().to_dict()
        pivot.index = [f"{cid}\n[{str(gene_map.get(cid,'')).split(';')[0]}]" for cid in pivot.index]
        fig, ax = plt.subplots(figsize=(8, max(6, 0.4*len(pivot))))
        sns.heatmap(pivot, cmap='RdBu_r', center=0, ax=ax, annot=False, cbar_kws={'label':'Spearman ρ'})
        ax.set_title(f'Top 30 novel enhancers — {PRIMARY_TV}')
        plt.tight_layout()
        plt.savefig(PLOTS / f'novel_enhancer_celltype_heatmap_{TS}.pdf')
        plt.savefig(PLOTS / f'novel_enhancer_celltype_heatmap_{TS}.png', dpi=300)
        plt.close()

    log('\nALL DONE')
    log(f'Outputs in: {OUT_DIR}')

if __name__ == '__main__':
    main()


import pickle, gzip, re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd

PROJ = Path('${HOME}/projects/integrated_aida_synapse')
R = PROJ / 'results'
OUT = R / 'intragenic_initiation'
GTF = Path('${DATA_DIR}/adachi/reference/GRCh38.113.gtf.gz')
OUT.mkdir(parents=True, exist_ok=True)
TS = datetime.now().strftime('%Y%m%d_%H%M%S')
PROM_WIN = 500
L = []

def w(s=''):
    print(s, flush=True); L.append(s)

def log(*a):
    print(f'[{datetime.now().strftime("%H:%M:%S")}]', *a, flush=True)

log('CRE 座標の読み込み...')
cache = pickle.load(open(R / 'phase3b_cre_corr/aggregation_cache_aida_celltype.pkl', 'rb'))
CRES = cache['cres']
p = pd.Series(CRES).str.rsplit('_', n=3, expand=True)
C = pd.DataFrame({'cre_id': CRES, 'chrom': p[0].values,
                  'start': pd.to_numeric(p[1]).values, 'end': pd.to_numeric(p[2]).values,
                  'strand': p[3].values})
C['mid'] = (C.start + C.end) // 2
w('=' * 96)
w('遺伝子内部からの転写開始 — 指標の構築')
w('=' * 96)
w(f'  tCRE {len(C):,}')
w(f'  strand: ' + ' / '.join(f'{k} {v:,}' for k, v in C.strand.value_counts().items()))

log('GTF 読み込み (gene + transcript)...')
genes, tss = [], []
with gzip.open(GTF, 'rt') as fh:
    for line in fh:
        if line[0] == '#':
            continue
        f = line.split('\t', 9)
        if f[2] == 'gene':
            m = re.search(r'gene_name "([^"]+)"', f[8]); b = re.search(r'gene_biotype "([^"]+)"', f[8])
            genes.append(('chr' + f[0], int(f[3]), int(f[4]), f[6],
                          m.group(1) if m else 'NA', b.group(1) if b else 'NA'))
        elif f[2] == 'transcript':
            m = re.search(r'gene_name "([^"]+)"', f[8])
            t = int(f[3]) if f[6] == '+' else int(f[4])
            tss.append(('chr' + f[0], t, f[6], m.group(1) if m else 'NA'))
G = pd.DataFrame(genes, columns=['chrom', 'gstart', 'gend', 'gstrand', 'gene', 'biotype'])
T = pd.DataFrame(tss, columns=['chrom', 'tss', 'tstrand', 'gene']).drop_duplicates()
w(f'  GTF: 遺伝子 {len(G):,} / 転写産物 TSS {len(T):,} (ユニーク座標)')

log('tCRE の分類...')

prom = set()
for ch, gt in T.groupby('chrom'):
    sub = C[C.chrom == ch]
    if not len(sub):
        continue
    tv = np.sort(gt.tss.values)
    idx = np.searchsorted(tv, sub.mid.values)
    for i, (cid, m) in enumerate(zip(sub.cre_id.values, sub.mid.values)):
        j = idx[i]
        for k in (j - 1, j):
            if 0 <= k < len(tv) and abs(tv[k] - m) <= PROM_WIN:
                prom.add(cid); break
C['is_prom'] = C.cre_id.isin(prom)
w(f'  promoter (TSS +-{PROM_WIN} bp): {C.is_prom.sum():,}')

Gm = G[G.biotype.isin(['protein_coding', 'lncRNA'])].copy()
w(f'  遺伝子体の定義に使う遺伝子: {len(Gm):,} (protein_coding + lncRNA)')
body_gene, body_strand = {}, {}
for ch, gg in Gm.groupby('chrom'):
    sub = C[C.chrom == ch]
    if not len(sub):
        continue
    gs = gg.gstart.values; ge = gg.gend.values
    order = np.argsort(gs); gs, ge = gs[order], ge[order]
    gn = gg.gene.values[order]; gst = gg.gstrand.values[order]

    ins = np.searchsorted(gs, sub.mid.values, side='right')
    for i, (cid, m) in enumerate(zip(sub.cre_id.values, sub.mid.values)):
        best = None
        for k in range(ins[i] - 1, max(-1, ins[i] - 400), -1):
            if k < 0:
                break
            if ge[k] >= m:

                if best is None or (ge[k] - gs[k]) > (ge[best] - gs[best]):
                    best = k
        if best is not None:
            body_gene[cid] = gn[best]; body_strand[cid] = gst[best]
C['body_gene'] = C.cre_id.map(body_gene)
C['body_strand'] = C.cre_id.map(body_strand)
w(f'  遺伝子体内にある tCRE: {C.body_gene.notna().sum():,}')

def classify(r):
    if r.is_prom:
        return 'promoter'
    if pd.isna(r.body_gene):
        return 'intergenic'
    if r.strand == '.':
        return 'intragenic_unstranded'
    return 'intragenic_sense' if r.strand == r.body_strand else 'intragenic_anti'

C['klass'] = C.apply(classify, axis=1)
w('')
w(f'  {"分類":26s} {"tCRE":>9s} {"割合":>7s}')
for k, v in C.klass.value_counts().items():
    w(f'  {k:26s} {v:>9,} {100*v/len(C):>6.1f}%')

log('メタセル指標の計算...')
KL = ['promoter', 'intragenic_sense', 'intragenic_anti', 'intragenic_unstranded', 'intergenic']
masks = {k: (C.klass == k).values for k in KL}
rows = []
for coh in ['aida', 'synapse']:
    cc = pickle.load(open(R / f'phase3b_cre_corr/aggregation_cache_{coh}_celltype.pkl', 'rb'))
    assert cc['cres'] == CRES, f'{coh}: CRE 順序が一致しない'
    for key, arr in cc['expr_sum'].items():
        tot = arr.sum()
        if tot <= 0:
            continue
        d = dict(cohort=key[0], donor=key[1], batch=key[2], cell_type=key[3],
                 umi_sum=cc['umi_sum'][key], n_cells=cc['n_cells'][key], total_cre=tot)
        for k in KL:
            d[k] = float(arr[masks[k]].sum())
            d[k + '_frac'] = d[k] / tot
            d['n_' + k] = int((arr[masks[k]] > 0).sum())
        rows.append(d)
    log(f'  {coh}: 累計 {len(rows)} メタセル')
M = pd.DataFrame(rows)
M['II'] = np.log2((M.intragenic_sense + 1) / (M.promoter + 1))
M['II_anti'] = np.log2((M.intragenic_anti + 1) / (M.promoter + 1))
M['detect_ratio'] = M.n_intragenic_sense / M.n_promoter.replace(0, np.nan)
w('')
w(f'  メタセル {len(M):,} (AIDA {int((M.cohort=="AIDA").sum()):,} / '
  f'Synapse {int((M.cohort=="Synapse").sum()):,})')
w(f'  II 中央 {M.II.median():.3f} (25% {M.II.quantile(.25):.3f} / 75% {M.II.quantile(.75):.3f})')
w('')
w(f'  {"細胞種":12s} {"n":>5s} {"II中央":>8s} {"prom割合":>9s} {"内部sense":>10s} {"内部anti":>9s}')
for ct, g in M.groupby('cell_type'):
    w(f'  {ct:12s} {len(g):>5,} {g.II.median():>8.3f} {g.promoter_frac.median():>8.3f} '
      f'{g.intragenic_sense_frac.median():>10.3f} {g.intragenic_anti_frac.median():>9.3f}')

log('遺伝子ごとの内訳...')
gb = C[C.body_gene.notna()].groupby('body_gene').agg(
    n_cre=('cre_id', 'size'),
    n_prom=('is_prom', 'sum'),
    n_intra_sense=('klass', lambda v: int((v == 'intragenic_sense').sum())),
    n_intra_anti=('klass', lambda v: int((v == 'intragenic_anti').sum()))).reset_index()
gb = gb.rename(columns={'body_gene': 'gene'}).merge(
    Gm[['gene', 'gstart', 'gend', 'biotype']].drop_duplicates('gene'), on='gene', how='left')
gb['glen'] = gb.gend - gb.gstart
w(f'  tCRE を持つ遺伝子 {len(gb):,}')
w(f'  遺伝子体内 sense tCRE を 5 個以上持つ遺伝子: {int((gb.n_intra_sense>=5).sum()):,}')
w(f'  同 10 個以上: {int((gb.n_intra_sense>=10).sum()):,}')

C.to_csv(OUT / f'cre_classification_{TS}.csv.gz', index=False, compression='gzip')
M.to_csv(OUT / f'metacell_index_{TS}.csv', index=False)
gb.to_csv(OUT / f'gene_breakdown_{TS}.csv', index=False)
(OUT / f'build_report_{TS}.txt').write_text('\n'.join(L))
log(f'完了 → {OUT}')
