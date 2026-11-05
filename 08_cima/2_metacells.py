#!/usr/bin/env python3
import os, glob, time
import h5py, numpy as np, pandas as pd
from scipy.sparse import csr_matrix, coo_matrix

D = '${PAPER_DIR}'
W = '${DATA_DIR}/cima'
OUT = f'{D}/results/cima_tage'
os.makedirs(OUT, exist_ok=True)
CHUNK = 100_000
MINCELL = 10
L = []
def w(s=''):
    print(s, flush=True); L.append(s)

MAP = dict(pd.read_csv(f'{W}/meta/celltype_map.csv').values)
files = sorted(glob.glob(f'{W}/rna/CIMA_RNA_*.h5ad'))
w(f'RNA ファイル {len(files)} 本')

genes = None
for f in files:
    with h5py.File(f, 'r') as h:
        g = np.array(h['var/_index']).astype(str)
    if genes is None: genes = g
    elif not np.array_equal(genes, g): raise SystemExit(f'遺伝子順が違う: {f}')
w(f'遺伝子 {len(genes):,}（全ファイルで一致）')

acc, meta, unmapped = {}, {}, {}
t0 = time.time()
for f in files:
    lin = os.path.basename(f).split('_')[2]
    with h5py.File(f, 'r') as h:
        def cat(n):
            gg = h[f'obs/{n}']
            if isinstance(gg, h5py.Group):
                return np.array(gg['categories']).astype(str)[np.array(gg['codes'])]
            return np.array(gg)
        smp = cat('sample'); l4 = cat('cell_type_l4')
        age = cat('age'); sex = cat('sex')
        n = len(smp)
        sub = np.array([MAP.get(x, '') for x in l4])
        for x in set(l4[sub == '']):
            unmapped[x] = unmapped.get(x, 0) + int((l4 == x).sum())
        keys = np.array([f'{a}||{b}' if b else '' for a, b in zip(smp, sub)])
        w(f'\n{lin}: {n:,} 細胞 / 対応 {int((sub!="").sum()):,} ({(sub!="").mean()*100:.1f}%)')
        for k, a, s in zip(keys, age, sex):
            if k and k not in meta: meta[k] = (int(a), str(s))
        X = h['layers/counts']; ip = np.array(X['indptr'])
        dX = X['data']; iX = X['indices']
        for c0 in range(0, n, CHUNK):
            c1 = min(c0 + CHUNK, n)
            kk = keys[c0:c1]; ok = kk != ''
            if not ok.any(): continue
            p0, p1 = int(ip[c0]), int(ip[c1])
            Xc = csr_matrix((np.asarray(dX[p0:p1], np.float32),
                             np.asarray(iX[p0:p1], np.int64),
                             ip[c0:c1 + 1] - p0), shape=(c1 - c0, len(genes)))
            uk = sorted(set(kk[ok]))
            ki = {u: i for i, u in enumerate(uk)}
            rows = np.array([ki[x] for x in kk[ok]])
            G = coo_matrix((np.ones(len(rows), np.float32), (rows, np.nonzero(ok)[0])),
                           shape=(len(uk), c1 - c0)).tocsr()
            S = np.asarray((G @ Xc).todense(), np.float64)
            cnt = np.bincount(rows, minlength=len(uk))
            for i, u in enumerate(uk):
                if u in acc: acc[u][0] += S[i]; acc[u][1] += cnt[i]
                else: acc[u] = [S[i], int(cnt[i])]
            if (c0 // CHUNK) % 5 == 0:
                w(f'    {c1:,}/{n:,}  経過 {(time.time()-t0)/60:.1f} 分')

if unmapped:
    w('\n対応表に無い RNA L4（細胞数順）:')
    for k, v in sorted(unmapped.items(), key=lambda x: -x[1])[:25]: w(f'  {k:32s} {v:8d}')

keys = sorted(k for k in acc if acc[k][1] >= MINCELL)
w(f'\nメタセル {len(keys):,}（{MINCELL} 細胞未満を除いた。除外 {len(acc)-len(keys)}）')
E = np.vstack([acc[k][0] for k in keys]).T
ids = [k.replace('||', '_') for k in keys]
md = pd.DataFrame({
    'ID': ids,
    'Sample': [k.split('||')[0] for k in keys],
    'Subcluster': [k.split('||')[1] for k in keys],
    'n_cells': [acc[k][1] for k in keys],
    'Coverage': E.sum(0).astype(np.int64),
    'age': [meta[k][0] for k in keys],
    'sex': [meta[k][1].lower() for k in keys],
    'Treatment': 'all',
})
md['CellType'] = md.Subcluster
md.set_index('ID').to_csv(f'{OUT}/Metadata_metacells.csv')
pd.DataFrame(E, index=genes, columns=ids).to_csv(f'{OUT}/Exprs_metacells.csv')
w(f'  被覆 中央 {int(np.median(md.Coverage)):,} UMI（AIDA は 1,084,601）')
w(f'  細胞数 中央 {int(md.n_cells.median())}（AIDA は 165）')
w(f'  ドナー {md.Sample.nunique()} / subset {md.Subcluster.nunique()}')
w('  subset ごと:'); 
for s, g in md.groupby('Subcluster'):
    w(f'    {s:14s} n={len(g):4d}  被覆中央 {int(g.Coverage.median()):9,}  細胞中央 {int(g.n_cells.median()):5d}')
open(f'{OUT}/report_metacells.txt', 'w').write('\n'.join(L))
print(f'-> {OUT}')
