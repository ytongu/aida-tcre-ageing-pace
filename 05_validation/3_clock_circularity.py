#!/usr/bin/env python3
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import joblib
import warnings
warnings.filterwarnings("ignore")

BASE = "${PAPER_DIR}"
MODEL = "${DATA_DIR}/tage_models/EN_Chronoage_Multispecies_Multitissue_yugenediff.pkl"
ORTH  = "${CONDA_ENVS}/tage/lib/R/library/tAge/extdata/metadata/Table_of_orthologs.csv"
HUMAN = "${CONDA_ENVS}/tage/lib/R/library/tAge/extdata/metadata/Gene_table_human.csv"
UPD   = f"{BASE}/results/target_side_dichotomy/enh_up_distal_target_genes.csv"
DNP   = f"{BASE}/results/target_side_dichotomy/prom_down_proximal_genes.csv"
OUT   = f"{BASE}/results/tage_circularity/a1_gene_overlap"
os.makedirs(OUT, exist_ok=True)

log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); log_lines.append(s)

log("="*78)
log("tAge 循環監査 a1: 時計遺伝子 × 2プログラム遺伝子 の重なり")
log("="*78)
clock = joblib.load(MODEL)
feat_in = np.asarray(clock.feature_names_in_).astype(str)
log(f"[clock] Pipeline steps: {[ (n,type(s).__name__) for n,s in clock.steps]}")
log(f"[clock] 入力特徴(feature_names_in_) n = {len(feat_in)} (マウスEntrez)")

support = None
coef = None
for name, step in clock.steps:
    if hasattr(step, "get_support"):
        support = step.get_support()
    if hasattr(step, "coef_"):
        coef = np.ravel(step.coef_)
if support is None:
    support = np.ones(len(feat_in), dtype=bool)
selected_idx = np.where(support)[0]
nz_local = np.where(coef != 0)[0]
nz_global_idx = selected_idx[nz_local]
clock_nonzero_mouse = set(feat_in[nz_global_idx])
clock_input_mouse   = set(feat_in)
log(f"[clock] SelectKBest 通過 k = {support.sum()} / {len(feat_in)}")
log(f"[clock] ElasticNet 非ゼロ係数 = {len(clock_nonzero_mouse)} 遺伝子(=実際に予測へ寄与)")

orth = pd.read_csv(ORTH)
hum  = pd.read_csv(HUMAN)
orth["Entrez.Mouse"] = orth["Entrez.Mouse"].astype("Int64").astype(str)
orth["Entrez.Human"] = orth["Entrez.Human"].astype("Int64").astype(str)
hum["Entrez"] = hum["Entrez"].astype("Int64").astype(str)
m2h = dict(zip(orth["Entrez.Mouse"], orth["Entrez.Human"]))
h2sym = dict(zip(hum["Entrez"], hum["Gene.Symbol"]))

def mouse_to_symbols(mouse_set):
    syms, mapped = set(), 0
    for me in mouse_set:
        he = m2h.get(me)
        if he is None: continue
        sym = h2sym.get(he)
        if sym is None: continue
        syms.add(sym); mapped += 1
    return syms, mapped

clock_input_syms,  n_in_mapped  = mouse_to_symbols(clock_input_mouse)
clock_nonzero_syms, n_nz_mapped = mouse_to_symbols(clock_nonzero_mouse)
log(f"[map] 時計入力 10,487 → ヒトSymbol: {len(clock_input_syms)} unique (写像成功 {n_in_mapped})")
log(f"[map] 時計非ゼロ {len(clock_nonzero_mouse)} → ヒトSymbol: {len(clock_nonzero_syms)} unique (写像成功 {n_nz_mapped})")

pd.DataFrame(sorted(clock_nonzero_syms), columns=["clock_nonzero_gene_symbol"]).to_csv(
    f"{OUT}/clock_gene_map.csv", index=False)

up   = pd.read_csv(UPD)["gene"].dropna().astype(str)
up   = set(up[up != ""])
down = pd.read_csv(DNP)["gene"].dropna().astype(str)
down = set(down[down != ""])
log(f"[prog] up-distal   target genes  n = {len(up)}")
log(f"[prog] down-proximal nearest genes n = {len(down)}")

universe = clock_input_syms
K = len(clock_nonzero_syms & universe)
N = len(universe)

def audit(name, prog):
    prog_elig = prog & universe
    hit  = len(prog & clock_nonzero_syms)
    n    = len(prog_elig)
    frac_of_prog_in_input = len(prog_elig)/max(len(prog),1)
    frac_nonzero = hit/max(n,1)

    a = hit; b = n - hit; c = K - hit; d = N - n - c
    OR, p = fisher_exact([[a, b],[c, d]], alternative="greater")
    base_rate = K/N
    log("-"*78)
    log(f"[{name}]")
    log(f"  プログラム遺伝子(全) n = {len(prog)}, うち時計入力に載る(eligible) = {n} ({frac_of_prog_in_input:.1%})")
    log(f"  eligible のうち時計 非ゼロ = {hit} ({frac_nonzero:.1%})  vs 背景基準率 {base_rate:.1%}")
    log(f"  Fisher enrichment OR = {OR:.2f}, p(greater) = {p:.2e}")
    return dict(program=name, n_all=len(prog), n_eligible=n,
                frac_in_input=round(frac_of_prog_in_input,4),
                hit_nonzero=hit, frac_nonzero=round(frac_nonzero,4),
                base_rate=round(base_rate,4), fisher_OR=round(float(OR),3),
                fisher_p_greater=float(p))

log("")
log(f"背景ユニバース = 時計入力 eligible 遺伝子 N = {N}, うち非ゼロ K = {K} (基準率 {K/N:.1%})")
rows = [audit("up_distal_enhancer", up), audit("down_proximal_promoter", down)]

log("-"*78)
fu = rows[0]["frac_nonzero"]; fd = rows[1]["frac_nonzero"]
log(f"[非対称] 時計非ゼロ率  down-proximal {fd:.1%}  vs  up-distal {fu:.1%}  "
    f"(比 {fd/max(fu,1e-9):.2f}x)")
log(f"[解釈] 予測どおり down>up なら: 近位=循環注記付き / 遠位=循環頑健コア。")

pd.DataFrame(rows).to_csv(f"{OUT}/overlap_table.csv", index=False)
with open(f"{OUT}/report.txt","w") as fh:
    fh.write("\n".join(log_lines)+"\n")

json.dump(dict(model=MODEL, n_feat_input=int(len(feat_in)),
               n_clock_nonzero=int(len(clock_nonzero_mouse)),
               n_clock_nonzero_symbol=int(len(clock_nonzero_syms)),
               universe_N=int(N), universe_K=int(K)),
          open(f"{OUT}/meta.json","w"), indent=2, ensure_ascii=False)
log("")
log(f"[saved] {OUT}/report.txt , overlap_table.csv , clock_gene_map.csv , meta.json")


import numpy as np, pandas as pd
from scipy.stats import spearmanr, fisher_exact
BASE="${PAPER_DIR}"
OUT=f"{BASE}/results/tage_circularity/a2_or_recompute"
ANNOT=f"{BASE}/results/singlecell/scd_agingcre/aging_cre_annot.csv"
TAGE=f"{BASE}/results/tage_circularity/a2_tage_recompute/recompute_merged.csv"
MIN=10; SCOPES=["base","ablALL","ablNZ"]
def log(*a): print(*a,flush=True)
z=np.load(f"{OUT}/cpm_extract.npz",allow_pickle=True)
cpm=z["cpm"]; mc_s=z["samples"]; mc_c=z["celltypes"]
annot=pd.read_csv(ANNOT)
tg=pd.read_csv(TAGE); tage={sc:dict(zip(zip(tg.Sample,tg.CellType),tg[f"yugene_{sc}"])) for sc in SCOPES}

rho={sc:np.full(len(annot),np.nan) for sc in SCOPES}
for i,r in enumerate(annot.itertuples()):
    ct=r.cell_type; sel=np.where(mc_c==ct)[0]
    cv=cpm[i,sel]; ok=~np.isnan(cv)
    if ok.sum()<MIN: continue
    ss=mc_s[sel][ok]; ev=cv[ok]
    for sc in SCOPES:
        yv=np.array([tage[sc].get((s,ct),np.nan) for s in ss]); m=~np.isnan(yv)
        if m.sum()>=MIN: rho[sc][i]=spearmanr(yv[m],ev[m])[0]
df=annot[["cre_id","cell_type","proximity","dir","rho_aida","rho_synapse"]].copy()
for sc in SCOPES: df[f"rho_{sc}"]=rho[sc]
df.to_csv(f"{OUT}/rho_all.csv",index=False)

def OR(dirs,prox):
    d=pd.DataFrame({"dir":dirs,"prox":prox.values}).dropna()
    ud=((d.dir=="up")&(d.prox=="distal")).sum(); dd=((d.dir=="down")&(d.prox=="distal")).sum()
    up=((d.dir=="up")&(d.prox=="proximal")).sum(); dp=((d.dir=="down")&(d.prox=="proximal")).sum()
    orr,_=fisher_exact([[ud,dd],[up,dp]])
    return float(orr),100*ud/max(ud+dd,1),100*up/max(up+dp,1),int(ud),int(dd),int(up),int(dp)

R=["=== tAge循環性 a2 最終判定 ==="]

o=OR(annot.dir.values,annot.proximity)
R.append(f"[保存] OR={o[0]:.2f} 遠位上昇{o[1]:.1f}%/近位上昇{o[2]:.1f}% ※論文14.83")
R.append("")
R.append("【判定1: 純AIDA側】dir = sign(rho_AIDA)。ablationはAIDA tAgeのみ → 循環の直撃部位")
for sc in SCOPES:
    d=df.dropna(subset=[f"rho_{sc}"]).copy()
    dirs=np.where(d[f"rho_{sc}"]>0,"up","down")
    o=OR(dirs,d.proximity)
    flip=100*np.mean(dirs!=annot.loc[d.index,"dir"].values)
    R.append(f"  [{sc:6s}] n={len(d)} OR={o[0]:6.2f}  遠位上昇{o[1]:.1f}%/近位上昇{o[2]:.1f}%  "
             f"(distal up{o[3]}/dn{o[4]}, prox up{o[5]}/dn{o[6]}) 保存dirからflip={flip:.1f}%")
R.append("")
R.append("【判定2: 論文dir準拠】dir = sign(mean(rho_AIDA_ablated, rho_synapse)); synapseは非ablation")
for sc in SCOPES:
    d=df.dropna(subset=[f"rho_{sc}","rho_synapse"]).copy()
    rmean=(d[f"rho_{sc}"]+d.rho_synapse)/2
    dirs=np.where(rmean>0,"up","down")
    o=OR(dirs,d.proximity)
    R.append(f"  [{sc:6s}] n={len(d)} OR={o[0]:6.2f}  遠位上昇{o[1]:.1f}%/近位上昇{o[2]:.1f}%")
R.append("")
R.append("--- 細胞種別 OR (純AIDA側: base→ablNZ) ---")
for ct,g in df.groupby("cell_type"):
    gb=g.dropna(subset=["rho_base"]); ga=g.dropna(subset=["rho_ablNZ"])
    if len(gb)<20: continue
    ob=OR(np.where(gb.rho_base>0,"up","down"),gb.proximity)
    oa=OR(np.where(ga.rho_ablNZ>0,"up","down"),ga.proximity)
    R.append(f"  {ct:14s} n={len(gb):4d} OR base={ob[0]:6.2f} → ablNZ={oa[0]:6.2f}  (遠位上昇% {ob[1]:.1f}→{oa[1]:.1f})")
txt="\n".join(R)+"\n"; open(f"{OUT}/report_final.txt","w").write(txt); log(txt)
log(f"[saved] {OUT}/report_final.txt, rho_all.csv")


import os, pickle, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
BASE="${PAPER_DIR}"
ANNOT=f"{BASE}/results/singlecell/scd_agingcre/aging_cre_annot.csv"
CACHE="${PROJECT_DIR}/results/phase3b_cre_corr/aggregation_cache_aida_celltype.pkl"
TAGE=f"{BASE}/results/tage_circularity/a2_tage_recompute/recompute_merged.csv"
OUT=f"{BASE}/results/tage_circularity/a2_or_recompute"; os.makedirs(OUT,exist_ok=True)
MIN_CELLS=10; MIN_METACELLS=10
def log(*a): print(*a,flush=True)
annot=pd.read_csv(ANNOT)
with open(CACHE,"rb") as f: D=pickle.load(f)
cres=[str(x) for x in D["cres"]]; cre2idx={c:i for i,c in enumerate(cres)}
ES,US,NC=D["expr_sum"],D["umi_sum"],D["n_cells"]
mc0=next(iter(ES))
def arr(v): return np.asarray(v.todense()).ravel() if sp.issparse(v) else np.asarray(v).ravel()
log(f"[shape] es={arr(ES[mc0]).shape} us={arr(US[mc0]).shape} nc={arr(NC[mc0]).shape}  mc例={mc0}")
fro_idx=np.array([cre2idx[c] for c in annot.cre_id])
mckeys=list(ES.keys())
def per_cre(v):
    a=arr(v); return a if a.size==len(cres) else None

cpm=np.full((len(annot),len(mckeys)),np.nan)
mc_sc=[]
for j,mc in enumerate(mckeys):
    coh,s1,s2,ct=mc; mc_sc.append((s1,ct))
    es=arr(ES[mc]); us=arr(US[mc]); nc=arr(NC[mc])
    es_f=es[fro_idx]
    tot=us[0] if us.size==1 else us[fro_idx]
    ncf=nc[fro_idx] if nc.size==len(cres) else np.full(len(fro_idx), nc[0] if nc.size==1 else np.nan)
    with np.errstate(divide='ignore',invalid='ignore'):
        c=es_f/np.where(np.asarray(tot)==0,np.nan,tot)*1e6
    c=np.where(ncf>=MIN_CELLS, c, np.nan)
    cpm[:,j]=c
np.savez_compressed(f"{OUT}/cpm_extract.npz",
    cpm=cpm.astype(np.float32),
    samples=np.array([s for s,_ in mc_sc]), celltypes=np.array([c for _,c in mc_sc]),
    cre_id=annot.cre_id.values, row_cell_type=annot.cell_type.values)
log(f"[saved] cpm_extract.npz  cpm shape={cpm.shape}")

tg=pd.read_csv(TAGE); tage_base=dict(zip(zip(tg.Sample,tg.CellType),tg.yugene_base))
mc_s=np.array([s for s,_ in mc_sc]); mc_c=np.array([c for _,c in mc_sc])
rho_base=np.full(len(annot),np.nan)
for i,r in enumerate(annot.itertuples()):
    ct=r.cell_type; sel=np.where(mc_c==ct)[0]
    cv=cpm[i,sel]; ok=~np.isnan(cv)
    if ok.sum()<MIN_METACELLS: continue
    ss=mc_s[sel][ok]; ev=cv[ok]
    yv=np.array([tage_base.get((s,ct),np.nan) for s in ss]); m=~np.isnan(yv)
    if m.sum()>=MIN_METACELLS: rho_base[i]=spearmanr(yv[m],ev[m])[0]
v=pd.DataFrame({"rb":rho_base,"ra":annot.rho_aida.values}).dropna()
log(f"[sanity] testable={len(v)}/{len(annot)}  rho_base vs rho_aida: Pearson={np.corrcoef(v.rb,v.ra)[0,1]:.4f} "
    f"Spearman={spearmanr(v.rb,v.ra)[0]:.4f} 符号一致={100*(np.sign(v.rb)==np.sign(v.ra)).mean():.1f}%")
np.save(f"{OUT}/rho_base_recompute.npy", rho_base)


import os, numpy as np, pandas as pd
from scipy.stats import fisher_exact
P="${PROJECT_DIR}/results"
AID=f"{P}/phase3b_cre_corr/cre_corr_aida_celltype_significant.csv"
SYN=f"{P}/phase3b_cre_corr/cre_corr_synapse_celltype_full.csv"
NE =f"{P}/phase6_novel_enhancer/novel_enhancer_classification.csv"
OUT=f"{P}/PAPER_CRE_MECHANISM/results/tage_circularity/a3_heldout_synapse"
os.makedirs(OUT,exist_ok=True)
TV="yugene_diff_EN_tAge"; R=[]
def log(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); R.append(s)

def OR(dir_arr, prox_arr):
    d=pd.DataFrame({"dir":dir_arr,"prox":np.asarray(prox_arr)}).dropna()
    ud=((d.dir=="up")&(d.prox=="distal")).sum();   dd=((d.dir=="down")&(d.prox=="distal")).sum()
    up=((d.dir=="up")&(d.prox=="proximal")).sum(); dp=((d.dir=="down")&(d.prox=="proximal")).sum()
    orr,_=fisher_exact([[ud,dd],[up,dp]])
    return dict(OR=round(float(orr),2), pud=round(100*ud/max(ud+dd,1),1), pup=round(100*up/max(up+dp,1),1),
                ud=int(ud),dd=int(dd),up=int(up),dp=int(dp),n=int(len(d)))

log("="*78); log("U2-① AIDA単独選抜 → Synapse held-out 再現"); log("="*78)

ne=pd.read_csv(NE, usecols=["cre_id","proximity"]).dropna()
ne=ne[ne.proximity.isin(["proximal","distal"])].drop_duplicates("cre_id")
prox=dict(zip(ne.cre_id, ne.proximity))
log(f"[prox] cre with proximity: {len(prox)} (distal {sum(v=='distal' for v in prox.values())}, proximal {sum(v=='proximal' for v in prox.values())})")

a=pd.read_csv(AID, usecols=["cre_id","cell_type","target_var","spearman_rho","fdr"])
a=a[(a.target_var==TV)&(a.fdr<0.05)].copy()
a["prox"]=a.cre_id.map(prox); a=a.dropna(subset=["prox"])
a["dir_aida"]=np.where(a.spearman_rho>0,"up","down")
log(f"[AIDA単独] fdr<0.05 & proximity付き: {len(a)} (cre×celltype)  ※Synapse未使用")
od=OR(a.dir_aida, a.prox)
log(f"  OR_discovery (AIDA方向) = {od['OR']}  遠位上昇{od['pud']}%/近位上昇{od['pup']}%  (distal up{od['ud']}/dn{od['dd']}, prox up{od['up']}/dn{od['dp']})")

s=pd.read_csv(SYN, usecols=["cre_id","cell_type","target_var","spearman_rho","fdr"])
s=s[s.target_var==TV].copy()
s=s.rename(columns={"spearman_rho":"rho_syn","fdr":"fdr_syn"})
m=a.merge(s[["cre_id","cell_type","rho_syn","fdr_syn"]], on=["cre_id","cell_type"], how="left")
tested=m.dropna(subset=["rho_syn"])
log(f"\n[held-out] AIDA選抜のうち Synapse で被検された: {len(tested)}/{len(a)} ({100*len(tested)/len(a):.1f}%)")
m2=tested.copy(); m2["dir_syn"]=np.where(m2.rho_syn>0,"up","down")
oh=OR(m2.dir_syn, m2.prox)
log(f"  ★OR_heldout (Synapse方向, 選抜未使用) = {oh['OR']}  遠位上昇{oh['pud']}%/近位上昇{oh['pup']}%  (distal up{oh['ud']}/dn{oh['dd']}, prox up{oh['up']}/dn{oh['dp']})")

ud=m2[(m2.dir_aida=="up")&(m2.prox=="distal")]
dp=m2[(m2.dir_aida=="down")&(m2.prox=="proximal")]
log(f"  AIDA-up-distal がSynapseでも up: {100*(ud.rho_syn>0).mean():.1f}% (n={len(ud)})")
log(f"  AIDA-down-proximal がSynapseでも down: {100*(dp.rho_syn<0).mean():.1f}% (n={len(dp)})")
log(f"  全体 AIDA方向とSynapse方向の符号一致: {100*(m2.dir_aida==m2.dir_syn).mean():.1f}%  (chance≈50%)")

ob=m2[m2.fdr_syn<0.05].copy(); ob_dir=np.where(ob.rho_syn>0,"up","down")
obr=OR(ob_dir, ob.prox)
log(f"\n[参照] both_sig相当(AIDA選抜 ∩ Synapse fdr<0.05, n={len(ob)}): OR={obr['OR']} 遠位上昇{obr['pud']}%/近位上昇{obr['pup']}%  ※元7,508 OR14.83")

log("\n--- 細胞種別 OR_heldout (Synapse方向, AIDA選抜) ---")
for ct,g in m2.groupby("cell_type"):
    if len(g)<30: continue
    o=OR(np.where(g.rho_syn>0,"up","down"), g.prox)
    oda=OR(g.dir_aida, g.prox)
    log(f"  {ct:14s} n={len(g):5d}  OR_disc(AIDA)={oda['OR']:6.2f} → OR_heldout(Syn)={o['OR']:6.2f}  (遠位上昇% {oda['pud']}→{o['pud']})")

m2.to_csv(f"{OUT}/aida_selected_synapse_heldout.csv", index=False)
open(f"{OUT}/report.txt","w").write("\n".join(R)+"\n")
log(f"\n[判定] OR_heldout が高い(>>1)= 選抜に使っていないSynapseで2プログラムが再現 = double-dipping でない実在の構造。")
log(f"[saved] {OUT}/report.txt, aida_selected_synapse_heldout.csv")
