set.seed(42)
Sys.setenv(RETICULATE_PYTHON = "${CONDA_ENVS}/tage/bin/python")
suppressPackageStartupMessages({ library(tAge) })

BASE   <- "${PAPER_DIR}"
EXPRS  <- "${DATA_DIR}/lnc_project_AIDA_dataset/by_cluster/Exprs_metacells_by_cluster.csv"
META   <- "${DATA_DIR}/lnc_project_AIDA_dataset/by_cluster/Metadata_metacells_by_cluster.csv"
PUB    <- "${DATA_DIR}/lnc_project_AIDA_dataset/by_cluster/tage/tage_Chronoage_EN.csv"
MDIR   <- "${DATA_DIR}/tage_models"
UPD    <- file.path(BASE, "results/target_side_dichotomy/enh_up_distal_target_genes.csv")
DNP    <- file.path(BASE, "results/target_side_dichotomy/prom_down_proximal_genes.csv")
NZMAP  <- file.path(BASE, "results/tage_circularity/a1_gene_overlap/clock_gene_map.csv")
OUT    <- file.path(BASE, "results/tage_circularity/a2_tage_recompute")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

model_paths <- list(
  scaled_diff = file.path(MDIR, "EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl"),
  yugene_diff = file.path(MDIR, "EN_Chronoage_Multispecies_Multitissue_yugenediff.pkl")
)

cat("=== a2 step1: ablation-recompute tAge (Chronoage EN) ===\n")
exprs_full <- read.csv(EXPRS, row.names = 1, check.names = FALSE)
metadata   <- read.csv(META,  check.names = FALSE)
rownames(metadata) <- metadata$ID
cat(sprintf("exprs: %d genes x %d metacells | meta: %d rows | celltypes: %s\n",
            nrow(exprs_full), ncol(exprs_full), nrow(metadata),
            paste(unique(metadata$CellType), collapse=",")))

up   <- read.csv(UPD)$gene;  dn <- read.csv(DNP)$gene
nz   <- read.csv(NZMAP)$clock_nonzero_gene_symbol
prog_all <- unique(c(up, dn))
prog_nz  <- intersect(prog_all, nz)
cat(sprintf("program genes: all=%d (up %d + down %d), clock-nonzero-overlap=%d\n",
            length(prog_all), length(up), length(dn), length(prog_nz)))
g <- rownames(exprs_full)
cat(sprintf("  present in exprs: all=%d, nz=%d\n",
            sum(prog_all %in% g), sum(prog_nz %in% g)))

run_scope <- function(label, drop_genes) {
  keep <- setdiff(rownames(exprs_full), drop_genes)
  ex   <- exprs_full[keep, , drop = FALSE]
  cat(sprintf("\n--- scope=%s : dropped %d genes, %d genes remain ---\n",
              label, nrow(exprs_full) - nrow(ex), nrow(ex)))
  eset <- make_ExpressionSet(ex, metadata, verbose = FALSE)
  res  <- tAge_by_group(eset = eset, split_by = "CellType",
                        model_paths = model_paths, species = "human", mode = "EN",
                        count_threshold = 10, percent_threshold = 20,
                        min_samples = 3, verbose = FALSE)
  res <- res[, c("ID","Sample","CellType","yugene_diff_EN_tAge","scaled_diff_EN_tAge")]
  names(res)[names(res)=="yugene_diff_EN_tAge"] <- paste0("yugene_", label)
  names(res)[names(res)=="scaled_diff_EN_tAge"] <- paste0("scaled_", label)
  write.csv(res, file.path(OUT, sprintf("tage_recompute_%s.csv", label)), row.names = FALSE)
  res
}

base   <- run_scope("base",   character(0))
ablALL <- run_scope("ablALL", prog_all)
ablNZ  <- run_scope("ablNZ",  prog_nz)

pub <- read.csv(PUB)
m <- merge(base[,c("ID","yugene_base")], pub[,c("ID","yugene_diff_EN_tAge")], by="ID")
r_pub  <- cor(m$yugene_base, m$yugene_diff_EN_tAge, method="pearson")
rs_pub <- cor(m$yugene_base, m$yugene_diff_EN_tAge, method="spearman")
mad_pub<- max(abs(m$yugene_base - m$yugene_diff_EN_tAge))

j <- Reduce(function(a,b) merge(a,b,by=c("ID","Sample","CellType")),
            list(base[,c("ID","Sample","CellType","yugene_base")],
                 ablALL[,c("ID","yugene_ablALL")], ablNZ[,c("ID","yugene_ablNZ")]))
r_all <- cor(j$yugene_base, j$yugene_ablALL, method="spearman")
r_nz  <- cor(j$yugene_base, j$yugene_ablNZ,  method="spearman")

vt <- c(
  "=== a2 step1 validation ===",
  sprintf("metacells matched vs published: %d", nrow(m)),
  sprintf("[baseline再現] recompute vs published yugene_diff_EN_tAge: Pearson r=%.6f, Spearman=%.6f, max|diff|=%.4g",
          r_pub, rs_pub, mad_pub),
  "  → r≈1 なら実パイプライン再現OK。ablation差分が解釈可能になる。",
  sprintf("[ablated vs base] Spearman(base, ablALL)=%.4f  (プログラム全%d遺伝子除去)", r_all, length(prog_all)),
  sprintf("[ablated vs base] Spearman(base, ablNZ )=%.4f  (時計非ゼロ%d遺伝子除去)",  r_nz,  length(prog_nz)),
  "  → 高いほど「tAgeの人ランキングは共有遺伝子に依らない」= 循環否定に有利。",
  "  次: step2 で ablated tAge を使い 7,508 CRE の Spearman/OR を再計算し方向が生存するか判定。"
)
writeLines(vt, file.path(OUT, "validation.txt"))
cat("\n"); cat(paste(vt, collapse="\n")); cat("\n")
cat(sprintf("\n[saved] %s/\n", OUT))
