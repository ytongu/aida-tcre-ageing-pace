local({
args <- commandArgs(trailingOnly = TRUE)
LEVEL <- if (length(args) >= 1) args[1] else "celltype"

Sys.setenv(RETICULATE_PYTHON = "${CONDA_ENVS}/tage/bin/python")

suppressPackageStartupMessages({
  library(tAge)
})

TS <- format(Sys.time(), "%Y%m%d_%H%M%S")
BASE <- "${PIPELINE_DIR}"
LEVEL_DIR <- "${PAPER_DIR}/results/tage_v110_check"
EXPRS <- "${PIPELINE_DIR}/results/06_tage/subcluster/Exprs_metacells_all.csv"
META  <- "${PIPELINE_DIR}/results/06_tage/subcluster/Metadata_metacells_all.csv"
MODEL_DIR <- "${DATA_DIR}/tage_models"
LOG <- file.path("${PAPER_DIR}/results/tage_v110_check", "run.log")

if (!file.exists(EXPRS) || !file.exists(META)) {
  stop("Missing expr/meta csv in: ", LEVEL_DIR)
}

cat("=== 35b_run_tage.R [LEVEL=", LEVEL, "] ===\n", sep="")
cat("Expr: ", EXPRS, "\n")
cat("Meta: ", META, "\n")
cat("Out:  ", LEVEL_DIR, "\n")

species <- "human"
species_tag <- "Multispecies"
model_types <- c("Chronoage", "Mortality", "NormalizedAge")
modes <- c("EN", "BR")

discover_models <- function() {
  all_pkls <- list.files(MODEL_DIR, pattern = "\\.pkl$", full.names = FALSE)
  out <- list()
  for (mt in model_types) {
    for (md in modes) {
      pat <- sprintf("^%s_%s_%s_(.+?)_(scaleddiff|yugenediff)\\.pkl$", md, mt, species_tag)
      matched <- grep(pat, all_pkls, value = TRUE, perl = TRUE)
      if (length(matched) == 0) next
      tissue <- unique(sub(pat, "\\1", matched, perl = TRUE))
      for (t in tissue) {
        sd <- grep(sprintf("^%s_%s_%s_%s_scaleddiff\\.pkl$", md, mt, species_tag, t), matched, value = TRUE)
        yg <- grep(sprintf("^%s_%s_%s_%s_yugenediff\\.pkl$", md, mt, species_tag, t), matched, value = TRUE)
        if (length(sd) == 1 && length(yg) == 1) {
          key <- paste(mt, md, species_tag, t, sep = "_")
          out[[key]] <- list(
            model_type = mt, mode = md, tissue = t,
            model_paths = list(
              scaled_diff = file.path(MODEL_DIR, sd),
              yugene_diff = file.path(MODEL_DIR, yg)
            )
          )
          cat("  Found: ", key, "\n", sep="")
        }
      }
    }
  }
  out
}

model_list <- discover_models()
cat("Total model sets: ", length(model_list), "\n", sep="")
if (length(model_list) == 0) stop("No human Multispecies models found")

cat("\nLoading expr...\n")
exprs_data <- read.csv(EXPRS, row.names = 1, check.names = FALSE)
cat("  expr: ", nrow(exprs_data), " genes x ", ncol(exprs_data), " metacells\n", sep="")

cat("Loading meta...\n")
metadata <- read.csv(META, row.names = 1, check.names = FALSE)
cat("  meta: ", nrow(metadata), " rows, cols: ", paste(colnames(metadata), collapse=","), "\n", sep="")

common <- intersect(colnames(exprs_data), rownames(metadata))
if (length(common) < ncol(exprs_data)) {
  cat("  WARN: ", ncol(exprs_data) - length(common), " expr columns not in meta\n")
}
exprs_data <- exprs_data[, common, drop = FALSE]
metadata   <- metadata[common, , drop = FALSE]
eset <- make_ExpressionSet(exprs_data, metadata, verbose = FALSE)

all_results <- list()
run_count <- 0
err_count <- 0

for (key in names(model_list)) {
  mi <- model_list[[key]]
  cat("\nRunning: ", key, "\n", sep="")
  res <- tryCatch({
    tAge_by_group(
      eset = eset,
      split_by = "CellType",
      model_paths = mi$model_paths,
      species = species,
      mode = mi$mode,
      control_group_column = "Treatment",
      control_group_label = "all",
      count_threshold = 10,
      percent_threshold = 20,
      min_samples = 3,
      verbose = TRUE
    )
  }, error = function(e) {
    cat("  ERROR [", key, "]: ", e$message, "\n", sep="")
    err_count <<- err_count + 1
    NULL
  })
  if (is.null(res)) next
  res$model_type <- mi$model_type
  res$mode       <- mi$mode
  res$tissue_model <- mi$tissue
  run_count <- run_count + 1
  all_results[[key]] <- res

  csv_out <- file.path(LEVEL_DIR, sprintf("tage_%s_%s_%s_%s.csv", mi$model_type, mi$mode, mi$tissue, TS))
  write.csv(res, csv_out, row.names = FALSE)
  cat("  Saved: ", basename(csv_out), "\n", sep="")
}

if (length(all_results) == 0) stop("No tAge results produced. ", err_count, " errors.")

all_cols <- unique(unlist(lapply(all_results, colnames)))
combined <- do.call(rbind, lapply(all_results, function(df) {
  for (c in setdiff(all_cols, colnames(df))) df[[c]] <- NA
  df[, all_cols, drop = FALSE]
}))
rownames(combined) <- NULL
combined_csv <- file.path(LEVEL_DIR, sprintf("tage_all_combined_%s.csv", TS))
write.csv(combined, combined_csv, row.names = FALSE)

cat("\n=== Summary ===\n")
cat("Successful: ", run_count, "\n", sep="")
cat("Failed:     ", err_count, "\n", sep="")
cat("Combined:   ", nrow(combined), " rows x ", ncol(combined), " cols\n", sep="")
cat("Output:     ", combined_csv, "\n", sep="")
})

local({
args <- commandArgs(trailingOnly = TRUE)
LEVEL <- if (length(args) >= 1) args[1] else "celltype"

Sys.setenv(RETICULATE_PYTHON = "${CONDA_ENVS}/tage/bin/python")

suppressPackageStartupMessages({
  library(tAge)
})

TS <- format(Sys.time(), "%Y%m%d_%H%M%S")
BASE <- "${PIPELINE_DIR}"
LEVEL_DIR <- "${PAPER_DIR}/results/tage_v110_check/aida"
EXPRS <- "${DATA_DIR}/lnc_project_AIDA_dataset/by_subcluster/Exprs_metacells_by_subcluster.csv"
META  <- "${PAPER_DIR}/results/tage_v110_check/aida/Metadata_with_treatment.csv"
MODEL_DIR <- "${DATA_DIR}/tage_models"
LOG <- file.path("${PAPER_DIR}/results/tage_v110_check/aida","run.log")

if (!file.exists(EXPRS) || !file.exists(META)) {
  stop("Missing expr/meta csv in: ", LEVEL_DIR)
}

cat("=== 35b_run_tage.R [LEVEL=", LEVEL, "] ===\n", sep="")
cat("Expr: ", EXPRS, "\n")
cat("Meta: ", META, "\n")
cat("Out:  ", LEVEL_DIR, "\n")

species <- "human"
species_tag <- "Multispecies"
model_types <- c("Chronoage", "Mortality", "NormalizedAge")
modes <- c("EN", "BR")

discover_models <- function() {
  all_pkls <- list.files(MODEL_DIR, pattern = "\\.pkl$", full.names = FALSE)
  out <- list()
  for (mt in model_types) {
    for (md in modes) {
      pat <- sprintf("^%s_%s_%s_(.+?)_(scaleddiff|yugenediff)\\.pkl$", md, mt, species_tag)
      matched <- grep(pat, all_pkls, value = TRUE, perl = TRUE)
      if (length(matched) == 0) next
      tissue <- unique(sub(pat, "\\1", matched, perl = TRUE))
      for (t in tissue) {
        sd <- grep(sprintf("^%s_%s_%s_%s_scaleddiff\\.pkl$", md, mt, species_tag, t), matched, value = TRUE)
        yg <- grep(sprintf("^%s_%s_%s_%s_yugenediff\\.pkl$", md, mt, species_tag, t), matched, value = TRUE)
        if (length(sd) == 1 && length(yg) == 1) {
          key <- paste(mt, md, species_tag, t, sep = "_")
          out[[key]] <- list(
            model_type = mt, mode = md, tissue = t,
            model_paths = list(
              scaled_diff = file.path(MODEL_DIR, sd),
              yugene_diff = file.path(MODEL_DIR, yg)
            )
          )
          cat("  Found: ", key, "\n", sep="")
        }
      }
    }
  }
  out
}

model_list <- discover_models()
cat("Total model sets: ", length(model_list), "\n", sep="")
if (length(model_list) == 0) stop("No human Multispecies models found")

cat("\nLoading expr...\n")
exprs_data <- read.csv(EXPRS, row.names = 1, check.names = FALSE)
cat("  expr: ", nrow(exprs_data), " genes x ", ncol(exprs_data), " metacells\n", sep="")

cat("Loading meta...\n")
metadata <- read.csv(META, row.names = 1, check.names = FALSE)
cat("  meta: ", nrow(metadata), " rows, cols: ", paste(colnames(metadata), collapse=","), "\n", sep="")

common <- intersect(colnames(exprs_data), rownames(metadata))
if (length(common) < ncol(exprs_data)) {
  cat("  WARN: ", ncol(exprs_data) - length(common), " expr columns not in meta\n")
}
exprs_data <- exprs_data[, common, drop = FALSE]
metadata   <- metadata[common, , drop = FALSE]
eset <- make_ExpressionSet(exprs_data, metadata, verbose = FALSE)

all_results <- list()
run_count <- 0
err_count <- 0

for (key in names(model_list)) {
  mi <- model_list[[key]]
  cat("\nRunning: ", key, "\n", sep="")
  res <- tryCatch({
    tAge_by_group(
      eset = eset,
      split_by = "CellType",
      model_paths = mi$model_paths,
      species = species,
      mode = mi$mode,
      control_group_column = "Treatment",
      control_group_label = "all",
      count_threshold = 10,
      percent_threshold = 20,
      min_samples = 3,
      verbose = TRUE
    )
  }, error = function(e) {
    cat("  ERROR [", key, "]: ", e$message, "\n", sep="")
    err_count <<- err_count + 1
    NULL
  })
  if (is.null(res)) next
  res$model_type <- mi$model_type
  res$mode       <- mi$mode
  res$tissue_model <- mi$tissue
  run_count <- run_count + 1
  all_results[[key]] <- res

  csv_out <- file.path(LEVEL_DIR, sprintf("tage_%s_%s_%s_%s.csv", mi$model_type, mi$mode, mi$tissue, TS))
  write.csv(res, csv_out, row.names = FALSE)
  cat("  Saved: ", basename(csv_out), "\n", sep="")
}

if (length(all_results) == 0) stop("No tAge results produced. ", err_count, " errors.")

all_cols <- unique(unlist(lapply(all_results, colnames)))
combined <- do.call(rbind, lapply(all_results, function(df) {
  for (c in setdiff(all_cols, colnames(df))) df[[c]] <- NA
  df[, all_cols, drop = FALSE]
}))
rownames(combined) <- NULL
combined_csv <- file.path(LEVEL_DIR, sprintf("tage_all_combined_%s.csv", TS))
write.csv(combined, combined_csv, row.names = FALSE)

cat("\n=== Summary ===\n")
cat("Successful: ", run_count, "\n", sep="")
cat("Failed:     ", err_count, "\n", sep="")
cat("Combined:   ", nrow(combined), " rows x ", ncol(combined), " cols\n", sep="")
cat("Output:     ", combined_csv, "\n", sep="")
})

local({
Sys.setenv(RETICULATE_PYTHON = "${CONDA_ENVS}/tage/bin/python")
suppressPackageStartupMessages(library(tAge))
args <- commandArgs(trailingOnly = TRUE)
TAG  <- if (length(args) >= 1) args[1] else "aida"
TS   <- format(Sys.time(), "%Y%m%d_%H%M%S")
BASE <- "${PAPER_DIR}"
DIR  <- file.path(BASE, "results", "tage_canonical", TAG)
MODEL_DIR <- "${DATA_DIR}/tage_models"
species <- "human"; species_tag <- "Multispecies"
model_types <- c("Chronoage"); modes <- c("EN")   # 正典が使うのは Chronoage/EN

discover <- function() {
  all_pkls <- list.files(MODEL_DIR, pattern = "\\.pkl$")
  out <- list()
  for (mt in model_types) for (md in modes) {
    pat <- sprintf("^%s_%s_%s_(.+?)_(scaleddiff|yugenediff)\\.pkl$", md, mt, species_tag)
    matched <- grep(pat, all_pkls, value = TRUE, perl = TRUE)
    if (!length(matched)) next
    for (t in unique(sub(pat, "\\1", matched, perl = TRUE))) {
      sd <- grep(sprintf("^%s_%s_%s_%s_scaleddiff\\.pkl$", md, mt, species_tag, t), matched, value = TRUE)
      yg <- grep(sprintf("^%s_%s_%s_%s_yugenediff\\.pkl$", md, mt, species_tag, t), matched, value = TRUE)
      if (length(sd) == 1 && length(yg) == 1)
        out[[paste(mt, md, t, sep = "_")]] <- list(model_type=mt, mode=md, tissue=t,
          model_paths=list(scaled_diff=file.path(MODEL_DIR, sd), yugene_diff=file.path(MODEL_DIR, yg)))
    }
  }
  out
}
ml <- discover(); cat("Model sets:", length(ml), "\n")
cat("Loading", TAG, "...\n")
exprs_data <- read.csv(file.path(DIR, "Exprs_metacells.csv"), row.names = 1, check.names = FALSE)
metadata   <- read.csv(file.path(DIR, "Metadata_metacells.csv"), row.names = 1, check.names = FALSE)
cat("  ", nrow(exprs_data), "genes x", ncol(exprs_data), "metacells\n")
common <- intersect(colnames(exprs_data), rownames(metadata))
exprs_data <- exprs_data[, common, drop=FALSE]; metadata <- metadata[common, , drop=FALSE]
eset <- make_ExpressionSet(exprs_data, metadata, verbose = FALSE)
res_all <- list()
for (k in names(ml)) {
  mi <- ml[[k]]; cat("\nRunning:", k, "\n")
  r <- tryCatch(tAge_by_group(eset=eset, split_by="CellType", model_paths=mi$model_paths,
        species=species, mode=mi$mode, control_group_column="Treatment",
        control_group_label="all", count_threshold=10, percent_threshold=20,
        min_samples=3, verbose=TRUE),
        error=function(e){cat("  ERROR:", e$message, "\n"); NULL})
  if (is.null(r)) next
  r$model_type <- mi$model_type; r$mode <- mi$mode
  res_all[[k]] <- r
  write.csv(r, file.path(DIR, sprintf("tage_%s_%s_%s.csv", mi$model_type, mi$mode, TS)), row.names=FALSE)
}
if (length(res_all)) {
  cmb <- do.call(rbind, res_all); rownames(cmb) <- NULL
  o <- file.path(DIR, sprintf("tage_all_combined_%s.csv", TS))
  write.csv(cmb, o, row.names=FALSE)
  cat("\nRows:", nrow(cmb), "\nOutput:", o, "\n")
}
})
