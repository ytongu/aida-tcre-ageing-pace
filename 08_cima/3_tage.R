Sys.setenv(RETICULATE_PYTHON = "${CONDA_ENVS}/tage/bin/python")
suppressPackageStartupMessages(library(tAge))

TS   <- format(Sys.time(), "%Y%m%d_%H%M%S")
DIR  <- "${PAPER_DIR}/results/cima_tage"
MODEL_DIR <- "${DATA_DIR}/tage_models"
EXPRS <- file.path(DIR, "Exprs_metacells.csv")
META  <- file.path(DIR, "Metadata_metacells.csv")

species <- "human"; species_tag <- "Multispecies"
model_types <- c("Chronoage", "Mortality", "NormalizedAge"); modes <- c("EN", "BR")

discover_models <- function() {
  all_pkls <- list.files(MODEL_DIR, pattern = "\\.pkl$")
  out <- list()
  for (mt in model_types) for (md in modes) {
    pat <- sprintf("^%s_%s_%s_(.+?)_(scaleddiff|yugenediff)\\.pkl$", md, mt, species_tag)
    matched <- grep(pat, all_pkls, value = TRUE, perl = TRUE)
    if (!length(matched)) next
    for (t in unique(sub(pat, "\\1", matched, perl = TRUE))) {
      sd <- grep(sprintf("^%s_%s_%s_%s_scaleddiff\\.pkl$", md, mt, species_tag, t), matched, value = TRUE)
      yg <- grep(sprintf("^%s_%s_%s_%s_yugenediff\\.pkl$", md, mt, species_tag, t), matched, value = TRUE)
      if (length(sd) == 1 && length(yg) == 1) {
        out[[paste(mt, md, species_tag, t, sep="_")]] <- list(
          model_type=mt, mode=md, tissue=t,
          model_paths=list(scaled_diff=file.path(MODEL_DIR, sd),
                           yugene_diff=file.path(MODEL_DIR, yg)))
        cat("  Found:", paste(mt, md, t), "\n")
      }
    }
  }
  out
}
model_list <- discover_models()
cat("Model sets:", length(model_list), "\n")

cat("Loading expr...\n")
exprs_data <- read.csv(EXPRS, row.names = 1, check.names = FALSE)
cat("  ", nrow(exprs_data), "genes x", ncol(exprs_data), "metacells\n")
metadata <- read.csv(META, row.names = 1, check.names = FALSE)
cat("  meta:", nrow(metadata), "rows |", paste(colnames(metadata), collapse=","), "\n")

common <- intersect(colnames(exprs_data), rownames(metadata))
exprs_data <- exprs_data[, common, drop=FALSE]; metadata <- metadata[common, , drop=FALSE]
eset <- make_ExpressionSet(exprs_data, metadata, verbose = FALSE)

all_results <- list(); ok <- 0; err <- 0
for (key in names(model_list)) {
  mi <- model_list[[key]]
  cat("\nRunning:", key, "\n")
  res <- tryCatch(
    tAge_by_group(eset=eset, split_by="CellType", model_paths=mi$model_paths,
                  species=species, mode=mi$mode,
                  control_group_column="Treatment", control_group_label="all",
                  count_threshold=10, percent_threshold=20, min_samples=3, verbose=TRUE),
    error=function(e){ cat("  ERROR:", e$message, "\n"); err <<- err+1; NULL })
  if (is.null(res)) next
  res$model_type <- mi$model_type; res$mode <- mi$mode; res$tissue_model <- mi$tissue
  all_results[[key]] <- res; ok <- ok + 1
  write.csv(res, file.path(DIR, sprintf("tage_%s_%s_%s_%s.csv", mi$model_type, mi$mode, mi$tissue, TS)),
            row.names = FALSE)
}
if (!length(all_results)) stop("No tAge results. errors=", err)
all_cols <- unique(unlist(lapply(all_results, colnames)))
combined <- do.call(rbind, lapply(all_results, function(df){
  for (c in setdiff(all_cols, colnames(df))) df[[c]] <- NA
  df[, all_cols, drop=FALSE] }))
rownames(combined) <- NULL
out <- file.path(DIR, sprintf("tage_all_combined_%s.csv", TS))
write.csv(combined, out, row.names = FALSE)
cat("\n=== Summary ===\nSuccessful:", ok, "\nFailed:", err, "\nRows:", nrow(combined), "\nOutput:", out, "\n")
