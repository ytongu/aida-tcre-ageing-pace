#!/bin/bash
step1() (
set -uo pipefail

TS=$(date +%Y%m%d_%H%M%S)
CELLRANGER="/opt/cellranger-9.0.1/cellranger"
FASTQ_BASE="${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/01_download_fastq"
NAS_OUT="${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/02_cellranger_nig"
SSD_WORK="$HOME/projects/lnc_project_scafe/results/cr_work"
LOGDIR="$HOME/projects/lnc_project_scafe/logs"
REF="${NAS}/genome/refdata-gex-GRCh38-2024-A"

mkdir -p "$SSD_WORK" "$LOGDIR"

SAMPLES=(
    "AS054 ALAW-79878-AS054-2-5GEX AS054-2"
    "AS054 ALAW-79881-AS054-5-5GEX AS054-5"
    "AS054 ALAW-79882-AS054-6-5GEX AS054-6"
    "AS055 ALAW-AS055-7-5-GEX-AS055-7-5-GEX AS055-7"
    "AS057 ALAW-80039-AS057-3-5GEX AS057-3"
    "AS057 ALAW-80042-AS057-6-5GEX AS057-6"
    "AS057 ALAW-80044-AS057-8-5GEX AS057-8"
    "AS058 ALAW-80101-AS058-1-5GEX AS058-1"
    "AS058 ALAW-80102-AS058-2-5GEX AS058-2"
    "AS058 ALAW-80103-AS058-3-5GEX AS058-3"
    "AS058 ALAW-80104-AS058-4-5GEX AS058-4"
    "AS058 ALAW-80106-AS058-6-5GEX AS058-6"
    "AS059 ALAW-80165-AS059-1-5GEX AS059-1"
    "AS059 ALAW-80168-AS059-4-5GEX AS059-4"
    "AS059 ALAW-80171-AS059-7-5GEX AS059-7"
    "AS060 ALAW-80297-AS060-5-5GEX AS060-5"
    "AS060 ALAW-80300-AS060-8-5GEX AS060-8"
    "AS061 ALAW-80359-AS061-3-5GEX AS061-3"
    "AS061 ALAW-80362-AS061-6-5GEX AS061-6"
    "AS062 ALAW-80421-AS062-1-5GEX AS062-1"
    "AS062 ALAW-80424-AS062-4-5GEX AS062-4"
    "AS062 ALAW-80427-AS062-7-5GEX AS062-7"
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "Starting local CR pipeline: ${#SAMPLES[@]} samples"

for entry in "${SAMPLES[@]}"; do
    read -r FASTQ_DIR SAMPLE_PREFIX SAMPLE_ID <<< "$entry"

    if [ -f "$NAS_OUT/$SAMPLE_ID/outs/possorted_genome_bam.bam" ]; then
        log "SKIP: $SAMPLE_ID (BAM already on NAS)"
        continue
    fi

    if ! ls "$FASTQ_BASE/$FASTQ_DIR/"*"$SAMPLE_PREFIX"* &>/dev/null; then
        log "SKIP: $SAMPLE_ID (FASTQ not yet on NAS — waiting for NIG transfer)"
        continue
    fi

    ssd_free=$(df --output=avail ${HOME} | tail -1 | awk '{print int($1/1024/1024)}')
    if [ "$ssd_free" -lt 200 ]; then
        log "ERROR: SSD free ${ssd_free}GB < 200GB. Stopping."
        exit 1
    fi

    mem_free=$(free -g | awk '/Mem:/{print $7}')
    if [ "$mem_free" -lt 64 ]; then
        log "ERROR: Free memory ${mem_free}GB < 64GB. Stopping."
        exit 1
    fi

    log "START: $SAMPLE_ID (SSD free: ${ssd_free}GB, Mem free: ${mem_free}GB)"

    cd "$SSD_WORK"
    [ -d "$SAMPLE_ID" ] && rm -rf "$SAMPLE_ID"

    $CELLRANGER count \
        --id="$SAMPLE_ID" \
        --create-bam=true \
        --transcriptome="$REF" \
        --fastqs="$FASTQ_BASE/$FASTQ_DIR" \
        --sample="$SAMPLE_PREFIX" \
        --localcores=32 \
        --localmem=256 \
        2>&1 | tee "$LOGDIR/25_cr_${SAMPLE_ID}_${TS}.log"

    if [ -f "$SSD_WORK/$SAMPLE_ID/outs/possorted_genome_bam.bam" ]; then
        log "DONE: $SAMPLE_ID — transferring to NAS..."

        dest="$NAS_OUT/$SAMPLE_ID/outs"
        mkdir -p "$dest/filtered_feature_bc_matrix" "$dest/raw_feature_bc_matrix"
        rsync -a "$SSD_WORK/$SAMPLE_ID/outs/possorted_genome_bam.bam" "$dest/"
        rsync -a "$SSD_WORK/$SAMPLE_ID/outs/possorted_genome_bam.bam.bai" "$dest/" 2>/dev/null || true
        rsync -a "$SSD_WORK/$SAMPLE_ID/outs/filtered_feature_bc_matrix/" "$dest/filtered_feature_bc_matrix/"
        rsync -a "$SSD_WORK/$SAMPLE_ID/outs/metrics_summary.csv" "$dest/" 2>/dev/null || true
        rsync -a "$SSD_WORK/$SAMPLE_ID/outs/raw_feature_bc_matrix/" "$dest/raw_feature_bc_matrix/" 2>/dev/null || true
        rsync -a "$SSD_WORK/$SAMPLE_ID/outs/filtered_feature_bc_matrix.h5" "$dest/" 2>/dev/null || true

        log "TRANSFERRED: $SAMPLE_ID — cleaning SSD..."
        rm -rf "$SSD_WORK/$SAMPLE_ID"
        log "CLEANED: $SAMPLE_ID (SSD free: $(df --output=avail ${HOME} | tail -1 | awk '{print int($1/1024/1024)}')GB)"
    else
        log "FAIL: $SAMPLE_ID"
        rm -rf "$SSD_WORK/$SAMPLE_ID"
    fi
done

log "All samples processed."
)

step2() (
set -uo pipefail

PROJECT="$HOME/projects/lnc_project_scafe"
NAS_CR_LOCAL="${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/02_cellranger"
NAS_CR_NIG="${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/02_cellranger_nig"
SCAFE_OUTDIR="$PROJECT/results/03_scafe"
SCAFE_TMP="${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/scafe_tmp"
SCAFE_GENOME="$PROJECT/results/scafe_genome/genome_dl"
LOGDIR="$PROJECT/logs"
DONE_FILE="$LOGDIR/.scafe_done.txt"
LOCKFILE="$LOGDIR/.scafe_pipeline.lock"
LOCKDIR="$LOGDIR/.scafe_locks"
MAX_PARALLEL=2
SKIP_FILE="$LOGDIR/.scafe_skip.txt"  # CRやり直し等でスキップするサンプル

mkdir -p "$LOCKDIR"
touch "$DONE_FILE"
touch "$SKIP_FILE"

exec 200>"$LOCKFILE"
if ! flock -n 200; then
    echo "ERROR: Another pipeline instance is already running (lockfile: $LOCKFILE)"
    echo "       Kill it first or remove $LOCKFILE manually."
    exit 1
fi
trap 'rm -f "$LOCKFILE" "$LOCKDIR"/*.lock 2>/dev/null' EXIT

sync_done_file() {
    local tmp_done=$(mktemp)
    while IFS= read -r sid; do
        if [ -d "$SCAFE_OUTDIR/$sid/directionality" ]; then
            echo "$sid"
        else
            echo "[SYNC] Removed false entry: $sid" >&2
        fi
    done < "$DONE_FILE" | sort -u > "$tmp_done"
    mv "$tmp_done" "$DONE_FILE"
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

get_pending_samples() {
    local samples=()

    for bam in "$NAS_CR_LOCAL"/*/outs/possorted_genome_bam.bam; do
        [ ! -f "$bam" ] && continue
        local sid=$(echo "$bam" | sed "s|$NAS_CR_LOCAL/||;s|/outs/.*||")
        grep -qx "$sid" "$DONE_FILE" 2>/dev/null && continue
        grep -qx "$sid" "$SKIP_FILE" 2>/dev/null && continue
        [ -f "$LOCKDIR/${sid}.lock" ] && continue
        samples+=("local|$sid")
    done

    for bam in "$NAS_CR_NIG"/*/outs/possorted_genome_bam.bam; do
        [ ! -f "$bam" ] && continue
        local sid=$(echo "$bam" | sed "s|$NAS_CR_NIG/||;s|/outs/.*||")
        grep -qx "$sid" "$DONE_FILE" 2>/dev/null && continue
        grep -qx "$sid" "$SKIP_FILE" 2>/dev/null && continue
        [ -f "$LOCKDIR/${sid}.lock" ] && continue
        samples+=("nig|$sid")
    done

    echo "${samples[@]}"
}

process_sample() {
    local source=$1 sample_id=$2
    local ts=$(date +%Y%m%d_%H%M%S)
    local tmp_dir="$SCAFE_TMP/$sample_id"
    local SCAFE_OUT="$SCAFE_OUTDIR/$sample_id"
    local sample_lock="$LOCKDIR/${sample_id}.lock"

    echo $$ > "$sample_lock"
    trap "rm -f '$sample_lock'" RETURN

    local nas_dir
    if [ "$source" = "local" ]; then
        nas_dir="$NAS_CR_LOCAL/$sample_id"
    else
        nas_dir="$NAS_CR_NIG/$sample_id"
    fi

    local src_bam="$nas_dir/outs/possorted_genome_bam.bam"
    local src_bc="$nas_dir/outs/filtered_feature_bc_matrix/barcodes.tsv.gz"
    if [ ! -s "$src_bam" ] || [ ! -s "$src_bc" ]; then
        log "SCAFE SKIP: $sample_id (source missing: BAM=$src_bam BC=$src_bc)"
        return 0
    fi

    mkdir -p "$tmp_dir/outs/filtered_feature_bc_matrix"
    log "$sample_id: copying BAM/BAI/fbm to tmp..."

    local src_bam_size dst_bam_size
    src_bam_size=$(stat -c %s "$src_bam" 2>/dev/null || echo 0)
    dst_bam_size=$(stat -c %s "$tmp_dir/outs/possorted_genome_bam.bam" 2>/dev/null || echo 0)
    if [ "$src_bam_size" = "$dst_bam_size" ] && [ "$src_bam_size" != "0" ]; then
        log "$sample_id: BAM already complete ($src_bam_size bytes), skip rsync"
    else
        if ! rsync -a "$src_bam" "$tmp_dir/outs/"; then
            log "SCAFE FAIL: $sample_id (BAM rsync failed)"
            return 1
        fi
    fi
    rsync -a "$src_bam.bai" "$tmp_dir/outs/" 2>/dev/null || true

    if ! rsync -a "$nas_dir/outs/filtered_feature_bc_matrix/" "$tmp_dir/outs/filtered_feature_bc_matrix/"; then
        log "SCAFE FAIL: $sample_id (fbm rsync failed)"
        return 1
    fi

    local tmp_bam="$tmp_dir/outs/possorted_genome_bam.bam"
    local tmp_bc="$tmp_dir/outs/filtered_feature_bc_matrix/barcodes.tsv.gz"
    local tmp_bam_size
    tmp_bam_size=$(stat -c %s "$tmp_bam" 2>/dev/null || echo 0)
    if [ "$tmp_bam_size" != "$src_bam_size" ]; then
        log "SCAFE FAIL: $sample_id (BAM size mismatch: src=$src_bam_size tmp=$tmp_bam_size)"
        return 1
    fi
    if [ ! -s "$tmp_bc" ]; then
        log "SCAFE FAIL: $sample_id (barcodes.tsv.gz missing in tmp: $tmp_bc)"
        return 1
    fi
    log "$sample_id: copy verified (BAM=${tmp_bam_size}b, BC=$(stat -c %s "$tmp_bc")b)"

    [ -d "$SCAFE_OUT" ] && rm -rf "$SCAFE_OUT"
    mkdir -p "$SCAFE_OUT"

    log "SCAFE START: $sample_id"
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -v ${NAS}:${NAS} \
        -v "$HOME:$HOME" \
        -v "$SCAFE_GENOME:/SCAFE/resources/genome" \
        scafe:latest \
        perl /SCAFE/scripts/scafe.workflow.sc.solo \
        --run_bam_path "$tmp_bam" \
        --run_cellbarcode_path "$tmp_bc" \
        --genome hg38.gencode_v32 \
        --run_tag "$sample_id" \
        --run_outDir "$SCAFE_OUT" \
        --detect_TS_oligo trim \
        --max_thread 10 \
        2>&1 | head -c 100M | tee "$LOGDIR/03_scafe_${sample_id}_${ts}.log"

    if [ -d "$SCAFE_OUT/directionality" ]; then
        log "SCAFE DONE: $sample_id"
        echo "$sample_id" >> "$DONE_FILE"
        local NAS_SCAFE="${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/03_scafe"
        if [ ! -L "$SCAFE_OUT" ]; then
            rsync -a "$SCAFE_OUT/" "$NAS_SCAFE/$sample_id/" && rm -rf "$SCAFE_OUT" && ln -s "$NAS_SCAFE/$sample_id" "$SCAFE_OUT"
            log "$sample_id: moved to NAS"
        fi
        rm -rf "$tmp_dir"
    else
        log "SCAFE FAIL: $sample_id (directionality not produced; tmp retained for resume at $tmp_dir)"
        rm -rf "$SCAFE_OUT"
    fi
}

transfer_nig_new() {
    local NIG_BASE="${HPC_HOME}/age_promoter_analysis/synapse_dataset/syn49637038/cellranger_output"
    local new_bams=$(ssh -o ConnectTimeout=10 ${HPC_LOGIN} "for s in AS054-{1..8} AS055-{1..8} AS056-{1..8} AS057-{1..8} AS058-{1..6} AS059-{1,4,7} AS060-{2,5,8} AS061-{3,6} AS062-{1,4,7}; do [ -f $NIG_BASE/\$s/outs/possorted_genome_bam.bam ] && echo \$s; done" 2>/dev/null || true)

    for sid in $new_bams; do
        grep -qx "$sid" "$DONE_FILE" 2>/dev/null && continue
        [ -f "$NAS_CR_NIG/$sid/outs/possorted_genome_bam.bam" ] && continue

        local dest="$NAS_CR_NIG/$sid/outs"
        mkdir -p "$dest/filtered_feature_bc_matrix"
        log "$sid: transferring from NIG to NAS..."
        rsync -a "${HPC_LOGIN}:$NIG_BASE/$sid/outs/possorted_genome_bam.bam" "$dest/"
        rsync -a "${HPC_LOGIN}:$NIG_BASE/$sid/outs/possorted_genome_bam.bam.bai" "$dest/" 2>/dev/null || true
        rsync -a "${HPC_LOGIN}:$NIG_BASE/$sid/outs/filtered_feature_bc_matrix/barcodes.tsv.gz" "$dest/filtered_feature_bc_matrix/" 2>/dev/null || true
    done
}

echo "=========================================="
echo " Parallel SCAFE Worker (${MAX_PARALLEL}x)"
echo " $(date +%Y%m%d_%H%M%S)"
echo " PID: $$"
echo " Lock: $LOCKFILE"
echo "=========================================="

log "Syncing done file..."
sync_done_file
log "Done file: $(wc -l < "$DONE_FILE") verified entries"

while true; do
    n_done=$(wc -l < "$DONE_FILE")
    log "SCAFE done: $n_done. Scanning for pending samples..."

    if ! pgrep -f "transfer_nig_bg" > /dev/null 2>&1; then
        (transfer_nig_new || true) &
    fi

    pending=($(get_pending_samples))

    if [ ${#pending[@]} -eq 0 ]; then
        local_nig_check=$(ssh -o ConnectTimeout=5 ${HPC_LOGIN} "ssh a002 'squeue -u ${HPC_USER} -h 2>/dev/null'" 2>/dev/null | wc -l)
        if [ "$local_nig_check" -gt 0 ]; then
            log "No pending samples. $local_nig_check NIG jobs still running. Waiting 5min..."
            sleep 300
        else
            log "All done! No pending samples and no NIG jobs."
            break
        fi
        continue
    fi

    log "Pending: ${#pending[@]} samples. Launching up to $MAX_PARALLEL..."

    ssd_free_gb=$(df --output=avail ${HOME} | tail -1 | awk '{print int($1/1024/1024)}')
    if [ "$ssd_free_gb" -lt 200 ]; then
        log "WARNING: SSD free ${ssd_free_gb}GB < 200GB. Waiting for space..."
        sleep 300
        continue
    fi

    pids=()
    launched=0
    for entry in "${pending[@]}"; do
        [ $launched -ge $MAX_PARALLEL ] && break
        IFS='|' read -r source sid <<< "$entry"

        docker ps --format "{{.Command}}" 2>/dev/null | grep -q "run_tag $sid" && continue

        process_sample "$source" "$sid" &
        pids+=($!)
        log "Launched: $sid (PID: ${pids[-1]})"
        launched=$((launched + 1))

        [ $launched -lt $MAX_PARALLEL ] && sleep 60
    done

    for pid in "${pids[@]}"; do
        wait $pid || true
    done

    sync_done_file
    log "Batch of $launched complete. Verified done: $(wc -l < "$DONE_FILE")"
done
)

step3() (
set -euo pipefail

TS=$(date +%Y%m%d_%H%M%S)
PROJ=${HOME}/projects/integrated_aida_synapse
NAS_BASE=${DATA_DIR}
NAS_AGG=$NAS_BASE/integrated_aida_synapse_hg38_440
mkdir -p "$NAS_AGG" "$PROJ/logs"

AIDA_BED_DIR=$NAS_BASE/lnc_project_AIDA_dataset/aida_per_sample_bed
SYN_SCAFE_DIR=$NAS_BASE/lnc_project_synapse_dataset/syn49637038/03_scafe

LIB_LIST=$NAS_AGG/lib_list_native_${TS}.tsv
LOG=$PROJ/logs/01_build_lib_list_${TS}.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== build lib_list (AIDA 333 + Synapse 107 = 440) ==="
log "AIDA bed dir: $AIDA_BED_DIR"
log "Synapse SCAFE dir: $SYN_SCAFE_DIR"
log "out: $LIB_LIST"

> "$LIB_LIST"
n_aida=0; n_syn=0; n_miss=0

AIDA_LIST=/tmp/aida_335_samples.txt
[ -f "$AIDA_LIST" ] || { log "ERROR: $AIDA_LIST not found"; exit 1; }
while read sid; do
  c="$AIDA_BED_DIR/$sid/bam_to_ctss/$sid/bed/${sid}.collapse.ctss.bed.gz"
  u="$AIDA_BED_DIR/$sid/bam_to_ctss/$sid/bed/${sid}.unencoded_G.collapse.ctss.bed.gz"
  if [ -f "$c" ] && [ -f "$u" ]; then
    echo -e "${sid}\t${c}\t${u}" >> "$LIB_LIST"
    n_aida=$((n_aida+1))
  else
    log "MISS AIDA: $sid"
    n_miss=$((n_miss+1))
  fi
done < "$AIDA_LIST"

for sample_dir in "$SYN_SCAFE_DIR"/AS*/; do
  sid=$(basename "$sample_dir")
  [ "$sid" = "AS062-2_new" ] && { log "SKIP Synapse: $sid (duplicate)"; continue; }
  c="$sample_dir/bam_to_ctss/$sid/bed/${sid}.collapse.ctss.bed.gz"
  u="$sample_dir/bam_to_ctss/$sid/bed/${sid}.unencoded_G.collapse.ctss.bed.gz"
  if [ -f "$c" ] && [ -f "$u" ]; then
    echo -e "${sid}\t${c}\t${u}" >> "$LIB_LIST"
    n_syn=$((n_syn+1))
  else
    log "MISS Synapse: $sid"
    n_miss=$((n_miss+1))
  fi
done

log "AIDA: $n_aida, Synapse: $n_syn, missing: $n_miss"
log "total: $((n_aida + n_syn))"
log "LIB_LIST: $LIB_LIST"

if [ $((n_aida + n_syn)) -lt 430 ]; then
  log "ERROR: expected ~440, got $((n_aida + n_syn))"
  exit 1
fi

ln -sf "$LIB_LIST" "$NAS_AGG/lib_list_latest.tsv"
log "symlink: $NAS_AGG/lib_list_latest.tsv"
)

step4() (
set -euo pipefail

TS=$(date +%Y%m%d_%H%M%S)
PROJ=${HOME}/projects/integrated_aida_synapse
NAS_AGG=${DATA_DIR}/integrated_aida_synapse_hg38_440
SCAFE_DIR=${SCAFE_DIR}
RUN_TAG="integrated_aida_synapse_hg38_440"
LOG=$PROJ/logs/02_aggregate_${TS}.log

LIB_LIST=$NAS_AGG/lib_list_latest.tsv
[ -f "$LIB_LIST" ] || { echo "ERROR: $LIB_LIST not found. Run 01 first."; exit 1; }

OUTPUT_DIR=$NAS_AGG/output
mkdir -p "$OUTPUT_DIR" "$PROJ/logs"

export PATH="${CONDA_ENVS}/bioinfo/bin:$SCAFE_DIR/resources/bin/paraclu:$SCAFE_DIR/resources/bin/bedGraphToBigWig:$SCAFE_DIR/resources/bin/bigWigAverageOverBed:$PATH"
export R_LIBS_USER="${HOME}/R/libs"

echo "[$(date)] === Integrated SCAFE aggregate workflow (hg38, native) ===" | tee "$LOG"
echo "[$(date)] RUN_TAG: $RUN_TAG" | tee -a "$LOG"
echo "[$(date)] LIB_LIST: $LIB_LIST  ($(wc -l < "$LIB_LIST") samples)" | tee -a "$LOG"
echo "[$(date)] OUTPUT_DIR: $OUTPUT_DIR" | tee -a "$LOG"
echo "[$(date)] Rscript: $(Rscript --version 2>&1 | head -1)" | tee -a "$LOG"
echo "[$(date)] R_LIBS_USER: $R_LIBS_USER" | tee -a "$LOG"
echo "[$(date)] Memory available: $(free -h | awk '/Mem:/{print $7}')" | tee -a "$LOG"
echo "[$(date)] NAS free: $(df -h ${NAS} | awk 'NR==2{print $4}')" | tee -a "$LOG"

echo "[$(date)] R packages:" | tee -a "$LOG"
Rscript -e 'cat(sapply(c("ROCR","PRROC","caret","e1071","ggplot2","scales","reshape2"), function(p) if (requireNamespace(p, quietly=TRUE)) "OK" else "MISSING"), sep=" ")' 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "[$(date)] Starting SCAFE aggregate workflow (max_thread=1, deadlock-safe)..." | tee -a "$LOG"

perl "$SCAFE_DIR/scripts/scafe.workflow.cm.aggregate" \
    --lib_list_path "$LIB_LIST" \
    --genome hg38.gencode_v32 \
    --run_tag "$RUN_TAG" \
    --run_outDir "$OUTPUT_DIR" \
    --max_thread 1 \
    --overwrite no \
    2>&1 | tee -a "$LOG"

echo "[$(date)] SCAFE aggregate completed" | tee -a "$LOG"
echo "[$(date)] Output size: $(du -sh "$OUTPUT_DIR" | cut -f1)" | tee -a "$LOG"

RUN_DIR="$OUTPUT_DIR/$RUN_TAG"
echo "[$(date)] CRE coord bed: $RUN_DIR/filter/$RUN_TAG/bed/${RUN_TAG}.CRE.coord.bed.gz" | tee -a "$LOG"
echo "[$(date)] CRE annot bed: $RUN_DIR/annotate/$RUN_TAG/bed/${RUN_TAG}.CRE.annot.bed.gz" | tee -a "$LOG"
echo "[$(date)] directionality: $RUN_DIR/directionality/$RUN_TAG/log/${RUN_TAG}.directionality.log.tsv.gz" | tee -a "$LOG"
)

step5() (
set -euo pipefail

TS=$(date +%Y%m%d_%H%M%S)
PROJ=${HOME}/projects/integrated_aida_synapse
NAS_BASE=${DATA_DIR}/integrated_aida_synapse_hg38_440
SCAFE_DIR=${SCAFE_DIR}
RUN_TAG=integrated_aida_synapse_hg38_440

CRE_BED=$NAS_BASE/output/annotate/$RUN_TAG/bed/${RUN_TAG}.CRE.coord.bed.gz
LIB_LIST=${DATA_DIR}/integrated_aida_synapse_hg38_440/lib_list_latest.tsv
OUT_BASE=$NAS_BASE/output/sc_count
LOG=$PROJ/logs/03_sc_count_${TS}.log
PARALLEL=${PARALLEL:-6}

mkdir -p "$OUT_BASE" "$PROJ/logs"

export PATH="${CONDA_ENVS}/bioinfo/bin:$SCAFE_DIR/resources/bin/paraclu:$SCAFE_DIR/resources/bin/bedGraphToBigWig:$SCAFE_DIR/resources/bin/bigWigAverageOverBed:$PATH"
export R_LIBS_USER="${HOME}/R/libs"

echo "[$(date)] === sc.count recount on unified 329,471 tCRE ===" | tee "$LOG"
echo "[$(date)] CRE_BED: $CRE_BED" | tee -a "$LOG"
echo "[$(date)] OUT_BASE: $OUT_BASE" | tee -a "$LOG"
echo "[$(date)] PARALLEL: $PARALLEL" | tee -a "$LOG"
echo "[$(date)] LIB_LIST: $LIB_LIST ($(wc -l < "$LIB_LIST") samples)" | tee -a "$LOG"

run_one() {
  local sample="$1"
  local ctss="$2"
  local bc="$3"
  local outdir="$OUT_BASE/$sample"

  if [ -f "$outdir/$sample/matrix/matrix.mtx" ]; then
    echo "[$(date)] SKIP $sample (already done)"
    return 0
  fi

  mkdir -p "$outdir"
  if perl "$SCAFE_DIR/scripts/scafe.tool.sc.count" \
      --genome=hg38.gencode_v32 \
      --countRegion_bed_path="$CRE_BED" \
      --ctss_bed_path="$ctss" \
      --cellBarcode_list_path="$bc" \
      --outputPrefix="$sample" \
      --outDir="$outdir" \
      --overwrite=yes > "$outdir/run.log" 2>&1; then
    echo "[$(date)] OK $sample"
  else
    echo "[$(date)] FAIL $sample (see $outdir/run.log)"
    return 1
  fi
}
export -f run_one
export SCAFE_DIR CRE_BED OUT_BASE

TRIPLET=$PROJ/results/sc_count_triplet_${TS}.tsv
> "$TRIPLET"
n_total=0; n_skip=0
while IFS=$'\t' read -r sample collapse_path ung_path; do
  if [[ "$collapse_path" == */aida_per_sample_bed/* ]]; then
    base=${DATA_DIR}/lnc_project_AIDA_dataset/aida_per_cell_ctss/$sample
    ctss="$base/bam_to_ctss/$sample/bed/${sample}.CB.ctss.bed.gz"
    bc="$base/count/$sample/matrix/barcodes.tsv"
  else
    bed_dir=$(dirname "$collapse_path")
    ctss="$bed_dir/${sample}.CB.ctss.bed.gz"
    sample_root=$(dirname $(dirname $(dirname "$bed_dir")))
    bc="$sample_root/count/${sample}/matrix/barcodes.tsv"
  fi
  if [ ! -f "$ctss" ]; then echo "[$(date)] MISS ctss $sample: $ctss" | tee -a "$LOG"; continue; fi
  if [ ! -f "$bc" ]; then echo "[$(date)] MISS bc $sample: $bc" | tee -a "$LOG"; continue; fi
  if [ -f "$OUT_BASE/$sample/$sample/matrix/matrix.mtx" ]; then n_skip=$((n_skip+1)); continue; fi
  printf '%s\t%s\t%s\n' "$sample" "$ctss" "$bc" >> "$TRIPLET"
  n_total=$((n_total+1))
done < "$LIB_LIST"

echo "[$(date)] To run: $n_total, already done: $n_skip" | tee -a "$LOG"
echo "[$(date)] Triplet: $TRIPLET" | tee -a "$LOG"

if [ "$n_total" -eq 0 ]; then
  echo "[$(date)] Nothing to do" | tee -a "$LOG"
  exit 0
fi

echo "[$(date)] Starting parallel sc.count (P=$PARALLEL)..." | tee -a "$LOG"
< "$TRIPLET" xargs -P "$PARALLEL" -L 1 -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r s c b <<< "$@"; run_one "$s" "$c" "$b"' _ {} 2>&1 | tee -a "$LOG"

echo "[$(date)] === DONE ===" | tee -a "$LOG"
echo "[$(date)] Results: $OUT_BASE" | tee -a "$LOG"
echo "[$(date)] sample count with matrix.mtx:" | tee -a "$LOG"
find "$OUT_BASE" -name matrix.mtx 2>/dev/null | wc -l | tee -a "$LOG"
)

step1
step2
step3
step4
step5
