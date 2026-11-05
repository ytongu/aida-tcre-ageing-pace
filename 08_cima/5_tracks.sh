#!/bin/bash
set -u
TB=${CONDA_ENVS}/bioinfo/bin/tabix
W=${PAPER_DIR}/results/fig3g_canonical_tertile/trackwin
REG1="chr1:207449328-207459329"   # CR2 サミット 207454328 ±5kb
REG2="chr11:66312043-66322044"    # CD248 サミット 66317043 ±5kb
AR=${DATA_DIR}/lnc_project_AIDA_dataset/aida_per_cell_ctss
SR=${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/03_scafe
: > $W/aida_windows.tsv
for d in $AR/*/; do
  s=$(basename $d)
  f=$d/bam_to_ctss/$s/bed/$s.CB.ctss.bed.gz
  [ -f "$f" ] || continue
  $TB "$f" $REG1 $REG2 2>/dev/null | awk -v s=$s 'BEGIN{OFS="\t"}{print s,$1,$2,$4,$5,$6}' >> $W/aida_windows.tsv
done
echo "AIDA done: $(wc -l < $W/aida_windows.tsv) 行"
: > $W/terekhova_windows.tsv
for d in $SR/*/; do
  s=$(basename $d)
  f=$d/bam_to_ctss/$s/bed/$s.CB.ctss.bed.gz
  [ -f "$f" ] || continue
  $TB "$f" $REG1 $REG2 2>/dev/null | awk -v s=$s 'BEGIN{OFS="\t"}{print s,$1,$2,$4,$5,$6}' >> $W/terekhova_windows.tsv
done
echo "Terekhova done: $(wc -l < $W/terekhova_windows.tsv) 行"
