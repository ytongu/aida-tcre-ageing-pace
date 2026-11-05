# aida-tcre-ageing-pace

## Description

This repository contains the code used to generate the results of the following manuscript:

Tongu Y, Kasahara T, Zhang B, Gladyshev VN. Immune cells whose transcriptomic age exceeds chronological age show higher enhancer and lower identity-promoter transcription initiation.

## Contents

| Directory | Analysis |
|---|---|
| `01_tcre_catalogue` | Cell Ranger and SCAFE on 440 5′ scRNA-seq libraries (AIDA, Terekhova); tCRE catalogue and counts |
| `02_cell_labels` | Cell-type labels, including CD4 naive/memory |
| `03_transcriptomic_age` | Gene metacells, transcriptomic age (tAge) and ageing pace |
| `04_pace_association` | tCRE–pace associations in both cohorts and the replicated set |
| `05_validation` | Raw CTSS, sequencing depth, clock circularity, reproducibility, single-cell checks |
| `06_longitudinal` | Within-person analyses |
| `07_synchrony` | Ageing pace across cell types |
| `08_cima` | CIMA scATAC-seq cohort |

Scripts within each directory are numbered in the order they were run. Paths are written with the variables `${DATA_DIR}`, `${PROJECT_DIR}`, `${PAPER_DIR}`, `${PIPELINE_DIR}`, `${SCAFE_DIR}`, `${CONDA_ENVS}`, `${SCRATCH}` and `${HPC_*}`, which should be set to local locations.

## Dependencies

- Cell Ranger 9.0.1 (refdata-gex-GRCh38-2024-A)
- SCAFE v1.0.0 (https://github.com/chung-lab/SCAFE)
- R tAge 1.1.0
- Python and R environments: `env/`

## Data

AIDA (CZ CELLxGENE), Terekhova et al. (Synapse syn49637038) and CIMA are publicly available. No individual-level data are included.

## Installation

```
git clone https://github.com/ytongu/aida-tcre-ageing-pace.git
```

## License

MIT
