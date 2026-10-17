# Manual check of DWI and UK Bankruptcies (OOD)

## What this folder is

- `OOD_manual_check_sample.csv`: 1,000 unique occupational strings to label. 500 from each source, simple random sample with seed 20 (`set.seed(20)` before each source's draw). Semicolon-separated.
- `OOD_manual_check_sampling_frame.csv`: all unique strings sampled from, with a `sampled` flag.
- `README.txt`: seed, R version, MD5 checksums of the input prediction files, and the frame sizes (DWI 2,348 unique strings; UK Bankruptcies 4,948).
- `OOD_manual_check_labelled.csv`: **does not exist yet.** It is the labelled copy of the sample.

Everything is produced by `Model_eval_R/005_OOD_performance.R`. On each run it redraws the sample and stops with an error if the saved sample differs. It then stops with an error if `OOD_manual_check_labelled.csv` is missing. Otherwise it scores the labels and adds a third table (one row per source) to `Tables/ood_performance.txt`.

## Columns in the sample

| Column | Meaning |
|---|---|
| `id` | Row id. Must not change. |
| `source` | `DWI` or `UK Bankruptcies` |
| `n` | Number of observations with this string |
| `occupation` | The string, without the `en[SEP]` prefix. Must not change. |
| `occc_hisco_1`, `occc_desc_1`, `occc_hisco_2`, `occc_desc_2` | OccCANINE prediction (from `big_files/predictions_ood/`, confirmed identical to the current model on 2026-10-15) |
| `source_hisco`, `source_desc` | Most frequent code the source data gives the string |
| `source_hisco_all` | All source codes with their share of rows, e.g. `41025 (0.7); -1 (0.3)` |
| `correct_hisco_1` … `correct_hisco_5` | **To fill in.** 5-digit HISCO codes; `-1` = no occupation. `correct_hisco_1` is required. |
| `comment` | **To fill in**, optional |

## Labelling rules enforced by 005

- Same rows, `id` and `occupation` as the sample.
- `correct_hisco_1` filled in for every row.
- All codes must exist in `Data/Key.csv`. Leading zeros dropped by Excel are restored automatically.
- Note that some Bankruptcy *source* codes are not in the key (e.g. 43010, 81300, 95110), so don't copy those into `correct_hisco_*`.

## How it is scored

- OccCANINE: `acc()` from `000_Functions.R` (5 digits), comparing the set of predicted codes with the set of `correct_hisco_*`.
- Source: the share-weighted `acc()` over the codes in `source_hisco_all`.
- Each reported both per string and weighted by `n`.

