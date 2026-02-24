# OCC1950

## Medium sample, uniques (10_000)

```
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-occ1950-n=10k-ft --target-cols OCC1950_1 OCC1950_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 5689 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_OCC1950_IPUMS_US_n_unq10000_train.csv --batch-size 512 --log-wandb --wandb-project-name occ-canine-ft-v3
```

```
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-occ1950-n=10k-ft-frz-enc --target-cols OCC1950_1 OCC1950_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 5689 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_OCC1950_IPUMS_US_n_unq10000_train.csv --batch-size 512 --freeze-encoder --log-wandb --wandb-project-name occ-canine-ft-v3
```

## Medium sample, random (10_000)

```
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-occ1950-n=10k-r-ft --target-cols OCC1950_1 OCC1950_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 5689 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_OCC1950_IPUMS_US_n10000_train.csv --batch-size 512 --log-wandb --wandb-project-name occ-canine-ft-v3
```

```
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-occ1950-n=10k-r-ft-frz-enc --target-cols OCC1950_1 OCC1950_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 5689 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_OCC1950_IPUMS_US_n10000_train.csv --batch-size 512 --freeze-encoder --log-wandb --wandb-project-name occ-canine-ft-v3
```

## Full sample (18M)

```
set CUDA_VISIBLE_DEVICES=0
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-occ1950-ft --target-cols OCC1950_1 OCC1950_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 16 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_OCC1950_IPUMS_US_train.csv --batch-size 512 --eval-interval 10000 --save-interval 10000 --log-wandb --wandb-project-name occ-canine-ft-v3
```

# PSTI

## Full sample (270_000)

```
set CUDA_VISIBLE_DEVICES=1
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-psti-ft-v4 --target-cols PSTI_1 PSTI_2 PSTI_3 PSTI_4 PSTI_5 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 1050 --block-size 8 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_PSTI_CAMPOP_train.csv --batch-size 512 --save-interval 5000 --use-within-block-sep --drop-bad-labels --log-wandb --wandb-project-name occ-canine-ft-v4
```

We next run Vedel script to generate adversarial data.
Out aim is to create a dataset combining the real data used above with this adversarial data.
Doing so directly is not possible, since there's slight formatting issues, so instead we start a "dummy" run with *only* adversarial data, purely to prepare it.
We can then concatenate this with the data above to create a valid, combined dataset.

```
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-psti-ft-v4-prep-adv-data --target-cols PSTI_1 PSTI_2 PSTI_3 PSTI_4 PSTI_5 --num-epochs 1 --block-size 8 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\pst_adv_merged_results.csv --use-within-block-sep --drop-bad-labels --share-val 0.0001
```

We now manually concatenate this adversarial training data with our real training data.
We create a new folder with contents
1. `key.csv` from our `mixer-psti-ft-v4` run
1. `data_val.csv` from our `mixer-psti-ft-v4` run, for comparable test results and no data leakage
1. `data_train.csv` by combining `data_train.csv` from our `mixer-psti-ft-v4` run with `data_train.csv` from our `mixer-psti-ft-v4-prep-adv-data` run

```python
import pandas as pd

d_real = pd.read_csv(r'Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-psti-ft-v4\data_train.csv')
d_adv = pd.read_csv(r'Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-psti-ft-v4-prep-adv-data\data_train.csv')

m = pd.concat([d_real, d_adv])

m.to_csv(r'Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-psti-ft-v4-adv-ft\data_train.csv', index=False)
```

We now start a run from `mixer-psti-ft-v4-adv-ft` which fine-tunes `mixer-psti-ft-v4` on this combined dataset.

```
set CUDA_VISIBLE_DEVICES=1
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-psti-ft-v4-adv-ft --target-cols PSTI_1 PSTI_2 PSTI_3 PSTI_4 PSTI_5 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-psti-ft-v4\last.bin --num-epochs 113 --block-size 8 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_PSTI_CAMPOP_train.csv --batch-size 512 --save-interval 5000 --use-within-block-sep --drop-bad-labels --log-wandb --wandb-project-name occ-canine-ft-v4
```

# PST2

This is currently handheld to ensure key completeness.
We thus run this script, stop upon data prep completion, manually construct keys, and then restarts.

```
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/pst2/mixer-pst2 --target-cols pst2_1 pst2_2 pst2_3 pst2_4 pst2_5 --warmup-steps 5000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 26 --block-size 8 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\pst2.csv --batch-size 512 --eval-interval 10000 --save-interval 5000 --use-within-block-sep --drop-bad-labels --log-wandb --wandb-project-name pst2
```

## Excluding synthetic "GPT" samples

This is currently handheld to ensure key completeness.
We thus run this script, stop upon data prep completion, manually construct keys, and then restarts.

```
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/pst2/mixer-pst2-no-gpt --target-cols pst2_1 pst2_2 pst2_3 pst2_4 pst2_5 --warmup-steps 5000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 143 --block-size 8 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\pst2_no_gpt.csv --batch-size 512 --eval-interval 10000 --save-interval 5000 --use-within-block-sep --drop-bad-labels --log-wandb --wandb-project-name pst2
```

# ISCO

```
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-isco-ft --target-cols ISCO68A_1 ISCO68A_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 4 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_ISCO68_IPUMS_UK_train.csv --batch-size 512 --eval-interval 50000 --save-interval 10000 --log-wandb --wandb-project-name occ-canine-ft-v3
```

## Medium sample, random (10_000)

```
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-isco-n=10k-r-ft --target-cols ISCO68A_1 ISCO68A_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 5689 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_ISCO68_IPUMS_UK_n10000_train.csv --batch-size 512 --log-wandb --wandb-project-name occ-canine-ft-v3
```

```
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-isco-n=10k-r-ft-frz-enc --target-cols ISCO68A_1 ISCO68A_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 5689 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_ISCO68_IPUMS_UK_n10000_train.csv --batch-size 512 --freeze-encoder --log-wandb --wandb-project-name occ-canine-ft-v3
```

# ICEM

```
set CUDA_VISIBLE_DEVICES=0
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-icem-ft --target-cols OCCICEM_1 OCCICEM_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 4 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_OCCICEM_IPUMS_UK_train.csv --batch-size 512 --eval-interval 10000 --save-interval 10000 --log-wandb --wandb-project-name occ-canine-ft-v3
```

## Medium sample, random (10_000)

```
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-icem-n=10k-r-ft --target-cols OCCICEM_1 OCCICEM_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 5689 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_OCCICEM_IPUMS_UK_n10000_train.csv --batch-size 512 --log-wandb --wandb-project-name occ-canine-ft-v3
```

```
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v3/mixer-icem-n=10k-r-ft-frz-enc --target-cols OCCICEM_1 OCCICEM_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 5689 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_OCCICEM_IPUMS_UK_n10000_train.csv --batch-size 512 --freeze-encoder --log-wandb --wandb-project-name occ-canine-ft-v3
```

# HISCO

```

```

# Model evaluation
For select models

## OCC1950, Full sample (18M)

```
python eval_gp_mixer.py --val-data Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-occ1950-ft\data_val.csv --mapping Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-occ1950-ft\key.csv --checkpoint Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-occ1950-ft\last.bin --target-cols OCC1950_1 OCC1950_2 --block-size 3 --fn-out Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-occ1950-ft\preds_val.csv
```

## PSTI

```
python eval_gp_mixer.py --val-data Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-psti-ft\data_val.csv --mapping Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-psti-ft\key.csv --checkpoint Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-psti-ft\last.bin --target-cols PSTI_1 PSTI_2 PSTI_3 PSTI_4 PSTI_5 --block-size 8 --use-within-block-sep --fn-out Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-psti-ft\preds_val.csv
```

## PST2

```
python eval_gp_mixer.py --val-data Z:\faellesmappe\tsdj\hisco\pst2\mixer-pst2\data_val.csv --mapping Z:\faellesmappe\tsdj\hisco\pst2\mixer-pst2\key.csv --checkpoint Z:\faellesmappe\tsdj\hisco\pst2\mixer-pst2\last.bin --target-cols pst2_1 pst2_2 pst2_3 pst2_4 pst2_5 --block-size 8 --use-within-block-sep --fn-out Z:\faellesmappe\tsdj\hisco\pst2\mixer-pst2\preds_val.csv
```

## ISCO

```
python eval_gp_mixer.py --val-data Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-isco-ft\data_val.csv --mapping Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-isco-ft\key.csv --checkpoint Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-isco-ft\last.bin --target-cols ISCO68A_1 ISCO68A_2 --block-size 3 --fn-out Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-isco-ft\preds_val.csv
```

## ICEM

```
python eval_gp_mixer.py --val-data Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-icem-ft\data_val.csv --mapping Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-icem-ft\key.csv --checkpoint Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-icem-ft\last.bin --target-cols OCCICEM_1 OCCICEM_2 --block-size 3 --fn-out Z:\faellesmappe\tsdj\hisco\ft-tests-v3\mixer-icem-ft\preds_val.csv
```
