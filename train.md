# Settings for training experiments

## Data references
```
SET TRAIN_DATA=Z:/faellesmappe/tsdj/hisco/data/Training_data\CA_bcn_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\DK_cedar_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\DK_census_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\DK_orsted_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\EN_ca_ipums_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\EN_loc_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\EN_marr_cert_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\EN_oclack_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\EN_parish_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\EN_patentee_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\EN_PortArthur_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\EN_ship_data_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\EN_uk_ipums_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\EN_us_ipums_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\FR_desc_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\GE_ipums_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\GE_occupational_census_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\GE_occupations1939_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\GE_Selgert_Gottlich_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\HISCO_website_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\HSN_database_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\IS_ipums_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\IT_fm_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\JIW_database_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\NO_ipums_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\SE_cedar_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\SE_chalmers_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\SE_swedpop_train.csv Z:/faellesmappe/tsdj/hisco/data/Training_data\SE_titles_train.csv Z:/faellesmappe/tsdj/hisco/data/Adversarial_data/Adv_data_double_translateFalse.csv Z:/faellesmappe/tsdj/hisco/data/Adversarial_data/Adv_data_double_translateTrue.csv Z:/faellesmappe/tsdj/hisco/data/Adversarial_data/Random_strings.csv Z:/faellesmappe/tsdj/hisco/data/Adversarial_data/Translated_data.csv

SET VAL_DATA=Z:/faellesmappe/tsdj/hisco/data/Validation_data1\CA_bcn_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\DK_cedar_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\DK_census_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\DK_orsted_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\EN_ca_ipums_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\EN_loc_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\EN_marr_cert_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\EN_oclack_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\EN_parish_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\EN_patentee_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\EN_PortArthur_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\EN_ship_data_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\EN_uk_ipums_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\EN_us_ipums_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\FR_desc_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\GE_ipums_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\GE_occupational_census_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\GE_occupations1939_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\GE_Selgert_Gottlich_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\HISCO_website_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\HSN_database_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\IS_ipums_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\IT_fm_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\JIW_database_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\NO_ipums_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\SE_cedar_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\SE_chalmers_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\SE_swedpop_val1.csv Z:/faellesmappe/tsdj/hisco/data/Validation_data1\SE_titles_val1.csv
```

## Baseline

Initialize encoder using `OccCANINE-V1`, warmup for 3000 steps, no decoder dropout
```
SET CUDA_VISIBLE_DEVICES=1
python train_v2.py --save-dir Z:/faellesmappe/tsdj/hisco/v2/baseline --train-data %TRAIN_DATA% --val-data %VAL_DATA% --log-interval 50 --eval-interval 15000 --save-interval 5000 --log-wandb --warmup-steps 3000 --initial-checkpoint occ-canine-v1
```

## Dropout

Baseline, but with 10% decoder dropout
```
SET CUDA_VISIBLE_DEVICES=2
python train_v2.py --save-dir Z:/faellesmappe/tsdj/hisco/v2/dropout --train-data %TRAIN_DATA% --val-data %VAL_DATA% --log-interval 50 --eval-interval 15000 --save-interval 5000 --log-wandb --warmup-steps 3000 --initial-checkpoint occ-canine-v1 --dropout 0.1
```

## CANINE initialization

Baseline, but initializing encoder from CANINE rather than OccCANINE
```
SET CUDA_VISIBLE_DEVICES=3
python train_v2.py --save-dir Z:/faellesmappe/tsdj/hisco/v2/canine-init --train-data %TRAIN_DATA% --val-data %VAL_DATA% --log-interval 50 --eval-interval 15000 --save-interval 5000 --log-wandb --warmup-steps 3000
```

## CANINE initialization, dropout

Baseline, but initializing encoder from CANINE rather than OccCANINE and using 10% decoder dropout
```
SET CUDA_VISIBLE_DEVICES=0
python train_v2.py --save-dir Z:/faellesmappe/tsdj/hisco/v2/canine-init-dropout --train-data %TRAIN_DATA% --val-data %VAL_DATA% --log-interval 50 --eval-interval 15000 --save-interval 5000 --log-wandb --warmup-steps 3000 --dropout 0.1
```

# Evaluation

## Baseline

```
SET CUDA_VISIBLE_DEVICES=1
python eval_s2s.py --val-data %VAL_DATA% --checkpoint Z:/faellesmappe/tsdj/hisco/v2/baseline/last.bin --fn-out Z:/faellesmappe/tsdj/hisco/v2/baseline/val-1-s2s-baseline-s=1600000.csv
```

## CANINE initialization

```
SET CUDA_VISIBLE_DEVICES=3
python eval_s2s.py --val-data %VAL_DATA% --checkpoint Z:/faellesmappe/tsdj/hisco/v2/canine-init/last.bin --fn-out Z:/faellesmappe/tsdj/hisco/v2/canine-init/val-1-s2s-canine-init-s=1595000.csv
```
