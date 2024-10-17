# Settings for training experiments

# Training

## CANINE initialization

Initialize encoder using `CANINE`, warmup for 3000 steps, no decoder dropout
```
SET CUDA_VISIBLE_DEVICES=0
python train_v2.py --save-dir Z:/faellesmappe/tsdj/hisco/v2-OCC1950/canine-init-s2s --log-interval 50 --eval-interval 15000 --save-interval 5000 --warmup-steps 3000 --formatter occ1950 --target-col-naming occ1950 --train-data Z:/faellesmappe/tsdj/hisco/data/Training_data_other/EN_OCC1950_IPUMS_US_train.csv --val-data Z:/faellesmappe/tsdj/hisco/data/Validation_data1_other/EN_OCC1950_IPUMS_US_val1.csv --num-epochs 16 --log-wandb --wandb-project-name occ-1950
```

## OccCANINE-seq2seq (baseline) initialization

Initialize *encoder* using `OccCANINE` (s2s baseline model), warmup for 3000 steps, no decoder dropout
```
SET CUDA_VISIBLE_DEVICES=1
python train_v2.py --save-dir Z:/faellesmappe/tsdj/hisco/v2-OCC1950/occ-canine-init-s2s --log-interval 50 --eval-interval 15000 --save-interval 5000 --warmup-steps 3000 --formatter occ1950 --target-col-naming occ1950 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --train-data Z:/faellesmappe/tsdj/hisco/data/Training_data_other/EN_OCC1950_IPUMS_US_train.csv --val-data Z:/faellesmappe/tsdj/hisco/data/Validation_data1_other/EN_OCC1950_IPUMS_US_val1.csv --num-epochs 16 --log-wandb --wandb-project-name occ-1950
```

## Mixer with CANINE initialization

Initialize encoder using `CANINE`, warmup for 3000 steps, no decoder dropout
```
set CUDA_VISIBLE_DEVICES=2
python train_mixer.py --save-dir Z:/faellesmappe/tsdj/hisco/v2-OCC1950/canine-init-mixer --log-interval 50 --eval-interval 15000 --save-interval 5000 --warmup-steps 3000 --seq2seq-weight 0.1 --formatter occ1950 --target-col-naming occ1950 --train-data Z:/faellesmappe/tsdj/hisco/data/Training_data_other/EN_OCC1950_IPUMS_US_train.csv --val-data Z:/faellesmappe/tsdj/hisco/data/Validation_data1_other/EN_OCC1950_IPUMS_US_val1.csv --num-epochs 16 --log-wandb --wandb-project-name occ-1950
```

## Mixer with OccCANINE-seq2seq (baseline) initialization

Initialize *encoder* using `OccCANINE` (s2s baseline model), warmup for 3000 steps, no decoder dropout
```
set CUDA_VISIBLE_DEVICES=3
python train_mixer.py --save-dir Z:/faellesmappe/tsdj/hisco/v2-OCC1950/occ-canine-init-mixer --log-interval 50 --eval-interval 15000 --save-interval 5000 --warmup-steps 3000 --seq2seq-weight 0.1 --formatter occ1950 --target-col-naming occ1950 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --train-data Z:/faellesmappe/tsdj/hisco/data/Training_data_other/EN_OCC1950_IPUMS_US_train.csv --val-data Z:/faellesmappe/tsdj/hisco/data/Validation_data1_other/EN_OCC1950_IPUMS_US_val1.csv --num-epochs 16 --log-wandb --wandb-project-name occ-1950
```

# Evaluation

## CANINE initialization

```
python eval_s2s.py --checkpoint Z:/faellesmappe/tsdj/hisco/v2-OCC1950/canine-init/last.bin --fn-out Z:/faellesmappe/tsdj/hisco/v2-OCC1950/canine-init/val-1-s2s-canine-init-s=10000.csv --formatter occ1950 --target-col-naming occ1950 --val-data Z:/faellesmappe/tsdj/hisco/data/Validation_data1_other/EN_OCC1950_IPUMS_US_val1.csv
```
