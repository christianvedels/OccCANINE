# OCC1950

## Small sample (1000)

## Medium sample (10_000)



## Large sample (100_000)

```
set CUDA_VISIBLE_DEVICES=1
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v2/mixer-occ1950-n=100k-ft --target-cols OCC1950_1 OCC1950_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 569 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_OCC1950_IPUMS_US_n100000_train.csv --batch-size 512 --log-wandb --wandb-project-name occ-canine-ft-v2-tests
```

```
set CUDA_VISIBLE_DEVICES=0
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v2/mixer-occ1950-n=100k-ft-frz-enc --target-cols OCC1950_1 OCC1950_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 569 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_OCC1950_IPUMS_US_n100000_train.csv --batch-size 512 --freeze-encoder --log-wandb --wandb-project-name occ-canine-ft-v2-tests
```

## Full sample (18M)

```
set CUDA_VISIBLE_DEVICES=3
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v2/mixer-occ1950-ft --target-cols OCC1950_1 OCC1950_2 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 16 --block-size 3 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_OCC1950_IPUMS_US_train.csv --batch-size 512 --eval-interval 10000 --save-interval 10000 --log-wandb --wandb-project-name occ-canine-ft-v2-tests
```

# PSTI

## Full sample (270_000)

```
set CUDA_VISIBLE_DEVICES=2
python finetune.py --save-path Z:/faellesmappe/tsdj/hisco/ft-tests-v2/mixer-psti-ft --target-cols PSTI_1 PSTI_2 PSTI_3 PSTI_4 PSTI_5 --warmup-steps 1000 --seq2seq-weight 0.1 --initial-checkpoint Z:\faellesmappe\tsdj\hisco\v2\baseline\last.bin --only-encoder --num-epochs 1050 --block-size 8 --input-col occ1 --language-col lang --dataset Z:\faellesmappe\tsdj\hisco\data\Training_data_other\EN_PSTI_CAMPOP_train.csv --batch-size 512 --save-interval 5000 --use-within-block-sep --drop-bad-labels --log-wandb --wandb-project-name occ-canine-ft-v2-tests
```

# ISCO

# ICEM

# HISCO