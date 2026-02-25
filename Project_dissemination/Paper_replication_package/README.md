# Replication package for the paper "Breaking the HISCO Barrier: Automatic Occupational Standardization with OccCANINE"

This replication package replicates all the evaluation results shown in the paper. For replication of the model training procedure, see the full repository [OccCANINE](https://github.com/christianvedels/OccCANINE).

First you run everything from `Model_eval_python` and then `Model_eval_R`. Everything in python checks whether relevant intermediate data has already been computed and skips those that have. To recompute everything delete content of `Intermediate data`. 

To run everything you can run:
    - `paper_results.py`
    - `Project_dissemination/Paper_replication_package/Model_eval_R/999_main.R`

