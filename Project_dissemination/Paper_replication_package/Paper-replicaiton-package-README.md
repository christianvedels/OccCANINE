

## Preparation
Data is structured in two folders:
- Data/Raw_data
- Data/Intermediate_data

Raw data is generally based on data from other research projects, that we are not allowed to redistribute. It is processed (cleaned and standardized) in 'Data_cleaning_scripts/*.R'.

## Python scripts i/o table
All found in Project_dissemination/Paper_replication_package/Model_eval_python.
You can run all of these by running `paper_results.py`. All results are intermediate to be processed by R scripts (described below). For quality of life, each script automatically checks if the relevant intermediate data has already been generated and skips if this is the case. To do a *proper* run, delete all files from '[PRP]/Data/Intermediate_data'

| Script | Description | Input | Output|
| :----  | :---------- | :---- | ----- |
| `threshold_tuning.py` | Runs threshold tuning. NOTE: Run first - needed for other scripts | [PRP]/Data/Raw_data/Validation_data1/*.csv | [PRP]/Data/Intermediate\_data/threshold\_tuning\_{prediction_type}\_langknown.csv; [PRP]/Data/Intermediate\_data/threshold\_tuning_{prediction_type}\_langunk.csv; [PRP}/Data/Intermediate\_data/thr\_tuning\_by\_lang/*.csv; [PRP]/Data/Intermediate\_data/thresholds\_by\_lang.json |
| `predict_test.py` | Runs eval on test based on optimal threshold | Data/Test_data/\*.csv | [PRP]/Data/Intermediate\_data/big_files/obs_test_performance_{prediction_type} |
| `predict_ood.py` | Runs prediction and evaluation on ood data (only *greedy*) | [PRP]/Data/Raw_data/OOD_data | [PRP]/Data/Intermediate_data/big_files/predictions_ood/*.csv |
| `predict_testbylang.py` | Runs eval for every single language | Data/Test_data/*.csv | [PRP]/Data/Intermediate\_data/test\_performance/lang/*.csv | 
| `predict_testby_source.py` | Runs evaluation for each source | Data/Test_data/*.csv | [PRP]/Data/Intermediate\_data/test\_performance/source/*.csv | 
| `embeddings.py` | Predicts the encoding ($n\times768$ pooler encoding) and runs t-sne on this | Data/Test_data/*.csv | [PRP]/Data/Intermediate\_data/big_files/embeddings_test.csv; [PRP]/Data/Intermediate\_data/big_files/tsne_results.csv; [PRP]/Data/Intermediate\_data/big_files/tsne_results_3d.csv | 
| `other_systems_eval.py` | Tests models finetuned in other systems | Data/models/[model], where model $\in$ [mixer-icem-ft; mixer-isco-ft; mixer-occ1950-ft]; Data/Test_data_other/* | /Data/occicem_performance.csv; /Data/isco68_performance.csv; /Data/occ1950_performance.csv | 
| `agreement_german_sources.py`<br> | Compute agreement between two RA sets of HISCO codes | Data/OOD_data/GE_denazification_RA1.csv;<br>Data/OOD_data/GE_denazification_RA1.csv | [PRP]/Data/Intermediate_data/big_files/predictions_ood/predictions_agreement_german_sources.csv |



Notes: [PRP]: Paper replication package directory






| `finetune.py` | Performs finetuning experiments | Data/Training\_data/GE\_\*.csv; Data/OOD_data/GE_denazification.csv | Model_eval_python/OccCANINE_finetuned_ge; Model_eval_python/OccCANINE_finetuned_ge_ra; Data/OOD_data/Predictions_finetuned/* |





