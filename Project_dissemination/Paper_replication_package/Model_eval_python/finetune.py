# Testing finetuning in German
from histocc import OccCANINE
import pandas as pd
import os
from histocc.eval_metrics import EvalEngine


def main():
    # Load model
    mod = OccCANINE()

    # Load data
    data1 = pd.read_csv("Data/Training_data/GE_ipums_train.csv", dtype=str)
    data2 = pd.read_csv("Data/Training_data/GE_occupational_census_train.csv", dtype=str)
    data3 = pd.read_csv("Data/Training_data/GE_occupations1939_train.csv", dtype=str)
    data4 = pd.read_csv("Data/Training_data/GE_Selgert_Gottlich_train.csv", dtype=str)
    
    # Concatenate data
    data_f = pd.concat([data1, data2, data3, data4], ignore_index=True)

    # Perform finetuning
    # Only if finetuned version is not already available
    if not os.path.exists("Project_dissemination/Paper_replication_package/Model_eval_python/OccCANINE_finetuned_ge/last.bin"):
        print("Finetuning OccCANINE model on German data...")
        mod.finetune(
            dataset=data_f,
            # dataset = "Data/Training_data/GE_ipums_train.csv",
            save_path="Project_dissemination/Paper_replication_package/Model_eval_python/OccCANINE_finetuned_ge",
            input_col="occ1",
            language="ge",
            language_col=None,
            target_cols = ["hisco_1", "hisco_2", "hisco_3", "hisco_4", "hisco_5"],
            drop_bad_labels=True,
            allow_codes_shorter_than_block_size=True,
            share_val=0.05,
            num_epochs=3,
            warmup_steps=500,
            freeze_encoder=True,
            eval_interval=5,
            log_interval=1
        )

    # Perform finetuning on the few RA observations
    if not os.path.exists("Project_dissemination/Paper_replication_package/Model_eval_python/OccCANINE_finetuned_ge_ra/last.bin"):
        print("Finetuning OccCANINE model on German RA data...")
        mod.finetune(
            dataset=pd.read_csv('Data/OOD_data/GE_denazification.csv', dtype=str),
            save_path="Project_dissemination/Paper_replication_package/Model_eval_python/OccCANINE_finetuned_ge_ra",
            input_col="occ1",
            language="ge",
            language_col=None,
            target_cols = ["hisco_1", "hisco_2", "hisco_3", "hisco_4", "hisco_5"],
            drop_bad_labels=True,
            allow_codes_shorter_than_block_size=True,
            share_val=0.1,
            num_epochs=50,
            warmup_steps=500,
            freeze_encoder=False,
            eval_interval=1,
            log_interval=1,
            save_interval=5
        )

    # Load finetuned model
    mod_finetuned = OccCANINE(
        "Project_dissemination/Paper_replication_package/Model_eval_python/OccCANINE_finetuned_ge/last.bin",
        hf = False
    )

    mod_finetuned_ra = OccCANINE(
        "Project_dissemination/Paper_replication_package/Model_eval_python/OccCANINE_finetuned_ge_ra/last.bin",
        hf = False
    )

    files = ['Data/OOD_data/GE_denazification.csv', 'Data/OOD_data/GE_denazification_RA1.csv', 'Data/OOD_data/GE_denazification_RA2.csv']
    
    for f in files:
        fname = f'Data/OOD_data/Predictions_finetuned/predictions_{f.split("/")[-1]}'
        fname_finetuned = f'Data/OOD_data/Predictions_finetuned/predictions__finetuned{f.split("/")[-1]}'
        fname_finetuned_ra = f'Data/OOD_data/Predictions_finetuned/predictions__finetuned_ra{f.split("/")[-1]}'

        data_f = pd.read_csv(f, dtype=str)
        print(f'------> Evaluating {f}')

        res = mod(data_f.occ1.tolist(), lang = f[0:2].lower(), deduplicate = True)
        res_finetuned = mod_finetuned(data_f.occ1.tolist(), lang = f[0:2].lower(), deduplicate = True)
        res_finetuned_ra = mod_finetuned_ra(data_f.occ1.tolist(), lang = f[0:2].lower(), deduplicate = True)

        eval_engine = EvalEngine(mod, ground_truth = data_f, predicitons = res, pred_col = "hisco_")
        eval_engine_finetuned = EvalEngine(mod_finetuned, ground_truth = data_f, predicitons = res_finetuned, pred_col = "hisco_")
        eval_engine_finetuned_ra = EvalEngine(mod_finetuned_ra, ground_truth = data_f, predicitons = res_finetuned_ra, pred_col = "hisco_")

        res[f"acc"] = eval_engine.accuracy(return_per_obs = True)
        res["precision"] = eval_engine.precision(return_per_obs = True)
        res["recall"] = eval_engine.recall(return_per_obs = True)
        res["f1"] = eval_engine.f1(return_per_obs = True)

        res_finetuned[f"acc"] = eval_engine_finetuned.accuracy(return_per_obs = True)
        res_finetuned["precision"] = eval_engine_finetuned.precision(return_per_obs = True)
        res_finetuned["recall"] = eval_engine_finetuned.recall(return_per_obs = True)
        res_finetuned["f1"] = eval_engine_finetuned.f1(return_per_obs = True)

        res_finetuned_ra[f"acc"] = eval_engine_finetuned_ra.accuracy(return_per_obs = True)
        res_finetuned_ra["precision"] = eval_engine_finetuned_ra.precision(return_per_obs = True)
        res_finetuned_ra["recall"] = eval_engine_finetuned_ra.recall(return_per_obs = True)
        res_finetuned_ra["f1"] = eval_engine_finetuned_ra.f1(return_per_obs = True)
        
        # Add in rowid
        res["rowid"] = data_f.RowID
        res_finetuned["rowid"] = data_f.RowID

        # Print
        print(f"    Acc: {eval_engine.accuracy()}")
        print(f"    Precision: {eval_engine.precision()}")
        print(f"    Recall: {eval_engine.recall()}")
        print(f"    F1: {eval_engine.f1()}")

        print(f"    Finetuned Acc: {eval_engine_finetuned.accuracy()}")
        print(f"    Finetuned Precision: {eval_engine_finetuned.precision()}")
        print(f"    Finetuned Recall: {eval_engine_finetuned.recall()}")
        print(f"    Finetuned F1: {eval_engine_finetuned.f1()}")

        print(f"    Finetuned RA Acc: {eval_engine_finetuned_ra.accuracy()}")
        print(f"    Finetuned RA Precision: {eval_engine_finetuned_ra.precision()}")
        print(f"    Finetuned RA Recall: {eval_engine_finetuned_ra.recall()}")
        print(f"    Finetuned RA F1: {eval_engine_finetuned_ra.f1()}")

        # Eval digit by digit
        for digits in range(1, 6): 
            print(f"Digits: {digits}")
            eval_engine = EvalEngine(mod, ground_truth = data_f, predicitons = res, pred_col = "hisco_", digits=digits)
            res[f"acc_{digits}"] = eval_engine.accuracy(return_per_obs = True)
            res[f"precision_{digits}"] = eval_engine.precision(return_per_obs = True)
            res[f"recall_{digits}"] = eval_engine.recall(return_per_obs = True)
            res[f"f1_{digits}"] = eval_engine.f1(return_per_obs = True)

            eval_engine_finetuned = EvalEngine(mod_finetuned, ground_truth = data_f, predicitons = res_finetuned, pred_col = "hisco_", digits=digits)
            res_finetuned[f"acc_{digits}"] = eval_engine_finetuned.accuracy(return_per_obs = True)
            res_finetuned[f"precision_{digits}"] = eval_engine_finetuned.precision(return_per_obs = True)
            res_finetuned[f"recall_{digits}"] = eval_engine_finetuned.recall(return_per_obs = True)
            res_finetuned[f"f1_{digits}"] = eval_engine_finetuned.f1(return_per_obs = True)

            eval_engine_finetuned_ra = EvalEngine(mod_finetuned_ra, ground_truth = data_f, predicitons = res_finetuned_ra, pred_col = "hisco_", digits=digits)
            res_finetuned_ra[f"acc_{digits}"] = eval_engine_finetuned_ra.accuracy(return_per_obs = True)
            res_finetuned_ra[f"precision_{digits}"] = eval_engine_finetuned_ra.precision(return_per_obs = True)
            res_finetuned_ra[f"recall_{digits}"] = eval_engine_finetuned_ra.recall(return_per_obs = True)
            res_finetuned_ra[f"f1_{digits}"] = eval_engine_finetuned_ra.f1(return_per_obs = True)

            # Print
            print(f"    Acc (digits: {digits}): {eval_engine.accuracy()}")
            print(f"    Precision (digits: {digits}): {eval_engine.precision()}")
            print(f"    Recall (digits: {digits}): {eval_engine.recall()}")
            print(f"    F1 (digits: {digits}): {eval_engine.f1()}")

            print(f"    Finetuned Acc (digits: {digits}): {eval_engine_finetuned.accuracy()}")
            print(f"    Finetuned Precision (digits: {digits}): {eval_engine_finetuned.precision()}")
            print(f"    Finetuned Recall (digits: {digits}): {eval_engine_finetuned.recall()}")
            print(f"    Finetuned F1 (digits: {digits}): {eval_engine_finetuned.f1()}")

            print(f"    Finetuned RA Acc (digits: {digits}): {eval_engine_finetuned_ra.accuracy()}")
            print(f"    Finetuned RA Precision (digits: {digits}): {eval_engine_finetuned_ra.precision()}")
            print(f"    Finetuned RA Recall (digits: {digits}): {eval_engine_finetuned_ra.recall()}")
            print(f"    Finetuned RA F1 (digits: {digits}): {eval_engine_finetuned_ra.f1()}")
            
        # Save predictions
        res.to_csv(fname)
        res_finetuned.to_csv(fname_finetuned)
        res_finetuned_ra.to_csv(fname_finetuned_ra)

    



    