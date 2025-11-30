
from histocc import OccCANINE
from histocc.eval_metrics import EvalEngine, TopKEvalEngine
import pandas as pd
import os

def main(data_path="OOD_data", K = 20):
    mod = OccCANINE()

    # list files
    files = os.listdir(data_path)

    for f in files:
        if f == 'Predictions':
            continue

        if f == 'Predictions_finetuned':
            continue

        fname = f'Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_ood_top_k/predictions_{f}'

        if os.path.exists(fname):
            print(f"Skipping {f} as predictions already exist.")
            continue

        # Check if dir exists, if not create it
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        print(f'------> Predicting {f}')
        data_f = pd.read_csv(f'Data/OOD_data/{f}')

        # If no HISCO columns, skip
        hisco_cols = [col for col in data_f.columns if col.startswith('hisco_')]
        if len(hisco_cols) == 0:
            print(f"Warning: No column starting with 'hisco_' found in {f}. Skipping file.")
            continue

        res = mod(data_f.occ1.tolist(), lang = f[0:2].lower(), deduplicate = True, prediction_type='greedy-topk', k_pred = K)

        # Add in rowid to match predictions with ground truth
        # Each original observation will have k_pred rows in the result
        res["rowid"] = data_f.RowID.repeat(K).values  # Repeat each rowid K times
        
        # Check if 'n' in data_f
        if "n" in data_f.columns:
            res["n"] = data_f.n.repeat(K).values
        else:
            res["n"] = 1  # Default to 1 if 'n' is not present

        # Use TopKEvalEngine for evaluation
        eval_engine = TopKEvalEngine(mod, ground_truth=data_f, predicitons=res, pred_col="hisco_", group_col="rowid")
        
        # Get per-observation metrics as flat lists matching prediction order
        res["acc"] = eval_engine.accuracy(return_per_obs=True)
        res["precision"] = eval_engine.precision(return_per_obs=True)
        res["recall"] = eval_engine.recall(return_per_obs=True)
        res["f1"] = eval_engine.f1(return_per_obs=True)

        # Original HISCO codes - add to each top-k row
        for i in range(1, 6):
            col = f"hisco_{i}"
            if col in data_f.columns:
                # Create a mapping from rowid to original hisco value
                hisco_mapping = data_f.set_index("RowID")[col]
                res[f"{col}_original"] = res["rowid"].map(hisco_mapping)

        # Check if HISCO codes are present in input data
        if not any(col.startswith("hisco_") for col in data_f.columns):
            print(f"Warning: No column starting with 'hisco_' found in {f}. Check results manually.")
            # Save predictions
            res.to_csv(fname)
            continue

        # Print aggregate metrics (dictionaries with rank -> metric)
        print("\n=== Aggregate Metrics by Rank ===")
        acc_dict = eval_engine.accuracy()
        prec_dict = eval_engine.precision()
        rec_dict = eval_engine.recall()
        f1_dict = eval_engine.f1()
        
        for rank in sorted(acc_dict.keys()):
            print(f"  Rank {rank}:")
            print(f"    Acc: {acc_dict[rank]:.4f}")
            print(f"    Precision: {prec_dict[rank]:.4f}")
            print(f"    Recall: {rec_dict[rank]:.4f}")
            print(f"    F1: {f1_dict[rank]:.4f}")

        # Eval digit by digit
        for digits in range(1, 6):
            print(f"\n=== Digits: {digits} ===")
            eval_engine_digits = TopKEvalEngine(mod, ground_truth=data_f, predicitons=res, pred_col="hisco_", group_col="rowid", digits=digits)
            
            # Add metrics directly as columns (flat lists)
            res[f"acc_{digits}"] = eval_engine_digits.accuracy(return_per_obs=True)
            res[f"precision_{digits}"] = eval_engine_digits.precision(return_per_obs=True)
            res[f"recall_{digits}"] = eval_engine_digits.recall(return_per_obs=True)
            res[f"f1_{digits}"] = eval_engine_digits.f1(return_per_obs=True)

            # Print aggregate metrics by rank
            acc_dict_d = eval_engine_digits.accuracy()
            prec_dict_d = eval_engine_digits.precision()
            rec_dict_d = eval_engine_digits.recall()
            f1_dict_d = eval_engine_digits.f1()
            
            for rank in sorted(acc_dict_d.keys()):
                print(f"  Rank {rank}:")
                print(f"    Acc: {acc_dict_d[rank]:.4f}")
                print(f"    Precision: {prec_dict_d[rank]:.4f}")
                print(f"    Recall: {rec_dict_d[rank]:.4f}")
                print(f"    F1: {f1_dict_d[rank]:.4f}")

        # Save predictions
        res.to_csv(fname)

if __name__ == "__main__":
    main()

