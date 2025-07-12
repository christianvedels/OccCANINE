
from histocc import OccCANINE
from histocc.eval_metrics import EvalEngine
import pandas as pd
import os

def main():
    mod = OccCANINE()
    
    fname = f'Data/OOD_data/Predictions/predictions_agreement_german_sources.csv'

    if os.path.exists(fname):
        print(f"Skipping {fname} as predictions already exist.")
        return 0

    data_f1 = pd.read_csv('Data/OOD_data/GE_denazification_RA1.csv')
    res = pd.read_csv('Data/OOD_data/GE_denazification_RA2.csv')

    eval_engine = EvalEngine(mod, ground_truth = data_f1, predicitons = res, pred_col = "hisco_")
    res[f"acc"] = eval_engine.accuracy(return_per_obs = True)
    res["precision"] = eval_engine.precision(return_per_obs = True)
    res["recall"] = eval_engine.recall(return_per_obs = True)
    res["f1"] = eval_engine.f1(return_per_obs = True)

    # Add in rowid
    res["rowid"] = data_f1.RowID

    # Check if 'n' in data_f
    if "n" in data_f1.columns:
        res["n"] = data_f1.n
    else:
        res["n"] = 1  # Default to 1 if 'n' is not present

    # Print
    print(f"    Acc: {eval_engine.accuracy()}")
    print(f"    Precision: {eval_engine.precision()}")
    print(f"    Recall: {eval_engine.recall()}")
    print(f"    F1: {eval_engine.f1()}")

    # Eval digit by digit
    for digits in range(1, 6): 
        print(f"Digits: {digits}")
        eval_engine = EvalEngine(mod, ground_truth = data_f1, predicitons = res, pred_col = "hisco_", digits=digits)
        res[f"acc_{digits}"] = eval_engine.accuracy(return_per_obs = True)
        res[f"precision_{digits}"] = eval_engine.precision(return_per_obs = True)
        res[f"recall_{digits}"] = eval_engine.recall(return_per_obs = True)
        res[f"f1_{digits}"] = eval_engine.f1(return_per_obs = True)

        # Print
        print(f"    Acc (digits: {digits}): {eval_engine.accuracy()}")
        print(f"    Precision (digits: {digits}): {eval_engine.precision()}")
        print(f"    Recall (digits: {digits}): {eval_engine.recall()}")
        print(f"    F1 (digits: {digits}): {eval_engine.f1()}")

    # Save predictions
    res.to_csv(fname)
        
if __name__ == "__main__":
    main()

