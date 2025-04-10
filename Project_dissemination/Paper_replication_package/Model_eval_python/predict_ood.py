
from histocc import OccCANINE
from histocc.eval_metrics import EvalEngine
import pandas as pd
import os

mod = OccCANINE()

# list files
files = os.listdir('Data/OOD_data')

for f in files:
    if f == 'Predictions':
        continue
    print(f'------> Predicting {f}')
    data_f = pd.read_csv(f'Data/OOD_data/{f}')

    res = mod(data_f.occ1.tolist(), lang = f[0:2].lower())

    eval_engine = EvalEngine(mod, ground_truth = data_f, predicitons = res, pred_col = "hisco_")
    res[f"acc"] = eval_engine.accuracy(return_per_obs = True)
    res["precision"] = eval_engine.precision(return_per_obs = True)
    res["recall"] = eval_engine.recall(return_per_obs = True)
    res["f1"] = eval_engine.f1(return_per_obs = True)

    # Add in rowid
    res["rowid"] = data_f.RowID

    # Print
    print(f"    Acc: {eval_engine.accuracy()}")
    print(f"    Precision: {eval_engine.precision()}")
    print(f"    Recall: {eval_engine.recall()}")
    print(f"    F1: {eval_engine.f1()}")

    # Eval digit by digit
    for digits in range(1, 6): 
        print(f"Digits: {digits}")
        eval_engine = EvalEngine(mod, ground_truth = data_f, predicitons = res, pred_col = "hisco_", digits=digits)
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
    res.to_csv(f'Data/OOD_data/Predictions/predictions_{f}')
