from histocc import OccCANINE, EvalEngine
import pandas as pd
import glob
import os

def load_data(n_obs=5000, data_path="Data/Test_data/*.csv", lang = None):
    """
    Load data from the given path and sample n_obs rows.
    Args:
        n_obs (int): Number of observations to sample.
        data_path (str): Path to the data files.

    Returns:
        pd.DataFrame: Sampled DataFrame.
    """
    csv_files = glob.glob(data_path)
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    if lang:
        df = df[df["lang"] == lang]

    df = df.sample(n=n_obs, random_state=10) if n_obs < df.shape[0] else df
    df = df.reset_index(drop=True)
    return df

def run_eval(df, mod, prediction_type, thr=0.31, digits=5):
    """
    Run evaluation for a given threshold.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        mod (OccCANINE): The OccCANINE model instance.
        thr (float): Threshold for predictions.
        digits (int): Number of digits for rounding evaluation metrics.

    Returns:
        pd.DataFrame: DataFrame with evaluation results.
    """
    if prediction_type == "full":
        # Downsample to 10k because its slow
        df = df.sample(n=10000, random_state=20) if df.shape[0] > 10000 else df

    fname = f"Project_dissemination/Paper_replication_package/Data/test_performance/obs_test_performance_{prediction_type}.csv"
    if os.path.exists(fname):
        print(f"Skipping {fname} as predictions already exist.")
        return

    preds = mod(
        df["occ1"].tolist(), 
        df["lang"].tolist(), 
        threshold=thr, 
        prediction_type=prediction_type, 
        deduplicate=True
    )

    preds_unk = mod(
        df["occ1"].tolist(), 
        "unk", 
        threshold=thr, 
        prediction_type=prediction_type, 
        deduplicate=True
    )

    eval_engine = EvalEngine(
        mod, 
        df, 
        preds, 
        pred_col='hisco_', 
        digits=digits
    )

    eval_engine_unk = EvalEngine(
        mod, 
        df, 
        preds_unk, 
        pred_col='hisco_', 
        digits=digits
    )

    # Individual level metrics
    preds[f"acc"] = eval_engine.accuracy(return_per_obs = True)
    preds["precision"] = eval_engine.precision(return_per_obs = True)
    preds["recall"] = eval_engine.recall(return_per_obs = True)
    preds["f1"] = eval_engine.f1(return_per_obs = True)
    preds["rowid"] = df.RowID
    preds.to_csv(fname, index=False)
    
    for d in range(1, digits + 1):
        eval_engine.digits = d
        res = pd.DataFrame([{
            "threshold": thr,
            "accuracy": eval_engine.accuracy(),
            "f1": eval_engine.f1(),
            "precision": eval_engine.precision(),
            "recall": eval_engine.recall(),
            "n": df.shape[0],
            "prediction_type": prediction_type,
            "lang": "known"
        }])

        file = f"Project_dissemination/Paper_replication_package/Data/test_performance/test_performance_{prediction_type}_digits_{d}.csv"
        res.to_csv(file, index=False)
        print(f"Results saved to {file}")
        print(res)

        eval_engine_unk.digits = d
        res_unk = pd.DataFrame([{
            "threshold": thr,
            "accuracy": eval_engine_unk.accuracy(),
            "f1": eval_engine_unk.f1(),
            "precision": eval_engine_unk.precision(),
            "recall": eval_engine_unk.recall(),
            "n": df.shape[0],
            "prediction_type": prediction_type,
            "lang": "unk"
        }])
        file_unk = f"Project_dissemination/Paper_replication_package/Data/test_performance/test_performance_{prediction_type}_unk_digits_{d}.csv"
        res_unk.to_csv(file_unk, index=False)
        print(f"Results saved to {file_unk}")
        print(res_unk)

def main():
    """
    Main function to load data, get predictions, and evaluate the model.
    """
    # Load the model
    mod = OccCANINE()

    # Load data
    df = load_data(n_obs=100000, data_path="Data/Test_data/*.csv", lang=None)
    
    # Run evaluations for different prediction types
    run_eval(df, mod, prediction_type="flat", thr=0.31, digits=5)
    run_eval(df, mod, prediction_type="greedy", digits=5)
    run_eval(df, mod, prediction_type="full", thr=0.25, digits=5)

if __name__ == "__main__":
    main()