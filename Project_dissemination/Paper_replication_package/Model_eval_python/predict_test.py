from histocc import OccCANINE, EvalEngine
import pandas as pd
import glob
import os
import json

def load_data(n_obs=5000, data_path=r"Project_dissemination\Paper_replication_package\Data\Raw_data\Test_data\*.csv", lang = None):
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

def load_optimal_thresholds():
    """
    Load optimal thresholds for flat and full predictions from a JSON file.
    
    Returns:
        dict: Dictionary with optimal thresholds for each language.
    """
    thresholds_path = "Project_dissemination/Paper_replication_package/Data/Intermediate_data/thresholds_by_lang.json"
    
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")
    
    with open(thresholds_path, "r") as f:
        thresholds_by_lang = json.load(f)
    
    return thresholds_by_lang

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
        df = df.sample(n=int(df.shape[0]*0.1), random_state=20)
        mod.batch_size = 16  # Set batch size for full predictions (lower to see steps)

    # Create dir if it does not exist
    os.makedirs("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files", exist_ok=True)
    fname = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/obs_test_performance_{prediction_type}.csv"
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

        # Make directory if it does not exist
        os.makedirs("Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance", exist_ok=True)
        # Save results to CSV
        file = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance/test_performance_{prediction_type}_digits_{d}.csv"
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
        file_unk = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance/test_performance_{prediction_type}_unk_digits_{d}.csv"
        res_unk.to_csv(file_unk, index=False)
        print(f"Results saved to {file_unk}")
        print(res_unk)

def main(toyrun=False):
    """
    Main function to load data, get predictions, and evaluate the model.
    """
    # Load the model
    mod = OccCANINE()

    # Load data
    if toyrun:
        df = load_data(n_obs=1000, lang=None)
    else:
        # Load a larger dataset for full evaluation
        df = load_data(n_obs=1000000000000000, lang=None)

    thr = load_optimal_thresholds()
    
    # Run evaluations for different prediction types
    run_eval(df, mod, prediction_type="flat", thr=thr["overall"]["flat"], digits=5)
    run_eval(df, mod, prediction_type="greedy", thr=99, digits=5)  # Placeholder threshold (not used in greedy)
    run_eval(df, mod, prediction_type="full", thr=thr["overall"]["full"], digits=5)

if __name__ == "__main__":
    main()