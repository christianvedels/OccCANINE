from histocc import OccCANINE
from histocc.eval_metrics import TopKEvalEngine
import pandas as pd
import glob
import os
import numpy as np


def load_data(data_path, n_obs=5000, lang=None):
    """
    Load data from the given path and sample n_obs rows.
    Args:
        n_obs (int): Number of observations to sample.
        data_path (str): Path to the data files.
        lang (str): Language filter (optional).

    Returns:
        pd.DataFrame: Sampled DataFrame.
    """
    csv_files = glob.glob(data_path)
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    if lang:
        df = df[df["lang"] == lang]

    df = df.sample(n=n_obs, random_state=10) if n_obs < df.shape[0] else df
    df = df.reset_index(drop=True)

    print(f"Loaded {df.shape[0]} observations from {data_path}")

    return df


def run_topk_eval(df, mod, K=5, digits=5, name="test"):
    """
    Run top-k evaluation on test data.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        mod (OccCANINE): The OccCANINE model instance.
        K (int): Number of top predictions to generate.
        digits (int): Number of digits for rounding evaluation metrics.
        name (str): Name for the evaluation (used in file naming).

    Returns:
        pd.DataFrame: DataFrame with evaluation results.
    """
    # Create dir if it does not exist
    os.makedirs("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_test_top_k", exist_ok=True)
    fname = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_test_top_k/obs_{name}_topk_{K}.csv"
    
    if os.path.exists(fname):
        print(f"Skipping {fname} as predictions already exist.")
        preds = pd.read_csv(fname)
    else:
        print(f"Generating top-{K} predictions...")
        preds = mod(
            df["occ1"].tolist(),
            df["lang"].tolist(),
            deduplicate=True,
            prediction_type='greedy-topk',
            k_pred=K,
            order_invariant_conf=False
        )

        # Add in rowid to match predictions with ground truth
        # Each original observation will have K rows in the result
        preds["RowID"] = df.RowID.repeat(K).values  # Repeat each rowid K times
        
        # Check if 'n' in df
        if "n" in df.columns:
            preds["n"] = df.n.repeat(K).values
        else:
            preds["n"] = 1  # Default to 1 if 'n' is not present
        
        # Use TopKEvalEngine for evaluation
        eval_engine = TopKEvalEngine(
            mod,
            ground_truth=df,
            predicitons=preds,
            pred_col="hisco_",
            group_col="RowID",
            digits=digits
        )
        
        # Get per-observation metrics as flat lists matching prediction order
        preds["acc"] = eval_engine.accuracy(return_per_obs=True)
        preds["precision"] = eval_engine.precision(return_per_obs=True)
        preds["recall"] = eval_engine.recall(return_per_obs=True)
        preds["f1"] = eval_engine.f1(return_per_obs=True)

        # Original HISCO codes - add to each top-k row
        for i in range(1, 6):
            col = f"hisco_{i}"
            if col in df.columns:
                # Create a mapping from rowid to original hisco value
                hisco_mapping = df.set_index("RowID")[col]
                preds[f"{col}_original"] = preds["RowID"].map(hisco_mapping)

        # Save predictions
        preds.to_csv(fname, index=False)
        print(f"Predictions saved to {fname}")

def main(toyrun=False, data_path=r"Data/Test_data/*.csv", name="test", K=5):
    """
    Main function to load data, get top-k predictions, and evaluate the model.
    Args:
        toyrun (bool): Whether to run a toy example with fewer observations.
        data_path (str): Path to all data files for OccCANINE model.
        name (str): Name for the evaluation (used in file naming).
        K (int): Number of top predictions to generate.
    """
    # Load the model
    mod = OccCANINE()

    # Load data
    if toyrun:
        df = load_data(data_path=data_path, n_obs=1000, lang=None)
    else:
        # Load a larger dataset for full evaluation
        df = load_data(data_path=data_path, n_obs=1000000000000000, lang=None)

    # Run top-k evaluation
    run_topk_eval(df, mod, K=K, digits=5, name=name)


if __name__ == "__main__":
    main()
