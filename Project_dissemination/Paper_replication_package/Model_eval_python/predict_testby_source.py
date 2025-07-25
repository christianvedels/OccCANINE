from histocc import OccCANINE, EvalEngine
import pandas as pd
from histocc.prediction_assets import THRESHOLD_LOOKUP
import glob
import json
import os

def load_data(n_obs=5000, data_path="Data/Test_data/*.csv"):
    """
    Load data from the given path and sample n_obs rows.
    Args:
        n_obs (int): Number of observations to sample.
        data_path (str): Path to the data files.

    Returns:
        pd.DataFrame: Sampled DataFrame.
    """
    df = pd.read_csv(data_path)

    df = df.sample(n=n_obs, random_state=20) if n_obs < df.shape[0] else df
    df = df.reset_index(drop=True)
    return df

def run_eval(df, mod, prediction_type, file, thr=0.31, digits=5):
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
    # Test if file exists
    dummy_files = f"Project_dissemination/Paper_replication_package/Data/test_performance/source/test_performance_{prediction_type}_digits_5_file_{file}"
    if os.path.exists(dummy_files):
        print(f"Skipping {dummy_files} as it already exists.")
        return

    if prediction_type == "full":
        # Downsample to 1k because its slow
        df = df.sample(n=1000, random_state=20) if df.shape[0] > 1000 else df

    preds = mod(
        df["occ1"].tolist(), 
        df["lang"].tolist(), 
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
            "file": file
        }])

        output_file = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance/source/test_performance_{prediction_type}_digits_{d}_file_{file}"
        # Make dir if missing
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Save results
        res.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        print(res)


def main(toyrun=False):
    """
    Main function to load data, get predictions, and evaluate the model.
    """
    # Load the model
    mod = OccCANINE()

    # Read the threshold lookup
    if not os.path.exists("Project_dissemination/Paper_replication_package/Data/Intermediate_data/thresholds_by_lang.json"):
        raise FileNotFoundError(
            "Threshold lookup file not found. Please run threshold tuning first."
        )
    
    with open("Project_dissemination/Paper_replication_package/Data/Intermediate_data/thresholds_by_lang.json", "r") as f:
        THRESHOLD_LOOKUP = json.load(f)

    # Load data
    files = glob.glob("Data/Test_data/*.csv")
    for f in files:

        print(f"Performing test for {f}")
        
        if toyrun:
            df = load_data(n_obs=100, data_path=f)
        else:
            # Load full data for production run
            df = load_data(n_obs=1000000000000, data_path=f)
        
        # Get file without path
        file = os.path.basename(f)

        # Run evaluations for different prediction types
        run_eval(df, mod, "greedy", file=file, digits=5)

if __name__ == "__main__":
    main()





