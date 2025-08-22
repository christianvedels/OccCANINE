from histocc import OccCANINE, EvalEngine
import pandas as pd
from histocc.prediction_assets import THRESHOLD_LOOKUP
import glob
import json
import os

def load_data(n_obs=5000, data_path=r"Z:\faellesmappe\tsdj\hisco\data/Test_data/*.csv", lang = None):
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

    df = df.sample(n=n_obs, random_state=20) if n_obs < df.shape[0] else df
    df = df.reset_index(drop=True)
    return df

def run_eval(df, mod, prediction_type, lang, thr=0.31, digits=5):
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
    dummy_files = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance/lang/test_performance_{prediction_type}_digits_5_lang_{lang}.csv"
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
            "lang": lang
        }])

        file = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance/lang/test_performance_{prediction_type}_digits_{d}_lang_{lang}.csv"
        # Make dir if missing
        os.makedirs(os.path.dirname(file), exist_ok=True)
        # Save results
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
            "lang": lang
        }])
        file_unk = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance/lang/test_performance_{prediction_type}_unk_digits_{d}_lang_{lang}.csv"
        res_unk.to_csv(file_unk, index=False)
        print(f"Results saved to {file_unk}")
        print(res_unk)

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
    unique_langs = list(THRESHOLD_LOOKUP.keys())
    for lang in unique_langs:
        if lang in ['overall']:
            continue # Skip overall as it is not a language

        print(f"Performing test for {lang}")
        
        if toyrun:
            df = load_data(n_obs=100, data_path=r"Z:\faellesmappe\tsdj\hisco\data/Test_data/*.csv", lang=lang)
        else:
            # Load full data for production run
            df = load_data(n_obs=1000000000000000, data_path=r"Z:\faellesmappe\tsdj\hisco\data/Test_data/*.csv", lang=lang)
        
        # Run evaluations for different prediction types
        run_eval(df, mod, prediction_type="flat", lang=lang, thr=THRESHOLD_LOOKUP.get(lang).get("flat"))
        run_eval(df, mod, prediction_type="greedy", lang=lang, thr=99) # Placeholder threshold (not used in greedy)
        if not toyrun: # Only run full for production
            run_eval(df, mod, prediction_type="full", lang=lang, thr=THRESHOLD_LOOKUP.get(lang).get("full"))
        print(f"Completed evaluations for {lang}")

if __name__ == "__main__":
    main()




