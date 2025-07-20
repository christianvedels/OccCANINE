from histocc import OccCANINE, EvalEngine
from histocc.prediction_assets import THRESHOLD_LOOKUP
import pandas as pd
import numpy as np
import glob
import os
import json

def perform_test(df, thr, probs, mod, digits = None, verbose = False):
    """
    Perform test on the model with the given threshold and mode.
    Args:
        df (pd.DataFrame): DataFrame containing the samples.
        thr (float): Threshold for the model.
        digits (int): Number of digits of code to use.
        verbose (bool): Whether to print updates.

    """
    if verbose:
        print(f"\rPerforming test with threshold {thr}", end = "")

    preds = mod._format( # Use internal OccCANINE formatter to apply threshold
        out=probs,
        out_type="probs",
        what="pred",
        inputs=df["occ1"].tolist(),
        lang=df["lang"].tolist(),
        threshold=thr,
        k_pred=5
    )

    eval_engine = EvalEngine(
        mod, 
        df, 
        preds, 
        pred_col='hisco_', 
        digits = digits
    )

    res = pd.DataFrame([{
        "threshold": thr,
        "accuracy": eval_engine.accuracy(),
        "f1": eval_engine.f1(),
        "precision": eval_engine.precision(),
        "recall": eval_engine.recall(),
    }])

    return res

def load_data(n_obs=5000, data_path="Project_dissemination/Paper_replication_package/Data/Raw_data/Validation_data1/*.csv", lang = None):
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

def get_probs(df, mod, prediction_type="flat", lang=None):
    """
    
    """
    
    if lang is None:
        langs = df["lang"].tolist()
    else:
        langs = [lang for _ in range(len(df))]

    probs = mod(df["occ1"].tolist(), langs, prediction_type=prediction_type, what="probs", deduplicate=True)

    return probs


def apply_test_across_thr(df, probs, mod):

    # Try vals from 0.01 to 0.99
    results = []
    for thr in [round(x, 2) for x in list(np.arange(0.01, 1.00, 0.01))]:
        res = perform_test(df, thr, probs, mod, digits=None, verbose=True)
        results.append(res)

    print("")
    print("Finished test for all thresholds")

    results_df = pd.concat(results, ignore_index=True)

    return results_df


def wrapper(prediction_type="flat", n_obs=100):
    # Define result fname and test if it exists
    result_fname = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/threshold_tuning_{prediction_type}_langknown.csv"
    if os.path.exists(result_fname):
        print(f"Results for {prediction_type} already exist. Skipping...")
        return 0

    # Load model
    mod = OccCANINE(name='OccCANINE_s2s_mix', verbose = True)

    # Load data
    df = load_data(n_obs=n_obs)

    # Get probs
    probs_unk = get_probs(df, mod, prediction_type=prediction_type, lang="unk")
    probs_lang = get_probs(df, mod, prediction_type=prediction_type)

    # Perform test across thresholds
    results_df_unk = apply_test_across_thr(df, probs_unk, mod)
    results_df_lang = apply_test_across_thr(df, probs_lang, mod)

    # Add n as column
    results_df_unk["n"] = df.shape[0]
    results_df_lang["n"] = df.shape[0]
    
    # Save results
    results_df_lang.to_csv(f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/threshold_tuning_{prediction_type}_langknown.csv", index=False)
    results_df_unk.to_csv(f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/threshold_tuning_{prediction_type}_langunk.csv", index=False)

    # Compute for each lang
    for lang in THRESHOLD_LOOKUP.keys():
        print(f"Performing test for {lang}")

        df = load_data(n_obs=n_obs, data_path="Project_dissemination/Paper_replication_package/Data/Raw_data/Validation_data1/*.csv", lang=lang)
        print(f"Loaded data for {lang}; NROWS: {df.shape[0]}")

        # Get probs
        probs_unk = get_probs(df, mod, prediction_type=prediction_type, lang="unk")
        probs_lang = get_probs(df, mod, prediction_type=prediction_type, lang=lang)

        # Perform test across thresholds
        results_df_unk = apply_test_across_thr(df, probs_unk, mod)
        results_df_lang = apply_test_across_thr(df, probs_lang, mod)

        # Add n as column
        results_df_unk["n"] = df.shape[0]
        results_df_lang["n"] = df.shape[0]

        # Save results
        if not os.path.exists("Project_dissemination/Paper_replication_package/Data/Intermediate_data/thr_tuning_by_lang"):
            os.makedirs("Project_dissemination/Paper_replication_package/Data/Intermediate_data/thr_tuning_by_lang")
        fname1 = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/thr_tuning_by_lang/threshold_tuning_{prediction_type}_langknown_{lang}.csv"
        fname2 = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/thr_tuning_by_lang/threshold_tuning_{prediction_type}_langunk_{lang}.csv"
        results_df_lang.to_csv(fname1, index=False)
        results_df_unk.to_csv(fname2, index=False)

        print(f"Finished test for {lang}")

def optimal_thresholds():
    """
    Calculate optimal thresholds for flat and full predictions.
    Returns:
        tuple: Optimal thresholds for flat and full predictions.
    """
    # Load the results
    df_flat = pd.read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/threshold_tuning_flat_langknown.csv")
    df_full = pd.read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/threshold_tuning_full_langknown.csv")
    
    optimal_flat = df_flat.loc[df_flat["f1"].idxmax(), "threshold"]
    optimal_full = df_full.loc[df_full["f1"].idxmax(), "threshold"]

    print(f"Optimal flat threshold: {optimal_flat}")
    print(f"Optimal full threshold: {optimal_full}")

    # Dict for storing thresholds by language
    thresholds_by_lang = {
        "overall": {
            "flat": optimal_flat,
            "full": optimal_full
        }
    }
    # Load results by lang
    for lang in THRESHOLD_LOOKUP.keys():
        fname1 = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/thr_tuning_by_lang/threshold_tuning_flat_langknown_{lang}.csv"
        fname2 = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/thr_tuning_by_lang/threshold_tuning_full_langknown_{lang}.csv"
        if not os.path.exists(fname1) or not os.path.exists(fname2):
            print(f"Results for {lang} not found(!!!). Skipping...")
            continue

        df_flat_lang = pd.read_csv(fname1)
        df_full_lang = pd.read_csv(fname2)

        optimal_flat_lang = df_flat_lang.loc[df_flat_lang["f1"].idxmax(), "threshold"]
        optimal_full_lang = df_full_lang.loc[df_full_lang["f1"].idxmax(), "threshold"]

        print(f"Optimal flat threshold for {lang}: {optimal_flat_lang}")
        print(f"Optimal full threshold for {lang}: {optimal_full_lang}")

        thresholds_by_lang[lang] = {
            "flat": optimal_flat_lang,
            "full": optimal_full_lang
        }

    # Save thresholds to json file
    thresholds_path = "Project_dissemination/Paper_replication_package/Data/Intermediate_data/thresholds_by_lang.json"
    if not os.path.exists(os.path.dirname(thresholds_path)):
        os.makedirs(os.path.dirname(thresholds_path))

    with open(thresholds_path, "w") as f:
        json.dump(thresholds_by_lang, f)


def main(toyrun=False):
    # Run the main function
    if toyrun:
        # Run a toy example
        wrapper(prediction_type="flat", n_obs=1000)
        wrapper(prediction_type="full", n_obs=100)
    else:
        # Run the full example
        wrapper(prediction_type="flat", n_obs=100000)
        wrapper(prediction_type="full", n_obs=10000)

    # Optimal thresholds for flat and full
    optimal_thresholds()

if __name__ == "__main__":
    main()