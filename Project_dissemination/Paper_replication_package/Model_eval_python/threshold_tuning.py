from histocc import OccCANINE, EvalEngine
import pandas as pd
import numpy as np
import glob

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

def load_data(n_obs=5000, data_path="Data/Validation_data1/*.csv", lang = None):
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
    # Load model
    mod = OccCANINE(name='OccCANINE_s2s_mix', verbose = True)

    # Load data
    df = load_data(n_obs=n_obs, data_path="Data/Validation_data1/*.csv")

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
    results_df_lang.to_csv(f"Project_dissemination/Paper_replication_package/Data/threshold_tuning_{prediction_type}_langknown.csv", index=False)
    results_df_unk.to_csv(f"Project_dissemination/Paper_replication_package/Data/threshold_tuning_{prediction_type}_langunk.csv", index=False)

    # Compute for each lang
    unique_langs = df["lang"].unique()
    for lang in unique_langs:
        print(f"Performing test for {lang}")

        df = load_data(n_obs=n_obs, data_path="Data/Validation_data1/*.csv", lang=lang)
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
        results_df_lang.to_csv(f"Project_dissemination/Paper_replication_package/Data/thr_tuning_by_lang/threshold_tuning_{prediction_type}_langknown_{lang}.csv", index=False)
        results_df_unk.to_csv(f"Project_dissemination/Paper_replication_package/Data/thr_tuning_by_lang/threshold_tuning_{prediction_type}_langunk_{lang}.csv", index=False)

        print(f"Finished test for {lang}")


def main():
    # Run the main function
    wrapper(prediction_type="flat", n_obs=100000)
    wrapper(prediction_type="full", n_obs=10000)

if __name__ == "__main__":
    main()