from histocc import OccCANINE, EvalEngine
import pandas as pd
import glob
import os
import json


def load_data(data_path, n_obs=5000, lang=None):
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

    print(f"Loaded {df.shape[0]} observations from {data_path}")

    return df


def load_optimal_thresholds():
    """
    Load optimal thresholds for flat predictions from a JSON file.

    Returns:
        dict: Dictionary with optimal thresholds for each language and overall.
    """
    thresholds_path = "Project_dissemination/Paper_replication_package/Data/Intermediate_data/thresholds_by_lang.json"

    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")

    with open(thresholds_path, "r") as f:
        thresholds_by_lang = json.load(f)

    return thresholds_by_lang


def _thr_tag(thr):
    return str(thr).replace(".", "p")


def run_eval_flat(df, mod, thr=0.31, digits=5, name="test_flat"):
    """
    Run evaluation for flat predictions at a given threshold.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        mod (OccCANINE): The OccCANINE model instance.
        thr (float): Threshold for predictions.
        digits (int): Number of digits for rounding evaluation metrics.
        name (str): Name for the evaluation (used in file naming).

    Returns:
        None
    """
    thr_tag = _thr_tag(thr)

    # Create dirs if they do not exist
    os.makedirs(f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/{name}_performance", exist_ok=True)

    # Check if all summary files exist
    summary_files = []
    for d in range(1, digits + 1):
        summary_files.append(
            f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/{name}_performance/test_performance_flat_digits_{d}_thr_{thr_tag}.csv"
        )
        summary_files.append(
            f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/{name}_performance/test_performance_flat_unk_digits_{d}_thr_{thr_tag}.csv"
        )

    if all(os.path.exists(f) for f in summary_files):
        print(f"Skipping threshold {thr} as all summary files already exist.")
        return

    preds = mod(
        df["occ1"].tolist(),
        df["lang"].tolist(),
        threshold=thr,
        prediction_type="flat",
        deduplicate=True
    )

    # Unknown-language predictions (always compute to ensure consistent eval)
    preds_unk = mod(
        df["occ1"].tolist(),
        "unk",
        threshold=thr,
        prediction_type="flat",
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
            "prediction_type": "flat",
            "lang": "known"
        }])

        file = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/{name}_performance/test_performance_flat_digits_{d}_thr_{thr_tag}.csv"
        if not os.path.exists(file):
            res.to_csv(file, index=False)
            print(f"Results saved to {file}")
            print(res)
        else:
            print(f"Skipping {file} as it already exists.")

        eval_engine_unk.digits = d
        res_unk = pd.DataFrame([{
            "threshold": thr,
            "accuracy": eval_engine_unk.accuracy(),
            "f1": eval_engine_unk.f1(),
            "precision": eval_engine_unk.precision(),
            "recall": eval_engine_unk.recall(),
            "n": df.shape[0],
            "prediction_type": "flat",
            "lang": "unk"
        }])

        file_unk = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/{name}_performance/test_performance_flat_unk_digits_{d}_thr_{thr_tag}.csv"
        if not os.path.exists(file_unk):
            res_unk.to_csv(file_unk, index=False)
            print(f"Results saved to {file_unk}")
            print(res_unk)
        else:
            print(f"Skipping {file_unk} as it already exists.")


def main(
    toyrun=False,
    data_path=r"Z:\faellesmappe\tsdj\hisco\data\Test_data\*.csv",
    thresholds=None,
    name="test_flat"
):
    """
    Main function to load data, get flat predictions, and evaluate the model.

    Args:
        toyrun (bool): Whether to run a toy example with fewer observations.
        data_path (str): Path to all data files for OccCANINE model.
        thresholds (list[float] | float | None): Thresholds to evaluate. If None,
            uses the optimal overall flat threshold from thresholds_by_lang.json.
        name (str): Name for the evaluation (used in file naming).
    """
    # Load the model
    mod = OccCANINE()

    # Load data
    if toyrun:
        df = load_data(data_path=data_path, n_obs=1000, lang=None)
    else:
        df = load_data(data_path=data_path, n_obs=1000000000000000, lang=None)

    if thresholds is None:
        thr = load_optimal_thresholds()
        thresholds = [thr["overall"]["flat"]]
    elif isinstance(thresholds, (int, float)):
        thresholds = [thresholds]

    for thr in thresholds:
        print(f"Running flat evaluation at threshold {thr}")
        run_eval_flat(df, mod, thr=thr, digits=5, name=name)


if __name__ == "__main__":
    main()
