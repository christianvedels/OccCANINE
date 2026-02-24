import glob
import os
import json

import pandas as pd

from histocc import OccCANINE, EvalEngine


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


def _build_inputs(mod, df, lang_override=None):
    occ_clean = mod._prep_str(df["occ1"].tolist())
    if lang_override is None:
        langs = df["lang"].tolist()
    else:
        langs = [lang_override] * len(occ_clean)
    return [f"{l}[SEP]{o}" for l, o in zip(langs, occ_clean)]


def run_eval_flat(
    df,
    mod,
    thresholds_by_lang,
    thresholds=None,
    digits=5,
    name="test_flat",
    by_lang=True,
    include_overall=True
):
    """
    Run evaluation for flat predictions at given thresholds.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        mod (OccCANINE): The OccCANINE model instance.
        thresholds_by_lang (dict): Threshold lookup by language.
        thresholds (list[float] | float | None): Threshold(s) for predictions.
        digits (int): Number of digits of HISCO codes to evaluate.
        name (str): Name for the evaluation (used in file naming).
        by_lang (bool): Whether to run language-specific evaluations.
        include_overall (bool): Whether to run the overall evaluation.

    Returns:
        None
    """
    # Create dirs if they do not exist
    os.makedirs(f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/{name}_performance", exist_ok=True)

    if thresholds is not None and isinstance(thresholds, (int, float)):
        thresholds = [thresholds]

    inputs_known_all = _build_inputs(mod, df)
    inputs_unk_all = _build_inputs(mod, df, lang_override="unk")

    probs_known_all = mod.predict(
        df["occ1"].tolist(),
        df["lang"].tolist(),
        prediction_type="flat",
        what="probs",
        deduplicate=True
    )

    probs_unk_all = mod.predict(
        df["occ1"].tolist(),
        "unk",
        prediction_type="flat",
        what="probs",
        deduplicate=True
    )

    def eval_subset(mask, lang_label, thresholds_list):
        if thresholds_list is None:
            thresholds_list = [thresholds_by_lang[lang_label]["flat"]]

        if len(mask) == 0:
            return

        df_sub = df.loc[mask].reset_index(drop=True)
        if df_sub.empty:
            print(f"Skipping {lang_label} as no observations were found.")
            return

        inputs_known = [inputs_known_all[i] for i in mask]
        inputs_unk = [inputs_unk_all[i] for i in mask]
        probs_known = probs_known_all[mask]
        probs_unk = probs_unk_all[mask]

        for thr in thresholds_list:
            thr_tag = _thr_tag(thr)

            file = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/{name}_performance/test_performance_flat_digits_{digits}_thr_{thr_tag}_lang_{lang_label}.csv"
            file_unk = f"Project_dissemination/Paper_replication_package/Data/Intermediate_data/{name}_performance/test_performance_flat_unk_digits_{digits}_thr_{thr_tag}_lang_{lang_label}.csv"

            if os.path.exists(file) and os.path.exists(file_unk):
                print(f"Skipping threshold {thr} as all summary files already exist.")
                continue

            preds = mod._format(
                probs_known,
                out_type="probs",
                what="pred",
                inputs=inputs_known,
                lang=lang_label,
                threshold=thr,
                k_pred=5,
                order_invariant_conf=True
            )

            preds_unk = mod._format(
                probs_unk,
                out_type="probs",
                what="pred",
                inputs=inputs_unk,
                lang="unk",
                threshold=thr,
                k_pred=5,
                order_invariant_conf=True
            )

            eval_engine = EvalEngine(
                mod,
                df_sub,
                preds,
                pred_col='hisco_',
                digits=digits
            )

            eval_engine_unk = EvalEngine(
                mod,
                df_sub,
                preds_unk,
                pred_col='hisco_',
                digits=digits
            )

            eval_engine.digits = digits
            res = pd.DataFrame([{
                "threshold": thr,
                "accuracy": eval_engine.accuracy(),
                "f1": eval_engine.f1(),
                "precision": eval_engine.precision(),
                "recall": eval_engine.recall(),
                "n": df_sub.shape[0],
                "prediction_type": "flat",
                "lang": lang_label
            }])

            if not os.path.exists(file):
                res.to_csv(file, index=False)
                print(f"Results saved to {file}")
                print(res)
            else:
                print(f"Skipping {file} as it already exists.")

            eval_engine_unk.digits = digits
            res_unk = pd.DataFrame([{
                "threshold": thr,
                "accuracy": eval_engine_unk.accuracy(),
                "f1": eval_engine_unk.f1(),
                "precision": eval_engine_unk.precision(),
                "recall": eval_engine_unk.recall(),
                "n": df_sub.shape[0],
                "prediction_type": "flat",
                "lang": lang_label
            }])

            if not os.path.exists(file_unk):
                res_unk.to_csv(file_unk, index=False)
                print(f"Results saved to {file_unk}")
                print(res_unk)
            else:
                print(f"Skipping {file_unk} as it already exists.")

    if include_overall:
        thresholds_overall = thresholds or [thresholds_by_lang["overall"]["flat"]]
        print(f"Running flat evaluation at thresholds {thresholds_overall} (overall)")
        eval_subset(mask=list(range(len(df))), lang_label="overall", thresholds_list=thresholds_overall)

    if by_lang:
        unique_langs = [k for k in thresholds_by_lang.keys() if k != "overall"]
        for lang in unique_langs:
            thresholds_lang = thresholds or [thresholds_by_lang[lang]["flat"]]
            print(f"Running flat evaluation at thresholds {thresholds_lang} (lang: {lang})")
            lang_mask = df.index[df["lang"] == lang].tolist()
            eval_subset(mask=lang_mask, lang_label=lang, thresholds_list=thresholds_lang)


def main(
    toyrun=False,
    data_path=r"Z:\faellesmappe\tsdj\hisco\data\Test_data\*.csv",
    thresholds=None,
    name="test_flat",
    by_lang=True,
    include_overall=True
):
    """
    Main function to load data, get flat predictions, and evaluate the model.

    Args:
        toyrun (bool): Whether to run a toy example with fewer observations.
        data_path (str): Path to all data files for OccCANINE model.
        thresholds (list[float] | float | None): Thresholds to evaluate. If None,
            uses the optimal overall flat threshold from thresholds_by_lang.json.
        name (str): Name for the evaluation (used in file naming).
        by_lang (bool): Whether to run language-specific evaluations.
        include_overall (bool): Whether to run the overall evaluation.
    """
    # Load the model
    mod = OccCANINE()

    # Load data
    if toyrun:
        df = load_data(data_path=data_path, n_obs=10000, lang=None)
    else:
        df = load_data(data_path=data_path, n_obs=1000000000000000, lang=None)

    thresholds_by_lang = load_optimal_thresholds()

    run_eval_flat(
        df,
        mod,
        thresholds_by_lang=thresholds_by_lang,
        thresholds=thresholds,
        digits=5,
        name=name,
        by_lang=by_lang,
        include_overall=include_overall
    )


if __name__ == "__main__":
    main()
