from histocc import OccCANINE, EvalEngine
import pandas as pd
import glob
import os

def test_performance(file = "tmp.csv", n_obs=1000, mod_path = "Data/models/mixer-icem-ft/last.bin", data_path="Data/Test_data_other/EN_OCCICEM_IPUMS_UK_test.csv", system = "OCCICEM"):
    # If file exists do nothing
    if os.path.exists(file):
        print(f"File {file} already exists. Skipping performance test.")
        return 0

    # Load the model
    mod = OccCANINE(mod_path, hf = False, system = system, batch_size=2048)

    # Load data
    df = pd.read_csv(data_path)
    df = df.sample(n=n_obs, random_state=20) if n_obs < df.shape[0] else df
    df = df.reset_index(drop=True)

    # Get predictions
    preds = mod(
        df["occ1"].tolist(),
        df["lang"].tolist(),
        deduplicate=True
    )

    eval_engine = EvalEngine(
        mod,
        df,
        preds,
        pred_col=f"{system}_"
    )

    # Individual level metrics
    res = pd.DataFrame([{
            "accuracy": eval_engine.accuracy(),
            "f1": eval_engine.f1(),
            "precision": eval_engine.precision(),
            "recall": eval_engine.recall(),
            "n": df.shape[0]
        }])
    res.to_csv(file, index=False)

    # Observation level metrics
    preds[f"acc"] = eval_engine.accuracy(return_per_obs = True)
    preds["precision"] = eval_engine.precision(return_per_obs = True)
    preds["recall"] = eval_engine.recall(return_per_obs = True)
    preds["f1"] = eval_engine.f1(return_per_obs = True)
    preds["rowid"] = df.RowID
    preds[f"true_{system}_1"] = df[f"{system}_1"]
    preds[f"true_{system}_2"] = df[f"{system}_2"]
    # Update file name and place in 'big_files/other_systems' instead
    fname = file.replace("performance", "obs_performance")
    fname = fname.replace("Data/Intermediate_data/other_systems", "Data/Intermediate_data/big_files/other_systems")
    preds.to_csv(fname, index=False)

    return 0

def main(toyrun=False, mod_path = "Data/models", data_path="Data/Test_data_other"):
    """
    Main function to load data, get predictions, and evaluate the model.
    """
    if toyrun:
        n_obs = 10000
    else:
        n_obs = 10000000000

    # ICEM
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/occicem_performance.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-icem-ft/last.bin",
        data_path=f"{data_path}/EN_OCCICEM_IPUMS_UK_test.csv",
        system = "OCCICEM"
    )

    # ICEM with 10k unique training strings (frozen encoder)
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/occicem_performance_n_unq10000.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-icem-n=10k-ft-frz-enc/last.bin",
        data_path=f"{data_path}/EN_OCCICEM_IPUMS_UK_test.csv",
        system = "OCCICEM"
    )

    # ICEM with 10k unique training strings (full finetuning)
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/occicem_performance_n_unq10000_full_finetuning.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-icem-n=10k-ft/last.bin",
        data_path=f"{data_path}/EN_OCCICEM_IPUMS_UK_test.csv",
        system = "OCCICEM"
    )

    # ICEM with 10k (non-unique) training strings (frozen encoder)
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/occicem_performance_n_10000.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-icem-n=10k-r-ft-frz-enc/last.bin",
        data_path=f"{data_path}/EN_OCCICEM_IPUMS_UK_test.csv",
        system = "OCCICEM"
    )

    # ICEM with 10k (non-unique) training strings (full finetuning)
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/occicem_performance_n_10000_full_finetuning.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-icem-n=10k-r-ft/last.bin",
        data_path=f"{data_path}/EN_OCCICEM_IPUMS_UK_test.csv",
        system = "OCCICEM"
    )

    # ISCO68
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/isco68_performance.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-isco-ft/last.bin",
        data_path=f"{data_path}/EN_ISCO68_IPUMS_UK_test.csv",
        system = "ISCO68A"
    )

    # ISCO68 with 10k unique training strings (frozen encoder)
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/isco68_performance_n_unq10000.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-isco-n=10k-ft-frz-enc/last.bin",
        data_path=f"{data_path}/EN_ISCO68_IPUMS_UK_test.csv",
        system = "ISCO68A"
    )

    # ISCO68 with 10k unique training strings (full finetuning)
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/isco68_performance_n_unq10000_full_finetuning.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-isco-n=10k-ft/last.bin",
        data_path=f"{data_path}/EN_ISCO68_IPUMS_UK_test.csv",
        system = "ISCO68A"
    )

    # ISCO68 with 10k (non-unique) training strings (frozen encoder)
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/isco68_performance_n_10000.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-isco-n=10k-r-ft-frz-enc/last.bin",
        data_path=f"{data_path}/EN_ISCO68_IPUMS_UK_test.csv",
        system = "ISCO68A"
    )
    # ISCO68 with 10k (non-unique) training strings (full finetuning)
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/isco68_performance_n_10000_full_finetuning.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-isco-n=10k-r-ft/last.bin",
        data_path=f"{data_path}/EN_ISCO68_IPUMS_UK_test.csv",
        system = "ISCO68A"
    )

    # OCC1950
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/occ1950_performance.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-occ1950-ft/last.bin",
        data_path=f"{data_path}/EN_OCC1950_IPUMS_US_test.csv",
        system = "OCC1950"
    )

    # OCC1950 with 10k unique training strings (frozen encoder)
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/occ1950_performance_n_unq10000.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-occ1950-n=10k-ft-frz-enc/last.bin",
        data_path=f"{data_path}/EN_OCC1950_IPUMS_US_test.csv",
        system = "OCC1950"
    )

    # OCC1950 with 10k unique training strings (full finetuning)
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/occ1950_performance_n_unq10000_full_finetuning.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-occ1950-n=10k-ft/last.bin",
        data_path=f"{data_path}/EN_OCC1950_IPUMS_US_test.csv",
        system = "OCC1950"
    )

    # OCC1950 with 10k (non-unique) training strings (frozen encoder)
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/occ1950_performance_n_10000.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-occ1950-n=10k-r-ft-frz-enc/last.bin",
        data_path=f"{data_path}/EN_OCC1950_IPUMS_US_test.csv",
        system = "OCC1950"
    )

    # OCC1950 with 10k (non-unique) training strings (full finetuning)
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/occ1950_performance_n_10000_full_finetuning.csv",
        n_obs=n_obs,
        mod_path=f"{mod_path}/mixer-occ1950-n=10k-r-ft/last.bin",
        data_path=f"{data_path}/EN_OCC1950_IPUMS_US_test.csv",
        system = "OCC1950"
    )


if __name__ == "__main__":
    main()