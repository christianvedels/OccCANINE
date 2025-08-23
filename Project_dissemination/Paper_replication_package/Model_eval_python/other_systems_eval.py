from histocc import OccCANINE, EvalEngine
import pandas as pd
import glob
import os

def test_performance(file = "tmp.csv", n_obs=1000, mod_path = r"Y:\pc-to-Y\hisco\ft-models\mixer-icem-ft/last.bin", data_path=r"Z:\faellesmappe\tsdj\hisco\data/Test_data_other/EN_OCCICEM_IPUMS_UK_test.csv", system = "OCCICEM"):
    # If file exists do nothing
    if os.path.exists(file):
        print(f"File {file} already exists. Skipping performance test.")
        return 0

    # Load the model
    mod = OccCANINE(mod_path, hf = False, system = system, batch_size=32)

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

    return 0

def main(toyrun=False):
    """
    Main function to load data, get predictions, and evaluate the model.
    """
    if toyrun:
        n_obs = 100
    else:
        n_obs = 10000000000

    # ICEM
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/occicem_performance.csv",
        n_obs=n_obs,
        mod_path=r"Y:\pc-to-Y\hisco\ft-models\mixer-icem-ft/last.bin",
        data_path=r"Z:\faellesmappe\tsdj\hisco\data/Test_data_other/EN_OCCICEM_IPUMS_UK_test.csv",
        system = "OCCICEM"
    )

    # ISCO68
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/isco68_performance.csv",
        n_obs=n_obs,
        mod_path=r"Y:\pc-to-Y\hisco\ft-models/mixer-isco-ft/last.bin",
        data_path=r"Z:\faellesmappe\tsdj\hisco\data/Test_data_other/EN_ISCO68_IPUMS_UK_test.csv",
        system = "ISCO68A"
    )

    # OCC1950
    test_performance(
        file="Project_dissemination/Paper_replication_package/Data/Intermediate_data/occ1950_performance.csv",
        n_obs=n_obs,
        mod_path=r"ZY:\pc-to-Y\hisco\ft-models/mixer-occ1950-ft/last.bin",
        data_path=r"Z:\faellesmappe\tsdj\hisco\data/Test_data_other/EN_OCC1950_IPUMS_US_test.csv",
        system = "OCC1950"
    )


if __name__ == "__main__":
    main()