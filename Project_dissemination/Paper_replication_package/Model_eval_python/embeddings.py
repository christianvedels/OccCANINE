from histocc import OccCANINE
import pandas as pd
import glob

def load_data(n_obs=5000, data_path="Data/Test_data/*.csv"):
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

    # Only data with one hisco 1 (hisco_2:hisco_5 == " ")
    df = df[df["hisco_2"]==" "]
    df = df[df["hisco_3"]==" "]
    df = df[df["hisco_4"]==" "]
    df = df[df["hisco_5"]==" "]

    df = df.sample(n=n_obs, random_state=20) if n_obs < df.shape[0] else df
    df = df.reset_index(drop=True)
    return df

def main():
    """
    Main function to test the OccCANINE model with embeddings.
    This is a simple test to ensure the model can handle embeddings.
    """
    # Load the model
    mod = OccCANINE()
    mod_null = OccCANINE(skip_load=True, model_type = "mix")

    # Load data
    df = load_data(n_obs=100000, data_path="Data/Test_data/*.csv")

    # Get embeddings for the input
    f = "Project_dissemination/Paper_replication_package/Data/big_files/embeddings_test.csv"
    if not glob.glob(f): # Check if file exists
        print(f"File {f} does not exist. Generating embeddings.")
        embeddings = mod(df.occ1.tolist(), lang=df.lang.tolist(), what="embeddings", deduplicate=True)
        embeddings.insert(0, 'hisco_1', df.hisco_1.tolist())
        embeddings.to_csv(f, index=False)

    # Get embeddings for null model (not trained)
    f_null = "Project_dissemination/Paper_replication_package/Data/big_files/embeddings_test_null.csv"
    if not glob.glob(f_null): # Check if file exists
        print(f"File {f_null} does not exist. Generating embeddings for null model.")
        embeddings_null = mod_null(df.occ1.tolist(), lang=df.lang.tolist(), what="embeddings", deduplicate=True)
        embeddings_null.insert(0, 'hisco_1', df.hisco_1.tolist())
        embeddings_null.to_csv(f_null, index=False)  


