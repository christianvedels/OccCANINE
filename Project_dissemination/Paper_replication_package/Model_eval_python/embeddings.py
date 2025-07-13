from histocc import OccCANINE
import pandas as pd
import glob
import sklearn

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

    # Distinct df
    df = df.drop_duplicates(subset=['occ1', 'hisco_1', 'lang'])

    df = df.sample(n=n_obs, random_state=20) if n_obs < df.shape[0] else df
    df = df.reset_index(drop=True)
    return df

def tsne(embeddings, d=2):
    """
    Perform t-SNE on the embeddings.
    Args:
        embeddings (pd.DataFrame): DataFrame containing embeddings.
        d (int): Number of dimensions for t-SNE.

    Returns:
        pd.DataFrame: DataFrame with t-SNE results.
    """
    from sklearn.manifold import TSNE

    # Prepare data for t-SNE
    tmp = embeddings.drop(columns=['occ1', 'hisco_1', 'lang'])
    
    # Perform t-SNE
    tsne_results = TSNE(n_components=d, random_state=20, verbose = 1).fit_transform(tmp)
    
    # Create DataFrame with t-SNE results
    tsne_data = pd.DataFrame(tsne_results, columns=[f'V{i+1}' for i in range(d)])
    tsne_data['occ1'] = embeddings['occ1']
    tsne_data['hisco_1'] = embeddings['hisco_1']
    tsne_data['lang'] = embeddings['lang']
    
    return tsne_data

def main():
    """
    Main function to test the OccCANINE model with embeddings.
    This is a simple test to ensure the model can handle embeddings.
    """
    # Load the model
    mod = OccCANINE()

    # Load data
    df = load_data(n_obs=10000, data_path="Data/Test_data/*.csv")

    # Get embeddings for the input
    f = "Project_dissemination/Paper_replication_package/Data/big_files/embeddings_test.csv"
    if not glob.glob(f): # Check if file exists
        print(f"File {f} does not exist. Generating embeddings.")
        embeddings = mod(df.occ1.tolist(), lang=df.lang.tolist(), what="embeddings", deduplicate=True)
        embeddings.insert(0, 'hisco_1', df.hisco_1.tolist())
        embeddings.to_csv(f, index=False)
    
    # Get t-SNE results
    if not glob.glob("Project_dissemination/Paper_replication_package/Data/big_files/tsne_results.csv"):
        print("Generating t-SNE results.")
        embeddings = pd.read_csv(f)
        tsne_data = tsne(embeddings, d=2)
        tsne_data.to_csv("Project_dissemination/Paper_replication_package/Data/big_files/tsne_results.csv", index=False)
        tsne_data_3d = tsne(embeddings, d=3)
        tsne_data_3d.to_csv("Project_dissemination/Paper_replication_package/Data/big_files/tsne_results_3d.csv", index=False)
    

