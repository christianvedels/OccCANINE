## Clone Repository and Prepare Environment
To get started, first clone the repository locally:
```
git clone https://github.com/TorbenSDJohansen/OccCANINE
```

Then prepare an environment (here using conda and the name `occ`, and a CPU-only installation):
```
conda create -n occ python=3.11
conda activate occ
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install numpy pandas scikit-learn matplotlib seaborn unidecode
```

After making sure all dependencies are installed, use the following code to install `hisco`.
```
pip install path/to/hisco
```