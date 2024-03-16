# Getting started
- See [Colab notebook](https://github.com/christianvedels/OccCANINE/blob/main/OccCANINE_colab.ipynb) for a demonstration of OccCANINE
- To use the model at scale please clone/download the repository and follow the following setup guide:

## 1. Create the vitual environment from an anaconda prompt (admin)
```
conda update -n base -c defaults conda
conda create --name HISCO python=3.11
conda activate HISCO
```

## 2. Install pytorch to run on cuda 11.8
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Verify the pytorch installation and that it is running on cuda 11.8
```
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 3. Install transformers
```
pip install transformers
```
## 4. Install other packages
```
conda install -c anaconda pandas scikit-learn seaborn openpyxl SentencePiece protobuf
```
## 5. Installing spyder
```
conda install spyder
```
## 6. Install the hisco package

First clone the repository locally:
```
git clone https://github.com/christianvedels/OccCANINE.git
```

Us pip install to install `hisco`.
```
pip install path/to/cloned/repo
```

## 7. Predict HISCO codes
You are now ready to use OccCANINE for automatic HISCO codes
Open [PREDICT_HISCOs.py](https://github.com/christianvedels/OccCANINE/blob/main/PREDICT_HISCOs.py) to get started.