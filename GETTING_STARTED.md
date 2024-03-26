# Getting started
- See [Colab notebook](https://github.com/christianvedels/OccCANINE/blob/main/OccCANINE_colab.ipynb) for a demonstration of OccCANINE.
- To use the model, follow the below setup guide.

## Create environment
```
conda create -n hisco python=3.11 numpy pandas scikit-learn matplotlib seaborn
conda activate hisco
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install unidecode transformers
```

### Optionally verify PyTorch installation
```
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Install `histocc`

First clone the repository locally:
```
git clone https://github.com/christianvedels/OccCANINE.git
```

Us pip install to install `histocc`.
```
pip install path/to/cloned/repo
```

## Predict HISCO codes
You are now ready to use OccCANINE for automatic HISCO codes
Open [PREDICT_HISCOs.py](https://github.com/christianvedels/OccCANINE/blob/main/PREDICT_HISCOs.py) to get started.