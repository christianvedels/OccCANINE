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
You are now ready to use OccCANINE for automatic HISCO codes. 

To obtain HISCO codes for a dataset `example.csv`, which has a column `occ1` with occupational descriptions you can run the following to get a csv `output.csv` with HISCO codes:

```
python predict.py --fn-in path/to/example.csv --col occ1 --fn-out path/to/output.csv
```

OccCANINE works well in a multlingual setting without having the language specified. But the performance is even better when the language is explicitly provided. To specify that the occupational descriptions are English (`en`) you can run the following:

```
python predict.py --fn-in path/to/example.csv --col occ1 --fn-out path/to/output.csv --language en
```

You can specify any of the 13 languages in which it was trained on. Here is a full list of languages OccCANINE is trained on, and the abbreviation used:
+ English: "en"
+ Danish: "da"
+ Swedish: "se"
+ Dutch: "nl"
+ Catalan: "ca"
+ French: "fr"
+ Norwegian: "no"
+ Icelandic: "is"
+ Portugese: "pt"
+ German: "ge/de"
+ Spanish: "es"
+ Italian: "it"
+ Greek: "gr"

Of course, OccCANINE has many options to tinker with. To get started writing a full script to handle your specific need we recommend starting with: [PREDICT_HISCOs.py](https://github.com/christianvedels/OccCANINE/blob/main/PREDICT_HISCOs.py)