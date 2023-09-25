## Create the vitual environment from an anaconda prompt (admin)
`conda update -n base -c defaults conda`
`conda create --name HISCO`
`conda activate HISCO`

## Install pytorch to run on cuda 11.8
`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
### Verify the pytorch installation and that it is running on cuda 11.8
`python -c "import torch; print(torch.cuda.get_device_name(0))`

## Install transformers
`pip install transformers`

## Install other packages
`conda install -c anaconda pandas scikit-learn seaborn`

## For interactive mode in VS code
`conda install -n HISCO ipykernel --update-deps --force-reinstall`