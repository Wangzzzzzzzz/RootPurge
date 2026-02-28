# RootPurge
Code Repository for Root Purge and post-train Rank Reduction as described in the paper: Characteristic Root Analysis and Regularization for Linear Time Series Forecasting

## Preparing Environment
You can install the environment with the following commands:

```
conda create -n timeseries python=3.9
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 cuda-version=12.4  -c pytorch -c nvidia
conda install numpy=1.26.0
conda install timm=1.0.12 -c conda-forge
pip install -r requirements.txt
```

where `requirements.txt` has the following content:

```
einops==0.8.0
local-attention==1.9.14
matplotlib==3.9.2
pandas==1.5.3
patool==1.12
reformer-pytorch==1.4.4
scikit-learn==1.2.2
scipy==1.10.1
sktime
sympy==1.11.1
tqdm==4.64.1
PyWavelets
```
