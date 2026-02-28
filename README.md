# RootPurge

Code repository for Root Purge and post-train Rank Reduction from the paper:
"Characteristic Root Analysis and Regularization for Linear Time Series Forecasting" (ICLR 2026).


### Quick Start — Environment Setup

Create the conda environment and install dependencies (recommended):

```bash
conda create -n timeseries python=3.9 -y
conda activate timeseries
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 cuda-version=12.4 -c pytorch -c nvidia -y
conda install numpy=1.26.0 -y
conda install timm=1.0.12 -c conda-forge -y
pip install -r requirements.txt
```

The `requirements.txt` used for the project contains the libraries required by various modules. Example contents (already present):

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

### Usage — Running experiments

- Reproduce Rank-Reduction experiments (examples): run the top-level scripts:
	- `python run_RRR.py` — run post-train Rank Reduction experiments.
	- `python run_DWRR.py` — run the DWRR variant experiments.

- Shell-run scripts for common experiment setups are in `run_scripts/`:
	- e.g. `run_scripts/run_rootpurge_speclin_logC.sh` and other specialized runners.

Examples (from project root):

```bash
# simple run (adjust args inside the script or call python files directly)
python run_RRR.py
python run_DWRR.py

# run a prepared shell script (make executable if needed)
bash run_scripts/run_root_purge.sh
```

### Project Structure (high-level)

- `RootPurge/` — core RootPurge implementation, model backbones, and utilities.
- `Rank_Reduction/` — code for post-training rank reduction methods, including both RRR and DWRR.

### Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```bibtex
@inproceedings{
    wang2026characteristic,
    title={Characteristic Root Analysis and Regularization for Linear Time Series Forecasting},
    author={Zheng Wang and Kaixuan Zhang and Wanfang Chen and Xiaonan Lu and Longyuan Li and Tobias Schlagenhauf},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=JTtwGRACte}
}
```

### Contact

If you find issues or want to contribute, please open an issue or a pull request on the repository. For direct questions, contact the authors listed on the paper. You can also contact Zheng Wang at `david.wang3@cn.bosch.com`.