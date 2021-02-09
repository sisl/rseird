# R-SEIRD
Learning standard and Reactive SEIRD models to explain COVID-19 outbreaks.

## Code Overview
- `src/models.py` defines the R-SEIRD and SEIRD dynamical systems.
- `src/datautils.py` defines helper functions for loading data.
- `scripts/ceem_run.py` is a script for fitting the SEIRD/R-SEIRD models to data
- `scripts/plot_trajectories_train.py` generates Figure 1 in the paper.
- `scripts/plot_trajectories_test.py` generates Figure 2 in the paper.
- `scripts/eval_for_hist.py` generates simulations Figure 3 in the paper.
- `scripts/plot_comparison_hist.py` generates Figure 3 in the paper.
- `scripts/plot_reactionfun.py` generates Figure 4 in the paper.

## Getting started
Clone CEEM and pip install the module.
```
git clone https://github.com/sisl/CEEM.git
cd CEEM
pip install -e .
cd ..
export PYTHONPATH=$(pwd):$PYTHONPATH
```
Install additional requirements
```
pip install -r requirements.txt
```

## Download datasets
Run `bash get_data.sh` to download datasets.

## Train models
Train the R-SEIRD model (takes ~16 minutes)
```
python scripts/ceem_run.py 
cp ./data/reactive/ckpts/best_model.th ./trained_models/reactive.th
```
Train the SEIRD model (takes ~8 minutes)
```
python scripts/ceem_run.py -c 1
cp ./data/const/ckpts/best_model.th ./trained_models/const.th
```

## Run plotting scripts

### Figure 1
```
python scripts/plot_trajectories_train.py
```
### Figure 2
```
python scripts/plot_trajectories_test.py
```
### Figure 3
```
python scripts/eval_for_hist.py
python scripts/eval_for_hist.py -c 1
python scripts/plot_comparison_hist.py
```
### Figure 4
```
python scripts/plot_reactionfun.py
```

