# Reanalysis to Satellite (R2S)

R2S is a framework for simulating satellite observations with reanalysis data.

### Features

1. Downloading reanalysis data and satellite observations
2. Matching reanalysis data and satellite observations temporally and spatially to construct the R-S dataset
3. Training the simulator with Gradient Boosting Decision Tree (GBDT)
4. Improving simulator's performance on the rare domain with imbalanced learning

This is the flowchart of training the simulator:
<img alt="flowchart" src=https://github.com/Neo-101/R2S/raw/master/flowchart.png>

### Dependencies

All python dependencies are shown in `config/environment.yml`. The easiest way to setup the environment is using `conda`:
```
conda env create -f environment.yml
```

Meanwhile, users need install MySQL Server 8.0 to manage the data.

### Usage

This repository currently provides a case study of simulating SMAP satellite observations of ocean surface wind speed around tropical cyclone (TC) using ERA5 reanalysis data. Users who want to use R2S for other scenarios will need to modify the relevant code.

The workflow in controlled by `r2s/manager.py`. Please refer to the docstrings of `.py` files for specific use. Here are some examples:

Downloading IBTrACS and extracting features from it:
`
python manager.py --period=2000-01-01-00-00-00,2020-01-01-00-00-00 --ibtracs --basin=na
`

Matching ERA5 reanalysis and SMAP observations temporally and spatially to constrcut the R-S dataset:
`
python manager.py --period=2015-04-01-00-00-00,2020-01-01-00-00-00 --match_smap --basin=na 
`
Training the simulator using GBDT and imbalanced learning:
`
python manager.py --period=2015-04-01-00-00-00,2020-01-01-00-00-00 --basin=na --reg=lgb,focus,save,load,smogn_final,valid,optimize --smogn_target=train
`

Gaps in SMAP's observation of hurricane Florence's wind speed:
<img alt="smap_gaps" src=https://github.com/Neo-101/R2S/raw/master/smap_gaps.png>

Simulation produced by R2S, which fills the gaps and improve the temporal resolution:
<img alt="r2s_simulation" src=https://github.com/Neo-101/R2S/raw/master/r2s_simulation.png>
The solid black lines are the tracks of the aircraft, and the circles along the solid black lines indicate the resampled high-precision wind speed observed by instruments on aircraft.
