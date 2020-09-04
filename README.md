# R2S: from Reanalysis to Satellite Observations

R2S is a framework for simulating satellite observations with reanalysis data.

### Features

1. Downloading reanalysis data and satellite observations
2. Matching reanalysis data and satellite observations temporally and spatially to construct the dataset
3. Training the simulator with Gradient Boosting Decision Tree (GBDT)
4. Improving simulator's performance on the rare domain with imbalanced learning

### Dependencies

All dependencies are shown in **config/environment.yml**. The easiest way to setup the environment is using `conda`:
```
{
  conda env create -f environment.yml
}
```
