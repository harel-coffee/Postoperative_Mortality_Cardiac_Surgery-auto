# Postoperative mortality cardiac surgery

Code repository for Castela Forte et al. (2021) study on ML prediction models for prediction of 30-day, 1-year and 5-year mortality risk after cardiac surgery using pre-, intra-, and postoperative data

## How to run
To run the project, install conda [here](https://docs.anaconda.com/anaconda/install/).
In order to install the conda environment, please run the following command in your terminal:
```
make setup-conda-environment
```
To run the model on a small peri-operative dataset with default parameter settings, use the following command:
```
make train_model_total
```
Otherwise, you can run from your terminal the model with some custom parameter settings. Use the following command to see those:
```
python predictor.py -h
```
You can also clean the model saves and plots:
```
make clean-results
```

## Project structure
* **analyze_model_performance**: folder of scripts for generating metric scores and plots
* **data**: data folder
* **data500**: data folder containing small dataset of 500 patients
* **model**: folder containing different models needed for training
* **plots**: folder containing different model plots
* **preprocessing**: folder containing preprocessing scripts
* **stats**: folder with model metric results
* **training_data**: folder containing model saves for each experiment

