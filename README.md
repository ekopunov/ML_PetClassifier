### 1. Task

`Pet Adoption Classifier`
- This is a supervised model which trains an XGBoost Model to predict if a pet would be adopted or not.


### 2. Data

Contains 13 features and a labeled column (Yes or No) if a pet can be adopted. `gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv`

### 3. Requirements

Python >3.9 is expected.
Requirements are listed in requirements.txt
Adoption prediction and google cloud downloader are packaged as separate modules.
To install for running, run make install
For installing for development run `make install-dev`

### 4. Usage
There are three functions:
- Run model training - run `python 1_train_model.py`
- Run predictions based off of trained model - run `python 2_run_prediction.py`
- Run unit tests of prediction function - run `python tests/test_adoption_predictor.py`

