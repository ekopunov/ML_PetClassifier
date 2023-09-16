
import warnings

# Filter out the FutureWarning from XGBoost
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from xgboost import XGBClassifier

import pandas as pd
import io
import numpy as np
import os
import logging
import time

logger = logging.getLogger(__name__)

logger.level = logging.INFO

class AdoptionPredictor:
    def __init__(self,input_data):
        self.features = None
        self.labels = None
        self.predictor = None
        self.data = input_data

        """Import training data into the class"""
        # Split Data into Training Features and Label
        self.features = self.data.iloc[:, :-1]
        self.labels = self.data.iloc[:, -1]
        
        # Convert training features from sting to category
        self.features = self.features.astype("category")

        # Encode Label column from string to integers
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(self.labels)
        self.labels = label_encoder.transform(self.labels)


    def split_data(self):
        """Split data into Train, Test & Val set (60-20-20)"""
        self.X_train, self.X_rem, self.y_train, self.y_rem = \
                train_test_split(self.features, self.labels, train_size=0.6)
        
        self.X_valid, self.X_test, self.y_valid, self.y_test = \
                train_test_split(self.X_rem, self.y_rem, test_size=0.5)

        print(f'Shape of Train Data: {self.X_train.shape}')  
        print(f'Shape of Train Label: {self.y_train.shape}\n\n')

        print(f'Shape of Test Data: {self.X_test.shape}')  
        print(f'Shape of Test Label: {self.y_test.shape}\n\n')  

        print(f'Shape of Train Data: {self.X_valid.shape}')  
        print(f'Shape of Train Data: {self.y_valid.shape}')  
           
    def check_for_null_columns(self):
        """Print out the number of empty columns in the input data"""
        null_data = self.data.isna().sum()
        
        print(f"There are {len(null_data) - self.data.shape[1]} null columns")
    
    def train_model(self,use_gpu=False,time_training=False):
        """Train Model Using XGBoost Classifier"""
        if use_gpu:
            xgb_tree_method = 'gpu_hist'
        else:
            xgb_tree_method = 'auto'

        eval_set = [(self.X_train, self.y_train), (self.X_valid, self.y_valid)]

        self.model = XGBClassifier(tree_method=xgb_tree_method,
                                   eval_metric=["error", "logloss"],
                                   enable_categorical=True,
                                   use_label_encoder=False,
                                   early_stopping_rounds=10)
        if time_training:
            start_time = time.perf_counter()
        
        self.model.fit(self.X_train, self.y_train,eval_set=eval_set, verbose=False)

        if time_training:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            logger.info(f"Training took {execution_time:.6f} seconds to execute.")
        
        return self.model   

    def evaluate_model(self):
        """Print out evaluations of the trained model"""
        
        #Boost round with the best score
        #TODO: Re-write to f-strings
        best_tree = self.model.best_iteration
        y_pred = self.model.predict(self.X_test, iteration_range=(0, best_tree))
        
        # Model Evaluation Metrics on Test Set
        print("|-----------------------------------------|")
        print('Accuracy Score: {:.2f}%'.format(accuracy_score(self.y_test, y_pred) * 100))  
        print('')
        print('Precision Score: {:.2f}%'.format(precision_score(self.y_test, y_pred, average='binary') * 100))  
        print('')
        print('Recall Score: {:.2f}%'.format(recall_score(self.y_test, y_pred, average='binary') * 100))
        print('')
        print('F1 Score: {:.2f}%'.format(f1_score(self.y_test, y_pred, average='binary') * 100)) 
        

    def write_model_to_disk(self,input_path:str):
        """Save Trained Model to Disk"""
        directory, filename = os.path.split(input_path)
        
        if not os.path.exists(directory):
            os.makedirs(directory) 
        self.model.save_model(input_path)
