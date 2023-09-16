# Import Necessary Libraries
import unittest
from adoption_predictor.adoption_predictor import AdoptionPredictor

import pandas as pd

# Create Unit Test Class
class TestInt(unittest.TestCase):
    def setUp(self):
        """create global variable for unit test"""
        self.classifier = AdoptionPredictor()
        self.classifier.read_model_from_disk('artifacts/model.json')

        # Define the columns
        self.test_columns = ['Type', 'Age', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize', 
                'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Fee', 'PhotoAmt', 
                'Adopted']
        
        self.test_rows = ["Dog,12,Mixed Breed,Male,Brown,White,Medium,Medium,No,Yes,Healthy,0,2,No".split(','),
                          "Cat,1,Domestic Medium Hair,Male,Black,Brown,Medium,Medium,Not Sure,Not Sure,Healthy,0,2,Yes".split(',')]

    def test_prediction1(self):
        '''Checking that prediction returns one additional column'''
        test_df = pd.DataFrame(self.test_rows, columns=self.test_columns)
        self.classifier.load_data(test_df)
        original_shape = self.classifier.data.shape[1]
        self.classifier.run_prediction()
        self.assertEqual(original_shape+1, self.classifier.data.shape[1])
    
    def test_prediction2(self):
        '''Checking known prediction results'''
        test_df = pd.DataFrame(self.test_rows, columns=self.test_columns)
        self.classifier.load_data(test_df)
        print(self.classifier.data)
        self.classifier.run_prediction()
        print(self.classifier.data)
        self.assertEqual('No',self.classifier.data.iloc[0, -1])
        self.assertEqual('Yes',self.classifier.data.iloc[1, -1])
        

# Call unittest class
unittest.main(argv=[''], verbosity=2, exit=False)
