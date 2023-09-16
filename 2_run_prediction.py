import pandas as pd
import io
import os

from adoption_predictor.adoption_predictor import AdoptionPredictor
from gcloud_downloader.gcloud_downloader import GCloudDownloader

from src.utils import read_from_disk,write_to_disk

import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
      #   logging.FileHandler('my_log.log')
    ]
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
   
   file_name = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"

   #Load the data from gcloud and save it locally
   if not os.path.exists('artifacts/input_file.csv'):
      
      gcloud_downloader = GCloudDownloader(client_project="testingtheapi-357421")
      content = gcloud_downloader.get_blob(file_name)
      write_to_disk('artifacts/input_file.csv',content)
   else:
      content = read_from_disk('artifacts/input_file.csv')

   df = pd.read_csv(io.StringIO(content))

   adoption_predictor = AdoptionPredictor(df)

   adoption_predictor.read_model_from_disk('artifacts/model.json')

   adoption_predictor.run_prediction()

   result_df = adoption_predictor.get_training_data()
   
   os.makedirs('output',exist_ok=True)
   result_df.to_csv('output/results.csv')

   