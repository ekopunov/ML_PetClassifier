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

   logger.info('Starting model trainer')
   
   #Load the data from gcloud and save it locally
   if not os.path.exists('artifacts/input_file.csv'):
      file_name = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
      gcloud_downloader = GCloudDownloader(client_project="testingtheapi-357421")
      content = gcloud_downloader.get_blob(file_name)
      write_to_disk('artifacts/input_file.csv',content)
   else:
      content = read_from_disk('artifacts/input_file.csv')

   df = pd.read_csv(io.StringIO(content))

   adoption_predictor = AdoptionPredictor()
   adoption_predictor.load_data(df)
   adoption_predictor.check_for_null_columns()
   adoption_predictor.split_data()
   adoption_predictor.train_model(time_training=True)
   adoption_predictor.evaluate_model()
   adoption_predictor.write_model_to_disk('artifacts/model.json')
   

   