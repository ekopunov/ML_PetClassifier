from google.cloud import storage
from sklearn.model_selection import train_test_split
import pandas as pd
import io

# file_name = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"


# bucket_name = "cloud-samples-data"
# file_path = "ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"

# # Initialize the client
# client = storage.Client(project="testingtheapi-357421")

# # Get the bucket
# bucket = client.get_bucket(bucket_name)

# # Get the blob (file)
# blob = bucket.blob(file_path)

# # Download the file to a local path (optional)
# local_path = "local-file.csv"

# #Download treats endlines as characters, replacing with real endlines
# content = blob.download_as_text().replace('\\n','\n')
# df = pd.read_csv(io.StringIO(content))

df = pd.read_csv("petfinder-tabular-classification.csv")

train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(len(train_df),len(validation_df),len(test_df))