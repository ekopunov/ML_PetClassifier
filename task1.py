from google.cloud import storage

file_name = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"


bucket_name = "cloud-samples-data"
file_path = "ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"

# Initialize the client
client = storage.Client(project="testingtheapi-357421")

# Get the bucket
bucket = client.get_bucket(bucket_name)

# Get the blob (file)
blob = bucket.blob(file_path)

# Download the file to a local path (optional)
local_path = "local-file.csv"

content = blob.download_as_text()

print(content)