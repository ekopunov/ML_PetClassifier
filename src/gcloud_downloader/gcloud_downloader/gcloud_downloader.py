from google.cloud import storage

class GCloudDownloader:
    def __init__(self, client_project):
        self.client_project = client_project
        self.client = storage.Client(project=client_project)

    def _split_input_path(self,input_path):    
        split_path = input_path.lstrip('gs://').split('/')
        bucket_name = split_path[0]
        blob_name = '/'.join(split_path[1:])

        return bucket_name,blob_name

    def get_blob(self,input_path):
        bucket_name,blob_name = self._split_input_path(input_path)
        # Get the bucket
        bucket = self.client.get_bucket(bucket_name)

        # Get the blob (file)
        blob = bucket.blob(blob_name)

        content = blob.download_as_text().replace('\\n','\n')

        return content
    
if __name__ == "__main__":
    gcloud_downloader = GCloudDownloader(client_project="testingtheapi-357421")
    
    content = gcloud_downloader.get_blob("gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv")
