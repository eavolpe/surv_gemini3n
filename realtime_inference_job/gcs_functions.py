import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secret.json"

bucket_name=  'gemma_prj'
source_file_name = 'sample.txt'
destination_blob_name = 'sample.txt'
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Upload a file to a private bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

upload_blob(bucket_name, source_file_name, destination_blob_name)