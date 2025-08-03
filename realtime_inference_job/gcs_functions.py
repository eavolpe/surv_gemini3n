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


from google.cloud import storage

def get_urls_from_gcs(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Read entire text file as a string
    content = blob.download_as_text()

    # Extract and return URLs as a list
    urls = [line.strip() for line in content.strip().splitlines() if line.strip()]
    return urls

# Example usage
print(get_urls_from_gcs(bucket_name,"urls.txt"))

#upload_blob(bucket_name, source_file_name, destination_blob_name)