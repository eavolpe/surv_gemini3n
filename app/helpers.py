from google.cloud import storage
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secret.json"

def get_image_urls_from_gcs(bucket_name, prefix="sources/"):
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)

    image_urls = []

    for blob in blobs:
        # Match: /sources/{cam_name}/latest/image.jpg
        if blob.name.endswith("latest/image.jpg"):
            parts = blob.name.split("/")
            if len(parts) >= 4:
                cam_name = parts[1]
                image_urls.append({
                    "camera": cam_name,
                    "image_url": blob.public_url
                })

    return image_urls

print(get_image_urls_from_gcs())
