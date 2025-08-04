import os
import json
import base64
import requests
import cv2
import numpy as np
from datetime import datetime
from google.cloud import storage
from secrets import gemma_url
import requests

# ==== CONFIG ====
GCS_BUCKET = "gemma_prj"
URLS_FILE = "urls.txt"
MODEL_NAME = "gemma3:4b"
PROMPT = """Briefly describe what is happening in the scene.

Then, assign **one** label that best describes the event. Choose from the following list:

normal, abuse, arrest, arson, assault, accident, burglary, explosion, fighting, robbery, shooting, stealing, shoplifting, vandalism.

Only choose a non-"normal" label **if there is clear, unmistakable evidence of that event type**. If there is any doubt or ambiguity, use "normal".
"""

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secret.json"

# ==== UTILITY FUNCTIONS ====


def extract_source_id(url):
    filename = os.path.basename(url)  # e.g., ultimacam_drenaje_urbano_monterrey.jpg
    if filename.endswith(".jpg"):
        return filename[:-len(".jpg")]  # keep "ultimacam_" in the ID
    raise ValueError(f"Invalid image URL format: {url}")



def get_urls_from_gcs(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()
    return [line.strip() for line in content.strip().splitlines() if line.strip()]

def download_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {url}")
    image_arr = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(image_arr, cv2.IMREAD_COLOR)

import requests

def send_to_model(image_np):
    _, buffer = cv2.imencode('.jpg', image_np)
    base64_str = base64.b64encode(buffer).decode('utf-8')

    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "images": [base64_str],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["label", "description"]
        }
    }

    try:
        response = requests.post(
            f"{gemma_url}/api/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        model_result = response.json()
        parsed = json.loads(model_result["response"])
        return parsed.get("label"), parsed.get("description")

    except requests.RequestException as e:
        print(f"❌ HTTP error: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    return "error", "Failed to analyze image"


def upload_image_to_gcs(client, bucket, path, image_np):
    _, buffer = cv2.imencode('.jpg', image_np)
    blob = bucket.blob(path)
    blob.upload_from_string(buffer.tobytes(), content_type="image/jpeg")
    print(f"📤 Uploaded image: {path}")

def upload_json_to_gcs(client, bucket, path, data):
    blob = bucket.blob(path)
    blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
    print(f"📤 Uploaded metadata: {path}")

# ==== MAIN JOB ====

def process_sources_to_gcs():
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    urls = get_urls_from_gcs(GCS_BUCKET, URLS_FILE)


    print(f"\n🔍 Found {len(urls)} URLs to process...\n")

    for url in urls:
        try:
            source_id = extract_source_id(url)
            timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")

            image = download_image(url)
            label, description = send_to_model(image)

            print(f"✅ {source_id} | Label: {label} | Description: {description[:60]}...")

            # Paths
            base_path = f"sources/{source_id}"
            result_prefix = f"{base_path}/results/{timestamp}"
            latest_prefix = f"{base_path}/latest"

            # Metadata
            metadata = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "label": label,
                "description": description
            }

            # Upload results
            upload_image_to_gcs(client, bucket, f"{result_prefix}/image.jpg", image)
            upload_json_to_gcs(client, bucket, f"{result_prefix}/metadata.json", metadata)

            # Upload latest
            upload_image_to_gcs(client, bucket, f"{latest_prefix}/image.jpg", image)
            upload_json_to_gcs(client, bucket, f"{latest_prefix}/metadata.json", metadata)

        except Exception as e:
            print(f"❌ Error processing {url}: {e}")

# ==== ENTRY POINT ====

if __name__ == "__main__":
    process_sources_to_gcs()
