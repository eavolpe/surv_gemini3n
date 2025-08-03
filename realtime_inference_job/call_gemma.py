import cv2
import base64
import json
import subprocess
import requests
import time
import numpy as np
from secrets import gemma_url
##Structured ouputs 
# CONFIG
img_url = "https://siata.gov.co/ultimasFotosCamaras/ultimacam_nivel_quebrada_despoblada.jpg"
model_name = "gemma3:4b"
prompt = """
Briefly describe what is happening in the scene.

Then, assign **one** label that best describes the event. Choose from the following list:

normal, abuse, arrest, arson, assault, accident, burglary, explosion, fighting, robbery, shooting, stealing, shoplifting, vandalism.

Only choose a non-"normal" label **if there is clear, unmistakable evidence of that event type**. If there is any doubt or ambiguity, use "normal".
"""

# Download image
response = requests.get(img_url)
if response.status_code != 200:
    raise Exception("Failed to download image.")

# Convert image bytes to OpenCV format
image_arr = np.frombuffer(response.content, np.uint8)
frame = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)

# Base64 encode
_, buffer = cv2.imencode('.jpg', frame)
base64_str = base64.b64encode(buffer).decode('utf-8')

# Create JSON payload
payload = {
    "model": model_name,
    "prompt": prompt,
    "images": [base64_str],
    "stream": False,
    "format": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "A short summary label for the generated content"
            },
            "description": {
                "type": "string",
                "description": "A detailed explanation or description of the content"
            }
        },
        "required": ["label", "description"]
    }
}



# Send with curl
start_time = time.time()
print("Sending to Ollama...")

import json
import time
import subprocess

start_time = time.time()

result = subprocess.run(
    [
        "curl", "-N", f"{gemma_url}/api/generate",
        "-d", json.dumps(payload)
    ],
    capture_output=True,
    text=True
)

duration = time.time() - start_time
full_output = result.stdout.strip()

if not full_output:
    print("❌ Error: No response received from Ollama.")
    print("Raw stdout:", result.stdout)
    print("Stderr:", result.stderr)
else:
    print(f"✅ Response time: {duration:.2f} seconds")
    print("🧠 Model output:\n", full_output)
    
    # full_output is the whole JSON from the model,
    # we first parse it as JSON
    try:
        model_result = json.loads(full_output)
    except json.JSONDecodeError as e:
        print("Failed to parse top-level JSON:", e)
        model_result = None

    if model_result:
        # Now parse the inner JSON string from the 'response' key
        try:
            parsed_response = json.loads(model_result["response"])
            label = parsed_response.get("label")
            description = parsed_response.get("description")
            
            print(f"Label: {label}")
            print(f"Description: {description}")
        except (KeyError, json.JSONDecodeError) as e:
            print("Failed to parse 'response' field:", e)
