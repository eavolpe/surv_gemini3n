import base64
from PIL import Image
import io
import requests
import fire

# Must have ollama installed and call a model that supports multimodal input.
# Both gsec2b and gsec4b can be created from the Modelfile in the prompts folders using the following command:
# ollama create gsec2b -f prompts/gemma2b/Modelfile
# ollama create gsec4b -f prompts/gemma4b/Modelfile

def main(image_path: str, model: str = 'gsec2b'):
    img = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    b64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': model,
            'prompt': 'Analyze this surveillance image for security threats and anomalies.',
            'images': [b64_img],
            'stream': False,
        }
    )

    print(response.json()['response'])

if __name__ == '__main__':
    fire.Fire(main)
