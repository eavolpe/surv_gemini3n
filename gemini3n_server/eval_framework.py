## Read from dataset a sample size. 
## Fine tune the prompt 
## Check accuracy 
## continuity simplemente crear los streams que se van a mostrar
## parse to enough tokens to gemma format, 
## Run a certain amount of videos 

# for context https://ai.google.dev/gemma/docs/gemma-3n
## 
# Multimodal (Vision)
# 4B parameter model (128k context window)


import cv2
import base64
import json
import subprocess
import os
import time

# CONFIG
video_path = "../videos/Anomaly-Videos-Part-1/Abuse/Abuse001_x264.mp4"
model_name = "gemma3:4b"
prompt = """What is the event in this surveillance frame? Respond with only one word from the following list:
normal, abuse, arrest, arson, assault, accident, burglary, explosion, fighting, robbery, shooting, stealing, shoplifting, vandalism.
"""
num_frames = 5
resize_width = 256  # Optional: resize for faster inference


import os

def get_uniform_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_ids = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames_base64 = []

    # Create temp folder relative to script
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Frames will be saved to: {temp_dir}")

    for idx, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        success, frame = cap.read()
        if not success:
            continue
        # Resize
        frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * resize_width / frame.shape[1])))
        # Save to file
        frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        # Base64 encode
        _, buffer = cv2.imencode('.jpg', frame)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        frames_base64.append(base64_str)

    cap.release()

    # Compute time spanned between first and last extracted frames
    if len(frame_ids) >= 2 and fps > 0:
        time_start = frame_ids[0] / fps
        time_end = frame_ids[-1] / fps
        time_span_seconds = round(time_end - time_start, 2)
    else:
        time_span_seconds = 0.0

    print(f"Time spanned in video: {time_span_seconds} seconds")

    return frames_base64, temp_dir, time_span_seconds

# Extract and encode
images_base64,_ , seconds_spanned= get_uniform_frames(video_path, num_frames)

# Create JSON payload
payload = {
    "model": model_name,
    "prompt": prompt,
    "images": images_base64
}

# Save to file for inspection
with open("ollama_payload.json", "w") as f:
    json.dump(payload, f, indent=2)

# Send with curl
import json

# Send with curl

start_time = time.time()

print("Sending to Ollama...")
result = subprocess.run(
    [
        "curl", "-N", "http://localhost:11434/api/generate",  # <- -N disables buffering
        "-d", json.dumps(payload)
    ],
    capture_output=True,
    text=True
)

end_time = time.time()
duration = end_time - start_time

# Process Ollama's streamed response
full_output = ""
for line in result.stdout.strip().splitlines():
    try:
        data = json.loads(line)
        full_output += data.get("response", "")
    except json.JSONDecodeError:
        continue

# Output results or error
if not full_output.strip():
    print("❌ Error: No response received from Ollama.")
    print("Raw stdout:", result.stdout)  # optional for debugging
    print("Stderr:", result.stderr)
else:
    print("✅ Response time: {:.2f} seconds".format(duration))
    print("🧠 Model output:\n", full_output)
