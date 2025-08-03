import cv2
import base64
import json
import subprocess
import os
import time
#ollama run gemma3n:4b
# CONFIG
video_path = "../videos/Anomaly-Videos-Part-1/Assault/Assault044_x264.mp4"
model_name = "gemma3:4b"
prompt = """Briefly describe the scene. Label the event with one word from: normal, abuse, arrest, arson, assault, accident, burglary, explosion, fighting, robbery, shooting, stealing, shoplifting, vandalism.
"""
num_frames = 3
resize_width = 256  # Optional: resize for faster inference
temp_dir_name ="offline_caps_4"
start_time_sec=13
end_time_sec=18
import os

def get_uniform_frames_from_segment(video_path, num_frames, start_time_sec, end_time_sec, resize_width=256):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or start_time_sec >= end_time_sec:
        raise ValueError("Invalid time range or FPS not available.")

    start_frame = int(start_time_sec * fps)
    end_frame = int(end_time_sec * fps)

    # Clamp to video bounds
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(0, min(end_frame, total_frames - 1))

    frame_ids = [int(start_frame + i * (end_frame - start_frame) / (num_frames - 1)) for i in range(num_frames)]
    frames_base64 = []

    temp_dir = os.path.join(os.getcwd(), temp_dir_name)
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Frames will be saved to: {temp_dir}")

    for idx, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        success, frame = cap.read()
        if not success:
            continue
        # Resize frame
        frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * resize_width / frame.shape[1])))
        frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.jpg")
        cv2.imwrite(frame_path, frame)

        # Base64 encode
        _, buffer = cv2.imencode('.jpg', frame)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        frames_base64.append(base64_str)

    cap.release()

    print(f"Extracted {len(frames_base64)} frames from {start_time_sec:.2f}s to {end_time_sec:.2f}s.")

    return frames_base64, temp_dir, round(end_time_sec - start_time_sec, 2)


# Extract and encode
images_base64,_ , seconds_spanned= get_uniform_frames_from_segment(video_path, num_frames,start_time_sec,end_time_sec)

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
