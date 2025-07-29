import os
import cv2
import random

VIDEO_ROOT = "../videos/Anomaly-Videos-Part-1/"
OUTPUT_DIR = "sampled_frames"
TOTAL_FRAMES = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gather all video file paths with their categories
video_files = []
for category in os.listdir(VIDEO_ROOT):
    category_path = os.path.join(VIDEO_ROOT, category)
    if not os.path.isdir(category_path):
        continue
    for f in os.listdir(category_path):
        if f.lower().endswith(('.mp4', '.avi', '.mov')):
            video_files.append((category, os.path.join(category_path, f)))

if not video_files:
    print("No videos found.")
    exit(1)

for i in range(TOTAL_FRAMES):
    category, video_path = random.choice(video_files)
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames == 0:
        cap.release()
        print(f"Skipped {video_path} (no frames)")
        continue

    frame_id = random.randint(0, total_video_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    success, frame = cap.read()
    if success:
        out_path = os.path.join(
            OUTPUT_DIR,
            f"{category}_{os.path.splitext(os.path.basename(video_path))[0]}_frame{frame_id}.jpg"
        )
        cv2.imwrite(out_path, frame)
        print(f"Saved frame {i+1}/{TOTAL_FRAMES} from {video_path} at frame {frame_id}")
    else:
        print(f"Failed to read frame {frame_id} from {video_path}")
    cap.release()

print(f"Done. Collected up to {TOTAL_FRAMES} frames.")