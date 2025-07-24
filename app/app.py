from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
import cv2
import os
import random
import time 
import asyncio

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Simulated list of available cameras
CAMERAS = [
    {"id": 1, "name": "Entrance"},
    {"id": 2, "name": "Backyard"},
    {"id": 3, "name": "Garage"},
    {"id": 4, "name": "Living Room"},
]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cameras": CAMERAS
    })



@app.get("/camera/{camera_id}/stream")
def stream_camera(camera_id: int):
    def generate():
        frame_folder = f"./frames/"
        frame_files = sorted(os.listdir(frame_folder))  # sort to maintain order

        while True:
            for filename in frame_files:
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue  # skip non-image files

                img_path = os.path.join(frame_folder, filename)
                frame = cv2.imread(img_path)

                if frame is None:
                    continue

                _, buffer = cv2.imencode('.jpg', frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
                time.sleep(0.5)  # control frame rate

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")



async def event_generator():
    text = "This is an example of streaming a response word by word."
    while True:  # Repeat forever
        for word in text.split():
            yield f"data: {word}\n\n"
            await asyncio.sleep(0.3)
        await asyncio.sleep(2)
        yield "data: [[CLEAR]]\n\n"

@app.get("/stream/{cam_id}")
async def sse_endpoint(request: Request, cam_id: int):
    async def event_stream():
        async for line in event_generator():
            if await request.is_disconnected():
                break
            yield line

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Optional: Start server if running standalone (not needed in Cloud Run)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))