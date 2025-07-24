from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
import cv2
import os
import asyncio
import time 

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


@app.get("/stream_text_test", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("text_stream.html", {
        "request": request,
        "cameras": CAMERAS
    })



@app.get("/camera/{camera_id}/stream")
async def stream_camera(camera_id: int):
    frame_folder = f"./frames/"
    frame_files = sorted([
        f for f in os.listdir(frame_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    async def generate():
        while True:
            for filename in frame_files:
                img_path = os.path.join(frame_folder, filename)
                frame = cv2.imread(img_path)

                if frame is None:
                    continue

                _, buffer = cv2.imencode('.jpg', frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
                await asyncio.sleep(0.3)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")



camera_ids = [1, 2, 3, 4]

# Simulate an incremental word-by-word stream
async def stream_message(message: str, delay: float = 0.3, event_name: str = "message"):
    words = message.split()
    sentence_so_far = ""
    for word in words:
        sentence_so_far += word + " "
        yield f"event: {event_name}\ndata: {sentence_so_far.strip()}\n\n"
        await asyncio.sleep(delay)
    await asyncio.sleep(1)  # Optional pause

# Generate events for all cameras
async def event_generator():
    example_messages = {
        1: "Camera 1 detected motion near the door.",
        2: "Camera 2 is operating normally.",
        3: "Camera 3 lost connection briefly.",
        4: "Camera 4 temperature threshold exceeded."
    }

    while True:
        for cam_id in camera_ids:
            message = example_messages[cam_id]
            event_name = f"cam-{cam_id}"
            async for update in stream_message(message, delay=0.2, event_name=event_name):
                yield update
        await asyncio.sleep(2)  # Delay before starting next loop
        

@app.get("/camera/all/stream_text")
async def camera_text_stream(request: Request):
    async def event_stream():
        try:
            async for message in event_generator():
                if await request.is_disconnected():
                    break
                yield message
        except asyncio.CancelledError:
            pass
    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Optional: Start server if running standalone (not needed in Cloud Run)
if __name__ == "__main__":
    import uvicorn
    
    #uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)),workers=10)
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), workers=4)
