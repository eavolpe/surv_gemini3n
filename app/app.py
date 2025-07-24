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




# async def event_generator(cam_id: int):
#     count = 0
#     while True:
#         count += 1
#         yield f"event: sse\ndata: <span>Token {count}<br/></span>\n\n"
#         await asyncio.sleep(1)

# @app.get("/stream_text/{cam_id}")
# async def sse_endpoint(request: Request, cam_id: int):
#     async def event_stream():
#         async for line in event_generator(cam_id):
#             if await request.is_disconnected():
#                 break
#             yield line
#     return StreamingResponse(event_stream(), media_type="text/event-stream")



# async def token_stream():
#     # Simulated tokens from an LLM response, one by one
#     tokens = ["Hello", ", ", "this", " ", "is", " ", "a", " ", "streamed", " ", "response", "."]
#     for token in tokens:
#         # SSE event with event: sse and data as token
#         yield f"event: sse\ndata: {token}\n\n"
#         await asyncio.sleep(0.5)  # simulate delay between tokens
#     # Keep connection alive (optional)
#     while True:
#         await asyncio.sleep(10)

# @app.get("/stream_text/123")
# async def stream_text(request: Request):
#     async def event_generator():
#         async for message in token_stream():
#             if await request.is_disconnected():
#                 break
#             yield message

#     return StreamingResponse(event_generator(), media_type="text/event-stream")




async def event_generator():
    messages = [
        "Welcome to the chatroom!",
        "User123 joined.",
        "User123: Hello everyone!",
        "User456 joined.",
        "User456: Hi User123!",
    ]
    for msg in messages:
        yield f"data: {msg}\n\n"
        await asyncio.sleep(3)
    while True:
        await asyncio.sleep(10)  # keep connection alive

@app.get("/chatroom")
async def chatroom(request: Request):
    async def event_stream():
        async for message in event_generator():
            if await request.is_disconnected():
                break
            yield message
    return StreamingResponse(event_stream(), media_type="text/event-stream")



# Optional: Start server if running standalone (not needed in Cloud Run)
if __name__ == "__main__":
    import uvicorn
    
    #uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)),workers=10)
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), workers=1)
