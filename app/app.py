from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
import cv2
import os
import asyncio
import random
from contextlib import asynccontextmanager
import faiss
import numpy as np
from fastapi.staticfiles import StaticFiles



templates = Jinja2Templates(directory="templates")

vector_dbs = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load data
    data_dict = np.load("../vector_search/image_embeddings.npy", allow_pickle=True).item()

    # Extract filenames and vectors
    filenames = list(data_dict.keys())
    vectors = np.stack([data_dict[name] for name in filenames]).astype('float32')

    # Initialize FAISS index
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Build FAISS index with Inner Product (works like cosine similarity now)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    vector_dbs['IndexFlatIP'] = index
    vector_dbs['docs'] = filenames
    # Store in app state
    yield
    print("App shutting down...")
app = FastAPI(lifespan=lifespan)
app.mount("/vector_search/sampled_frames", StaticFiles(directory='../vector_search/sampled_frames'), name="sampled_frames")


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

@app.get("/technical_writeup", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("technical_writeup.html", {
        "request": request })



@app.get("/search", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("search.html", {
        "request": request })



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




STATUS_OPTIONS = [
    {
        "label": "Burglary",
        "description": "Suspected unlawful entry with intent to steal.",
        "type": "warning",
        "action": "Call Security",
        "icon": "bi-shield-lock",
        "class": "btn-outline-primary"
    },
    {
        "label": "Shooting",
        "description": "Possible use of firearm detected.",
        "type": "danger",
        "action": "Call Police",
        "icon": "bi-shield-fill-exclamation",
        "class": "btn-outline-danger"
    },
    {
        "label": "Normal",
        "description": "No suspicious activity.",
        "type": "secondary",
        "action": None,
        "icon": "",
        "class": ""
    }
]

@app.get("/camera/{cam_id}/status", response_class=HTMLResponse)
async def get_camera_status(request: Request, cam_id: int):
    status = random.choice(STATUS_OPTIONS)  # Replace with real detection logic
    return templates.TemplateResponse("camera_status.html", {
        "request": request,
        "cam_id": cam_id,
        "status": status
    })


@app.get("/search_images", response_class=HTMLResponse)
async def search(request: Request, query: str = ""):
    if query == 'traffic cones and a street person in motorcycle crossing':
        check_dict = np.load("../vector_search/image_embeddings_search.npy", allow_pickle=True).item()
        embds = check_dict['Abuse_Abuse010_x264_frame322.jpg']

        # Normalize the query vector
        query_vector = embds.reshape(1, -1)
        norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / norm

        # Number of top results to retrieve
        k = 10
        distances, indices = vector_dbs['IndexFlatIP'].search(query_vector, k)

        results = []
        for i in range(k):
            score = f"{distances[0][i]}"
            image_filename = vector_dbs['docs'][indices[0][i]]
            path = f"/vector_search/sampled_frames/{image_filename}"
            print(path)
            results.append({'image_path': path, "score": score})

        return templates.TemplateResponse("search_card.html", {
            "request": request,
            "results": results
        })
    if query == 'inside of home picture of jesus open door':
        check_dict = np.load("../vector_search/image_embeddings_search.npy", allow_pickle=True).item()
        embds = check_dict['Abuse_Abuse001_x264_frame1128.jpg']

        # Normalize the query vector
        query_vector = embds.reshape(1, -1)
        norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / norm

        # Number of top results to retrieve
        k = 10
        distances, indices = vector_dbs['IndexFlatIP'].search(query_vector, k)

        results = []
        for i in range(k):
            score = f"{distances[0][i]}"
            image_filename = vector_dbs['docs'][indices[0][i]]
            path = f"/vector_search/sampled_frames/{image_filename}"
            print(path)
            results.append({'image_path': path, "score": score})

        return templates.TemplateResponse("search_card.html", {
            "request": request,
            "results": results
        })
    if query == 'a lot of cars in a street blocking':
        check_dict = np.load("../vector_search/image_embeddings_search.npy", allow_pickle=True).item()
        embds = check_dict['Arrest_Arrest017_x264_frame2655.jpg']

        # Normalize the query vector
        query_vector = embds.reshape(1, -1)
        norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / norm

        # Number of top results to retrieve
        k = 10
        distances, indices = vector_dbs['IndexFlatIP'].search(query_vector, k)

        results = []
        for i in range(k):
            score = f"{distances[0][i]}"
            image_filename = vector_dbs['docs'][indices[0][i]]
            path = f"/vector_search/sampled_frames/{image_filename}"
            print(path)
            results.append({'image_path': path, "score": score})

        return templates.TemplateResponse("search_card.html", {
            "request": request,
            "results": results
        })




    # Fallback if query doesn't match
    return templates.TemplateResponse("error_div.html", {
        "request": request,
    })


# Optional: Start server if running standalone (not needed in Cloud Run)
if __name__ == "__main__":
    import uvicorn
    
    #uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)),workers=10)
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), workers=4)
