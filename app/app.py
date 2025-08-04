from fastapi import FastAPI, Request,HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.responses import Response
import httpx
import cv2
import os
import asyncio
import random
from contextlib import asynccontextmanager
import faiss
import numpy as np
from fastapi.staticfiles import StaticFiles
from google.cloud import storage
import io
import json

templates = Jinja2Templates(directory="templates")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secret.json"

vector_dbs = {}

GCS_BUCKET = "gemma_prj"
URLS_FILE = "urls.txt"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load data
    data_dict = np.load("./vs_embds/image_embeddings.npy", allow_pickle=True).item()

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
app.mount("/vector_search/sampled_frames", StaticFiles(directory='./vs_embds/sampled_frames'), name="sampled_frames")
app.mount("/demo_videos", StaticFiles(directory="demo_videos"), name="demo_videos")



def get_urls_from_gcs(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()
    return [line.strip() for line in content.strip().splitlines() if line.strip()]


def extract_source_id(url):
    filename = os.path.basename(url)  # e.g., ultimacam_drenaje_urbano_monterrey.jpg
    if filename.endswith(".jpg"):
        return filename[:-len(".jpg")]  # keep "ultimacam_" in the ID
    raise ValueError(f"Invalid image URL format: {url}")



def get_gcs_blob(bucket_name: str, blob_path: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"Blob not found: {blob_path}")
    return blob





# Simulated list of available cameras
URL_LIST = get_urls_from_gcs( "gemma_prj", "urls.txt")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    config = []
    for url in URL_LIST:
        config.append({'id':extract_source_id(url),'name':extract_source_id(url)})
        #get metadata from last inference and image

    return templates.TemplateResponse("index.html", {
        "request": request,
        "cameras": config
    })

@app.get("/technical_writeup", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("technical_writeup.html", {
        "request": request })



@app.get("/search", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("search.html", {
        "request": request })



# @app.get("/stream_text_test", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("text_stream.html", {
#         "request": request,
#         "cameras": CAMERAS_CONFIG
#     })


# Legacy stream response for video type
# @app.get("/camera/{camera_id}/stream")
# async def stream_camera(camera_id: int):
#     frame_folder = f"./frames/"
#     frame_files = sorted([
#         f for f in os.listdir(frame_folder)
#         if f.lower().endswith(('.jpg', '.jpeg', '.png'))
#     ])

#     async def generate():
#         while True:
#             for filename in frame_files:
#                 img_path = os.path.join(frame_folder, filename)
#                 frame = cv2.imread(img_path)

#                 if frame is None:
#                     continue

#                 _, buffer = cv2.imencode('.jpg', frame)
#                 yield (
#                     b"--frame\r\n"
#                     b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
#                 )
#                 await asyncio.sleep(0.3)

#     return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera/{camera_id}/image")
async def get_camera_image(camera_id: str):
    bucket_name = GCS_BUCKET  # Replace with your actual GCS bucket
    base_path = f"sources/{camera_id}/latest"

    image_blob = get_gcs_blob(bucket_name, f"{base_path}/image.jpg")
    image_bytes = image_blob.download_as_bytes()
    image_stream = io.BytesIO(image_bytes)

    return StreamingResponse(image_stream, media_type="image/jpeg")



# @app.get("/camera/all/stream_text")
# async def camera_text_stream(request: Request):
#     async def event_stream():
#         try:
#             async for message in event_generator():
#                 if await request.is_disconnected():
#                     break
#                 yield message
#         except asyncio.CancelledError:
#             pass
#     return StreamingResponse(event_stream(), media_type="text/event-stream")



@app.get("/camera/{cam_id}/status", response_class=HTMLResponse)
async def get_metadata(request: Request, cam_id: str):
    blob = get_gcs_blob(GCS_BUCKET, f"sources/{cam_id}/latest/metadata.json")
    metadata = json.loads(blob.download_as_text())
    return templates.TemplateResponse("camera_status.html", {
        "request": request,
        "cam_id": cam_id,
        "status": metadata
    })

@app.get("/search_images", response_class=HTMLResponse)
async def search(request: Request, query: str = ""):
    if query == 'traffic cones and a street person in motorcycle crossing':
        check_dict = np.load("./vs_embds/image_embeddings_search.npy", allow_pickle=True).item()
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
            results.append({'image_path': path, "score": score})

        return templates.TemplateResponse("search_card.html", {
            "request": request,
            "results": results
        })
    if query == 'inside of home picture of jesus open door':
        check_dict = np.load("./vs_embds/image_embeddings_search.npy", allow_pickle=True).item()
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
            results.append({'image_path': path, "score": score})

        return templates.TemplateResponse("search_card.html", {
            "request": request,
            "results": results
        })
    if query == 'a lot of cars in a street blocking':
        check_dict = np.load("./vs_embds/image_embeddings_search.npy", allow_pickle=True).item()
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
            results.append({'image_path': path, "score": score})

        return templates.TemplateResponse("search_card.html", {
            "request": request,
            "results": results
        })




    # Fallback if query doesn't match
    return templates.TemplateResponse("error_div.html", {
        "request": request,
    })


@app.get("/offline", response_class=HTMLResponse)
async def offline_demo(request: Request):
    static_cameras = [
        {
            "id": 1,
            "name": "Explosion",
            "description": " The image shows a car dealership with multiple vehicles. There's a significant explosion occurring in the center of the showroom, sending debris flying. People are visible nearby, reacting to the event.",
            "recommendation": "explosion"
        },
        {
            "id": 2,
            "name": "Assault",
            "description": "The presence of an individual on the floor and the apparent movement suggests a physical altercation. While other categories could theoretically fit (like robbery or fighting), assault seems the most accurate description of the immediate visual evidence.",
            "recommendation": "assault"
        },
        {
            "id": 3,
            "name": "Burglary",
            "description": "The images show a dark, outdoor setting, likely a parking lot or driveway, at night. There are two individuals walking towards a car. One person appears to be carrying a tool (possibly a crowbar) and is actively interacting with the vehicle.",
            "recommendation": "burglary"
        },
        {
            "id": 4,
            "name": "Assault",
            "description": "The image shows several individuals surrounding a person lying on the ground, seemingly in distress. There are indications of a struggle, with one person holding a weapon. It appears to be a chaotic and potentially violent situation.",
            "recommendation": "assault"
        },
    ]
    return templates.TemplateResponse("offline.html", {"request": request, "static_cameras": static_cameras})

@app.get("/camera/{camera_id}/offline")
async def stream_camera(camera_id: int):
        frame_folder = f"./demo_videos/offline_caps_{camera_id}"
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
                    await asyncio.sleep(0.6)

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# Optional: Start server if running standalone (not needed in Cloud Run)
if __name__ == "__main__":
    import uvicorn
    
    #uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)),workers=10)
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), workers=4)
