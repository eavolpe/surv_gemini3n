# camera.py
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import random

router = APIRouter(prefix="/camera")  # Add prefix

templates = Jinja2Templates(directory="app/templates")

@router.get("/{camera_id}/frame", response_class=HTMLResponse)
async def get_camera_frame(request: Request, camera_id: int):
    image_url = f"/frames/frame_00{random.randint(1,9)}.jpg"
    return templates.TemplateResponse(
        "components/camera_card.html",
        {
            "request": request,
            "camera_id": camera_id,
            "description": "Simulated stream from camera",
            "image_url": image_url,
        }
    )
