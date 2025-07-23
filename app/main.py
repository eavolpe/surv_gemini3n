# main.py
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.routes import root, camera

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/frames", StaticFiles(directory="app/frames"), name="frames")  # fix mount path

templates = Jinja2Templates(directory="app/templates")

app.include_router(root.router)
app.include_router(camera.router)  # make sure camera router has prefix="/camera"

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
