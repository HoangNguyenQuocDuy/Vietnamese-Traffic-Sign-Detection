from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import pathlib

pathlib.PosixPath = pathlib.WindowsPath # Fix Windows's path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') # Load model
model.conf = 0.1  # Set confidence threshold preprocessing

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read() # Read image from request
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}

    results = model(img) # detect from model
    print(results.pandas().xyxy[0])  

    results_img = results.render()[0]  # image integrate bounding box
    pil_img = Image.fromarray(results_img)

    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
