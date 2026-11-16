from io import BytesIO
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

app = FastAPI(title="VisionAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@lru_cache(maxsize=1)
def get_model():
    # Downloads the pretrained weights on first run if they are not cached.
    return YOLO("yolo11n.pt")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "VisionAI API"}


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Please upload a JPG, PNG, or WEBP image.")

    if not 0.05 <= conf <= 0.95:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.05 and 0.95.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    try:
        model = get_model()
        result = model.predict(image, conf=conf, verbose=False)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(exc)}")

    names = result.names
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            detections.append({
                "id": len(detections) + 1,
                "class_id": class_id,
                "label": names[class_id],
                "confidence": round(float(box.conf[0].item()), 4),
                "box": {
                    "x1": round(coords[0], 2),
                    "y1": round(coords[1], 2),
                    "x2": round(coords[2], 2),
                    "y2": round(coords[3], 2),
                },
            })

    return {
        "filename": file.filename,
        "image": {"width": image.width, "height": image.height},
        "count": len(detections),
        "detections": detections,
    }
