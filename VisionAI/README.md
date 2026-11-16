# VisionAI – AI Detection & Annotation Workspace

A local web app for AI-assisted object detection, human annotation review, and training-label export.

## Features
- Upload an image and run object detection
- Display and review YOLO detections with bounding boxes and confidence scores
- Draw bounding boxes and polygons manually; edit class names and confirm annotations
- Track review status with a QC checklist and annotation guidance
- Export the current image's labels as YOLO, COCO JSON, or Pascal VOC XML
- FastAPI backend + React/Vite frontend

## Requirements
- Python 3.10+
- Node.js 18+
- Internet connection on first backend run so Ultralytics can download the pretrained model weights

## Run the backend
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Backend: http://localhost:8000  
API docs: http://localhost:8000/docs

## Run the frontend
Open another terminal:
```bash
cd frontend
npm install
npm run dev
```
Frontend: http://localhost:5173

## Model
The starter uses `yolo11n.pt` through Ultralytics. The model is loaded on first prediction and cached by the library. For production, pin package/model versions and review model licensing and deployment requirements.

## Annotation and exports

Upload an image, run AI detection or choose **Box**/**Polygon** to annotate by hand. Select an annotation to edit its class and confirm it, or use **Confirm all** after reviewing the image. The QC checklist counts annotations that still need review. Before export, check that class names are consistent, boxes tightly cover the intended objects, polygons follow object boundaries, and no visible objects or duplicate labels are missing/extra.

The export menu writes labels for the currently open image: YOLO normalized box rows (`class_id x_center y_center width height`), COCO JSON, or Pascal VOC XML. Polygon coordinates are included in COCO segmentation where available; YOLO and Pascal VOC exports represent boxes. This initial workspace exports one image at a time and does not bundle image files or class metadata into a multi-image dataset archive.

## API
`POST /api/detect`  
Multipart form fields:
- `file`: image file (JPEG, PNG, WEBP)
- `conf`: confidence threshold (0.05–0.95)

Returns image dimensions and a list of detections with class, confidence, and pixel coordinates (`x1`, `y1`, `x2`, `y2`).

## Notes
- This is a starter project, not a production deployment.
- Detection results are model predictions and may be incorrect; users should verify labels before using the output as training data.
- Uploaded files are processed in memory and are not persisted.
