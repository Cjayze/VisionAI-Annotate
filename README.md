# VisionAI Annotate

AI-assisted image detection, annotation, review, and dataset export. YOLO11 Nano inference runs in the browser with ONNX Runtime Web, so uploaded images are not sent to an inference server.

# Link Deploy
https://cjayze.github.io/VisionAI-Annotate/

## Features

- Run object detection and view confidence-scored bounding boxes.
- Draw boxes and polygons, edit labels, and confirm AI suggestions.
- Review annotation status with a QC checklist.
- Export labels for the current image as YOLO, COCO JSON, or Pascal VOC XML.

## Run locally

Requirements: Node.js 18 or later and a modern browser with WebAssembly support.

```bash
cd VisionAI/frontend
npm ci
npm run dev
```

Open the local URL printed by Vite. The first detection downloads the 10.7 MB ONNX model; detection then runs on the device. The optional FastAPI backend remains available for API development, but the frontend does not need it to detect images.

## Deploy free as a static site

The repository includes a GitHub Actions workflow for GitHub Pages. In the repo, open **Settings → Pages** and set the source to **GitHub Actions**. Push changes to `main`; the workflow builds `VisionAI/frontend` and deploys it. The expected URL is `https://cjayze.github.io/VisionAI-Annotate/`.

The root `render.yaml` also configures the frontend as a Render Static Site. Both options host the app without a paid inference backend because AI detection runs in the browser.

## Annotation review and export

Select an annotation to edit its class and confirm it, or confirm all after reviewing the image. Check that class names are consistent, boxes fit object boundaries, polygons follow object outlines, and visible objects are not missing or duplicated.

Exports contain labels for the currently open image. YOLO and Pascal VOC represent boxes; COCO also includes polygon segmentation when available. This version does not bundle multiple images into a dataset archive.

## Model

The browser uses the exported `yolo11n.onnx` model in `VisionAI/frontend/public/models`. Model weights are downloaded by the browser on first inference. Review the Ultralytics model and software license terms before redistribution or commercial use.
