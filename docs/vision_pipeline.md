# YOLOv8n Sidewalk Vision Pipeline

## Recommended Project Structure

```text
Gilnun/
|-- backend/
|   |-- main.py
|   |-- core/
|   |   `-- detector.py
|   `-- services/
|-- datasets/
|   `-- yolo_sidewalk/
|       |-- images/train/
|       |-- images/val/
|       |-- labels/train/
|       |-- labels/val/
|       `-- data.yaml
|-- vision/
|   |-- train.py
|   |-- validate.py
|   |-- predict.py
|   `-- tracker_logic.py
|-- runs/
|   `-- detect/runs/sidewalk/
|       `-- yolov8n_sidewalk-3/weights/best.pt
`-- requirements.txt
```

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m vision.train --data datasets/yolo_sidewalk/data.yaml
python -m vision.validate --weights runs/detect/runs/sidewalk/yolov8n_sidewalk-3/weights/best.pt
python -m vision.predict --weights runs/detect/runs/sidewalk/yolov8n_sidewalk-3/weights/best.pt --source test.jpg
python -m vision.predict --weights runs/detect/runs/sidewalk/yolov8n_sidewalk-3/weights/best.pt --source 0 --track-approach
```

## CPU Inference Settings

Use `yolov8n`, `device="cpu"`, `half=False`, and `model.fuse()`. Start with `imgsz=416`, `conf=0.35`, `iou=0.5`, and `max_det=20`.

For safety-critical classes such as `person`, `wheelchair`, `stroller`, `bicycle`, `motorcycle`, and `scooter`, consider a lower threshold around `0.25` to reduce missed detections. For noisy classes such as `pole`, `tree_trunk`, `traffic_sign`, and `movable_signage`, use `0.40` to `0.55` if false alerts become distracting.

## FastAPI Integration Snippet

```python
from fastapi import FastAPI, File, HTTPException, UploadFile

from vision.predict import SidewalkYoloDetector
from vision.tracker_logic import ApproachTracker, Detection

app = FastAPI()
detector: SidewalkYoloDetector | None = None
tracker = ApproachTracker(min_growth_ratio=1.35, center_band_only=False)


@app.on_event("startup")
async def load_detector() -> None:
    global detector
    detector = SidewalkYoloDetector(
        "runs/detect/runs/sidewalk/yolov8n_sidewalk-3/weights/best.pt",
        imgsz=416,
        conf=0.35,
        iou=0.5,
        max_det=20,
    )


@app.post("/detect")
async def detect(image: UploadFile = File(...)) -> dict:
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector is not loaded.")

    image_bytes = await image.read()
    frame = detector.predict_bytes_with_shape(image_bytes)

    # If the client sends continuous frames from one camera session, keep one
    # tracker per session/user instead of one global tracker.
    detections = [
        Detection(pred.label, pred.confidence, pred.bbox_xyxy)
        for pred in frame.predictions
    ]
    alerts = tracker.update(detections, frame_width=frame.width)

    return {
        "objects": [pred.__dict__ for pred in frame.predictions],
        "approach_alerts": [alert.__dict__ for alert in alerts],
    }
```

For production, keep tracker state per active camera stream. A global tracker is only acceptable for a single test stream.
