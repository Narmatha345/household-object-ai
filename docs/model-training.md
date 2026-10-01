# Training a custom household-object model

The MVP uses the pretrained **YOLO26n** model (COCO, 80 classes). About 50 of
those classes are household objects (chair, couch, bed, tv, refrigerator, cup…),
but COCO has no classes for many everyday items: ceiling fan, washing machine,
pressure cooker, mixer grinder, air conditioner, wardrobe, pillow, and so on.
Right now all of those go to the OpenAI fallback.

Fine-tuning YOLO26n on your own household dataset moves those cases to the
local model and lowers the fallback rate.

## 1. Collect images the local model gets wrong

The best training data is the images that currently trigger the fallback. A
simple future improvement is to save those images (with the OpenAI labels as
a starting point) into `data/raw/`. Also add your own photos of each target
class from different rooms, angles, distances and lighting.

## 2. Label in YOLO format

Use Label Studio, CVAT, or Roboflow and export in YOLO format into
`data/household/` (layout in [data/README.md](../data/README.md)). Class IDs
must match the `names:` list in
[data/household/data.yaml](../data/household/data.yaml). The v1 class list and
data sources are explained in [household-model-plan.md](household-model-plan.md).

Tips:
- Keep the COCO household classes you still need in the class list, and label
  them in your images too. Otherwise the model can "forget" them.
- Aim for 150–300+ labeled instances per class to start.
- Include some background images with no objects to reduce false positives.

## 3. Fine-tune from the pretrained weights

From the `backend/` folder with the virtual environment active:

```bash
yolo detect train \
  model=models/yolo26n.pt \
  data=../data/household/data.yaml \
  epochs=100 imgsz=640 batch=16 patience=20 \
  project=runs name=household-yolo26n
```

Or in Python:

```python
from ultralytics import YOLO

model = YOLO("models/yolo26n.pt")  # start from pretrained COCO weights
model.train(data="../data/household/data.yaml", epochs=100, imgsz=640, batch=16, patience=20,
            project="runs", name="household-yolo26n")
```

A CUDA GPU makes this much faster (`device=0`). On CPU, start with fewer
epochs or a smaller `imgsz` to check that the pipeline works.

**No local GPU?** Use [notebooks/train_household_yolo26n_colab.ipynb](../notebooks/train_household_yolo26n_colab.ipynb)
on a free Colab T4. Package the dataset with
`backend\.venv\Scripts\python.exe tools\dataset\07_package_for_colab.py`, upload the zip to
`My Drive/household-object-ai/`, and run the cells in order. Note that with a relative `project=runs`,
Ultralytics 8.4 saves to `runs/detect/runs/<name>`. Pass an absolute `project` path to get `runs/<name>`.

## 4. Evaluate

```bash
yolo detect val model=runs/household-yolo26n/weights/best.pt data=../data/household/data.yaml
```

Check mAP50-95 and per-class precision/recall. Look closely at classes with
low recall. Those are the ones that will keep falling back to OpenAI.

## 5. Deploy the new weights

```bash
cp runs/household-yolo26n/weights/best.pt models/household-yolo26n.pt
```

Then in `backend/.env`:

```env
MODEL_PATH=models/household-yolo26n.pt
# Make the new class names count as household objects
# ("*" accepts every class the model knows):
HOUSEHOLD_LABELS=*
```

Restart the backend. No routing or frontend changes are needed: the detector
reads class names from the weights file.

## 6. Re-tune the threshold

A fine-tuned model often has a different confidence distribution. Run a
held-out set of house photos through `/api/detect`, then pick the
`LOCAL_CONFIDENCE_THRESHOLD` that balances the fallback rate (cost) against
wrong local answers (accuracy).

## Optional: faster CPU inference

Export to ONNX or OpenVINO for faster inference on CPU-only machines:

```bash
yolo export model=models/household-yolo26n.pt format=onnx
```

Then set `MODEL_PATH=models/household-yolo26n.onnx`. Ultralytics loads exported
formats through the same `YOLO(...)` call.
