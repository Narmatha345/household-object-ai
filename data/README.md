# Data

This folder holds the dataset used to fine-tune a custom household-object model.
Image and label files are git-ignored; only this README and `dataset.yaml` are tracked.

```
data/
├── raw/          # Original photos, unlabeled, any folder structure
├── processed/    # YOLO-format dataset, ready for training
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/   # one .txt per image, same file name
│       ├── val/
│       └── test/
└── dataset.yaml  # class names + split paths
```

## Label format

One line per object in `labels/<split>/<image-name>.txt`:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to 0–1. `class_id` must match `names` in `dataset.yaml`.

## Collecting data

- Good sources: your own house photos, and images the app sent to the OpenAI
  fallback (those are exactly the cases the local model is weak on).
- Aim for at least 150–300 labeled instances per class to start, taken in
  varied lighting, angles and rooms.
- Split roughly 80 / 15 / 5 into train / val / test, keeping near-duplicate
  photos in the same split.
- Labeling tools: Label Studio, CVAT, or Roboflow (export as "YOLO").

See [docs/model-training.md](../docs/model-training.md) for the training steps.
