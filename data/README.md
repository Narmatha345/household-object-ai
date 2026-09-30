# Data

Datasets for training the specialized household model. Image and label files
are git-ignored; only the YAML configs and folder placeholders are tracked.

```
data/
├── raw/                    # Unlabelled photos you collect (any structure)
└── household/              # YOLO-format training dataset (v1, 43 classes)
    ├── data.yaml           # class names, nc, train/val paths
    ├── source_mapping.yaml # which COCO / Open Images classes feed each class
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/          # one .txt per image, same file name
        └── val/
```

The class list, dataset sources and collection targets are explained in
[docs/household-model-plan.md](../docs/household-model-plan.md).

## Label format

One line per object in `labels/<split>/<image-name>.txt`:

```
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates are normalised to 0–1, and `class_id` must match `names` in
`household/data.yaml`. **Label every one of the 43 classes that appears in
an image**, not only the object the photo was taken for.

## Custom photos

These classes need your own real photos: pressure cooker, water drum, mixer
grinder and bucket. Gas stove, washing machine, ceiling fan and cupboard
(steel almirah) need extra photos. Targets per class are listed in the plan.

- Vary the house, room, lighting, angle and distance. Include clutter and
  partial views.
- Keep near-duplicate or burst photos in the same split (about 80% train,
  20% val).
- Label with Label Studio, CVAT or Roboflow, and export in YOLO format.
