# Specialized household model – plan (Phases 1–4)

Status: **planning only.** No dataset has been downloaded, nothing has been
trained, and production still uses `models/yolo26n.pt`.

---

## Phase 1 – How the current system works

| Question | Answer (with location) |
|---|---|
| 1. How YOLO26n is loaded | `load_yolo_model()` in `backend/app/ml/model_loader.py` calls `ultralytics.YOLO(MODEL_PATH)` once and caches the result per path behind a lock. `create_app()` calls `LocalObjectDetector.load()` at startup, in the FastAPI lifespan. |
| 2. How inference runs | `LocalObjectDetector.detect()` (`backend/app/ml/detector.py`) calls `model.predict(image, conf=0.15)` under a thread lock. It returns `(label, confidence)` for every box, reading class names from the weights file (`result.names`). |
| 3. How classes are filtered | `LocalDetectionService` (`backend/app/services/local_detection.py`) drops labels that aren't in `HOUSEHOLD_LABELS`. The default is about 50 COCO classes defined in `backend/app/config.py`, and `*` disables the filter. |
| 4. How the 70% threshold applies | Same service. Detections with confidence ≥ `LOCAL_CONFIDENCE_THRESHOLD` (0.70) count as reliable. For each label only the highest-scoring box is kept. |
| 5. When the OpenAI fallback runs | `DetectionRouter.detect()` (`backend/app/services/detection_router.py`): if nothing is reliable, or the local model failed, it calls `OpenAIFallbackService.detect()` and passes the weak local guesses as hints. |
| 6. Existing training support | Only documentation (`docs/model-training.md`). |
| 7. Expected dataset format | Ultralytics YOLO detection format: one `.txt` per image, each line `class x_center y_center width height`, normalised to 0–1. |
| 8. Training scripts present? | **No.** Nothing yet downloads, converts, trains or evaluates. |
| 9. Can the model be swapped via `MODEL_PATH`? | **Yes.** Class names come from the weights file, so routing and the UI need no changes. |

Things the plan must handle:

1. **The allowlist hides new classes.** New class names such as
   `ceiling fan` are silently dropped by the default `HOUSEHOLD_LABELS`. When
   switching, set `HOUSEHOLD_LABELS=*` or list the new names.
2. **Fine-tuning replaces the class list.** The new model forgets any COCO
   class that isn't in `data.yaml`. The list keeps `person`, `cat` and `dog`.
   Otherwise selfies and pet photos would go to OpenAI and the fallback rate
   would rise.
3. **Hardware:** this laptop (i3-1215U, 7.7 GB RAM, no NVIDIA GPU) is fine for
   inference but impractical for training (days). Train on a free GPU (Google
   Colab or Kaggle T4), then copy `best.pt` back.
4. **Licensing:** Ultralytics is AGPL-3.0, while this repo's LICENSE says MIT.
   Review that before any commercial use.

---

## Phase 2 – Class list (v1: 43 classes)

Availability was checked against the class lists of COCO (80), Ultralytics
HomeObjects-3K (12), Open Images V7 (601 boxable), Objects365 (365) and LVIS
(1203). Open Images counts are boxes in its official **val + test** splits
(~167k images). Its train split is roughly 10× larger.

| id | Class | COCO | Open Images V7 source class(es) | OI val+test boxes | Source plan |
|---:|---|:-:|---|---:|---|
| 0 | bed | ✓ | Bed | 493 | COCO + OI |
| 1 | sofa | ✓ couch | Couch | 208 | COCO + OI + HomeObjects |
| 2 | chair | ✓ | Chair | 3,297 | COCO + OI |
| 3 | table | ✓ dining table | Table, Kitchen & dining room table, Coffee table, Desk | 5,426 | COCO + OI (merged) |
| 4 | tv | ✓ | Television | 189 | COCO + OI |
| 5 | monitor | – | Computer monitor | 390 | OI |
| 6 | laptop | ✓ | Laptop | 305 | COCO + OI |
| 7 | keyboard | ✓ | Computer keyboard | 318 | COCO + OI |
| 8 | mouse | ✓ | Computer mouse | 113 | COCO + OI |
| 9 | remote | ✓ | Remote control | 40 | COCO + OI |
| 10 | mobile phone | ✓ cell phone | Mobile phone | 610 | COCO + OI |
| 11 | refrigerator | ✓ | Refrigerator | 151 | COCO + OI |
| 12 | microwave | ✓ | Microwave oven | 107 | COCO + OI |
| 13 | washing machine | – | Washing machine | 163 | OI + **custom** |
| 14 | ceiling fan | – | Ceiling fan | 53 | OI + **custom** |
| 15 | fan | – | Mechanical fan | 98 | OI + custom |
| 16 | lamp | – | Lamp | 257 | OI + HomeObjects |
| 17 | cupboard | – | Cupboard, Cabinetry, Wardrobe, Closet | 1,457 | OI + HomeObjects + custom (steel almirah) |
| 18 | door | – | Door | 1,074 | OI + HomeObjects |
| 19 | window | – | Window | 33,140 | OI (capped and filtered) |
| 20 | mirror | – | Mirror | 148 | OI |
| 21 | clock | ✓ | Clock | 91 | COCO + OI |
| 22 | book | ✓ | Book | 3,129 | COCO + OI |
| 23 | bottle | ✓ | Bottle | 1,978 | COCO + OI |
| 24 | cup | ✓ | Mug, Coffee cup | 1,066 | COCO + OI |
| 25 | plate | – | Plate | 400 | OI |
| 26 | bowl | ✓ | Bowl | 289 | COCO + OI |
| 27 | pressure cooker | – | Pressure cooker | **2** | **Custom photos required** |
| 28 | dustbin | – | Waste container | 142 | OI + custom |
| 29 | backpack | ✓ | Backpack | 126 | COCO + OI |
| 30 | shoe | – | Footwear | 35,992 | OI (capped and filtered) |
| 31 | potted plant | ✓ | Houseplant, Flowerpot | 3,185 | COCO + OI + HomeObjects |
| 32 | photo frame | – | Picture frame | 279 | OI + HomeObjects |
| 33 | water drum | – | *(none suitable)* | – | **Custom photos required** |
| 34 | pillow | – | Pillow | 411 | OI |
| 35 | gas stove | – | Gas stove | 83 | OI + **custom** |
| 36 | sink | ✓ | Sink | 192 | COCO + OI |
| 37 | toilet | ✓ | Toilet | 103 | COCO + OI |
| 38 | mixer grinder | – | *(Blender/Mixer are different products)* | – | **Custom photos required** |
| 39 | bucket | – | *(not boxable in OI)* | – | **Custom photos required** |
| 40 | person | ✓ | Person | – | COCO (keeps current coverage) |
| 41 | cat | ✓ | Cat | – | COCO |
| 42 | dog | ✓ | Dog | – | COCO |

### Merged, deferred or renamed

| Requested | Decision | Reason |
|---|---|---|
| table + dining table | **Merged → `table`** | The same furniture in different rooms. As two classes they'd mostly swap labels. |
| cupboard + wardrobe | **Merged → `cupboard`** | Closed cabinets whose purpose can't be seen. Open Images' four related classes overlap. |
| desktop computer | **Deferred** | Only Objects365 has it ("Computer Box"), and that dataset is a single ~712 GB download. `monitor` + `keyboard` + `mouse` already cover a desk setup. |
| light | **Deferred** | Too vague (bulb, tube light, LED panel), and OI "Light bulb" has only 126 boxes. It could return as a custom `tube light` class. |
| standing fan | Renamed **`fan`** | OI "Mechanical fan" covers table, pedestal and wall fans. |
| plant | Renamed **`potted plant`** | Matches COCO and the current app output. |
| drum | Renamed **`water drum`** | OI "Drum" is the **musical instrument**, and "Barrel" is mostly wooden casks. |
| added | pillow, gas stove, sink, toilet, mixer grinder, bucket, person, cat, dog | Common in homes. Sink, toilet, person, cat and dog are detected today, so dropping them would be a regression. |
| dropped | COCO food classes (banana, pizza, …) | Not household objects here. Those photos would go to the fallback instead. |

**Confusion pairs to watch** (kept separate on purpose): `tv` ↔ `monitor`,
`mirror` ↔ `window` ↔ `photo frame`, `fan` ↔ `ceiling fan`,
`bucket` ↔ `dustbin` ↔ `water drum`. If a pair still gets confused after
training, merge it in v2.

---

## Phase 3 – Dataset strategy

| Dataset | Size | License | Verdict |
|---|---|---|---|
| **Open Images V7** | 1.9M images; can be downloaded per class with FiftyOne | Annotations CC BY 4.0; images listed as CC BY 2.0, but verify each image | **Primary.** Covers 35 of the 43 classes. |
| **COCO 2017** | 118k train images | Annotations CC BY 4.0; images under Flickr terms | Keeps the existing COCO classes. Use only images with our classes. |
| **HomeObjects-3K** | 2,689 indoor images, 390 MB | AGPL-3.0 | Supplement for furniture, doors and windows. Spot-check it first, because [issue #21037](https://github.com/ultralytics/ultralytics/issues/21037) reports missing labels. |
| Objects365 | 712 GB, no per-class download | Annotations CC BY 4.0; images under Flickr terms, no redistribution | Not in v1. |
| LVIS | COCO images | Annotations CC BY 4.0 | Not in v1. |

### Filtering rules

- **Cap each class at about 1,500 training boxes**, so `window` and `shoe`
  don't swamp rare classes.
- **`shoe`:** prefer footwear that isn't on a person, such as floors and racks.
- **`window`:** prefer images that also contain an indoor class.
- **Drop two kinds of Open Images boxes:** those flagged `IsDepiction`
  (drawings or photos of objects) and `IsGroupOf` (one box covering many
  objects).
- **Label every class in every image.** If an image fetched for `sofa` also
  shows a chair, the chair must be labelled too. Otherwise the model learns
  that visible chairs are background.

### Custom photos (real photos only; synthetic images may only supplement)

| Class | Why | Target (train / val) |
|---|---|---|
| pressure cooker | 2 OI boxes, mostly electric models | 300 / 60 |
| water drum | No matching public class | 300 / 60 |
| mixer grinder | No matching public class | 300 / 60 |
| bucket | Not boxable in OI | 300 / 60 |
| gas stove | OI is mostly Western ranges | +200 / 40 |
| washing machine | Add Indian top-load and semi-automatic models | +150 / 30 |
| ceiling fan | Only 53 OI boxes | +150 / 30 |
| cupboard | Add steel almirahs | +150 / 30 |

Take photos in several homes, rooms, lighting conditions and angles, and keep
burst shots in the same split. Label all 43 classes in each photo. Label Studio
and CVAT are free and export YOLO format. Later, the images that trigger the
OpenAI fallback, saved with consent, would be the best new training data.

**Planned size:** roughly 15–25k images, including about 2,200 custom photos.

---

## Phase 4 – Dataset structure

```
data/household/
├── data.yaml            # 43 classes, train/val paths, nc
├── source_mapping.yaml  # which COCO / Open Images classes feed each class
├── images/{train,val}/
└── labels/{train,val}/
```

`data.yaml` has **no `path:` key** on purpose. Ultralytics then resolves
`train`/`val` relative to the YAML file's own folder, so training works from any
working directory. Image and label files are git-ignored.

---

## Next phases (not started)

5. Build the dataset: download and convert the mapped COCO and Open Images
   classes, then import the custom photos.
6. Train YOLO26n from the COCO weights on a GPU. Compare it with the current
   model on the same held-out household photos: per-class mAP, confusion
   pairs, and the fallback rate at 70%.
7. Only if it's better: save it as `models/household-yolo26n.pt`, set
   `MODEL_PATH` and `HOUSEHOLD_LABELS=*`, and re-tune the threshold. The
   current `yolo26n.pt` stays as the rollback.
