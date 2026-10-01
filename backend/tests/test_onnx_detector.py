from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import BACKEND_DIR, DEFAULT_HOUSEHOLD_LABELS, Settings
from app.main import create_app
from app.ml.detector import LocalObjectDetector
from app.ml.onnx_detector import OnnxObjectDetector, _nms

ONNX_MODEL = BACKEND_DIR / "models" / "household-yolo26n.onnx"
PT_MODEL = BACKEND_DIR / "models" / "household-yolo26n.pt"
LIVING_ROOM = BACKEND_DIR.parent / "data" / "household" / "images" / "val" / "oi_1beafa4064d0ae71.jpg"


def test_nms_keeps_best_box_and_separate_objects():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert _nms(boxes, scores, 0.7) == [0, 2]


def test_postprocess_keeps_instances_and_maps_boxes_back():
    detector = OnnxObjectDetector(Path("unused.onnx"))
    detector._names = {0: "bed", 1: "sofa", 2: "chair"}
    # 3 anchors, rows = cx, cy, w, h, then one score per class.
    raw = np.array([
        [20, 100, 200],   # cx
        [20, 100, 200],   # cy
        [10, 10, 10],     # w
        [10, 10, 10],     # h
        [0.0, 0.0, 0.1],  # bed
        [0.9, 0.0, 0.0],  # sofa
        [0.0, 0.8, 0.05], # chair
    ], dtype=np.float32)[None]
    # Letterboxed with ratio 0.5 and 4 px left padding.
    detections = detector._postprocess(raw, ratio=0.5, pad=(4, 0), image_size=(1000, 1000))

    assert [(d.label, round(d.confidence, 2)) for d in detections] == [("sofa", 0.9), ("chair", 0.8)]
    assert detections[0].box == (22.0, 30.0, 42.0, 50.0)


@pytest.mark.skipif(not ONNX_MODEL.exists(), reason="ONNX model not exported")
def test_real_onnx_model_uses_its_own_class_names():
    detector = OnnxObjectDetector(ONNX_MODEL, image_size=480, threads=1)
    detector.load()
    assert tuple(detector.class_names.values()) == DEFAULT_HOUSEHOLD_LABELS


@pytest.mark.skipif(not (ONNX_MODEL.exists() and LIVING_ROOM.exists()), reason="model or test image missing")
def test_real_onnx_model_detects_living_room():
    detector = OnnxObjectDetector(ONNX_MODEL, image_size=480, threads=1)
    detector.load()
    confident = [d for d in detector.detect(Image.open(LIVING_ROOM).convert("RGB")) if d.confidence >= 0.70]

    assert sorted(d.label for d in confident) == ["chair", "chair", "chair", "sofa", "table"]
    assert all(d.box is not None for d in confident)


@pytest.mark.skipif(not PT_MODEL.exists(), reason="PyTorch model missing")
def test_falls_back_to_pytorch_model_when_onnx_fails(fallback):
    settings = Settings(
        model_path=BACKEND_DIR / "models" / "missing.onnx",
        fallback_model_path=PT_MODEL,
        openai_api_key="test-key",
    )
    app = create_app(settings=settings, fallback=fallback)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["local_model_loaded"] is True
        assert isinstance(app.state.local_service._detector, LocalObjectDetector)
