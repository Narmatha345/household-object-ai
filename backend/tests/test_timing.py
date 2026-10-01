from __future__ import annotations

from pathlib import Path

import app.ml.model_loader as model_loader
from app.ml.detector import RawDetection

from .conftest import make_image_bytes


def detect(client):
    return client.post("/api/detect", files={"file": ("photo.jpg", make_image_bytes(), "image/jpeg")})


def test_local_detection_timing(client, detector, fallback):
    detector.detections = [RawDetection("chair", 0.91)]
    detector.delay_s = 0.05

    body = detect(client).json()
    timing = body["timing"]

    assert body["source"] == "local"
    assert fallback.calls == 0
    assert set(timing) == {"local_inference_ms", "local_processing_ms", "openai_ms", "total_ms"}
    assert timing["openai_ms"] is None
    # Measured, not guessed: the simulated 50 ms inference must show up.
    assert 50 <= timing["local_inference_ms"] < 1000
    assert 0 <= timing["local_processing_ms"] < 50
    assert timing["total_ms"] >= timing["local_inference_ms"] + timing["local_processing_ms"]


def test_fallback_timing(client, detector, fallback):
    detector.detections = [RawDetection("sofa", 0.40)]
    detector.delay_s = 0.02
    fallback.delay_s = 0.08

    body = detect(client).json()
    timing = body["timing"]

    assert body["source"] == "openai"
    assert body["fallback_used"] is True
    assert 20 <= timing["local_inference_ms"] < 1000
    assert 80 <= timing["openai_ms"] < 2000
    assert timing["total_ms"] >= timing["local_inference_ms"] + timing["openai_ms"]


def test_fallback_error_has_no_timing_but_stays_clean(client, detector, fallback):
    detector.detections = []
    fallback.fail = True

    response = detect(client)

    assert response.status_code == 502
    assert set(response.json()) == {"detail", "code"}


def test_existing_fields_unchanged(client, detector):
    detector.detections = [RawDetection("bed", 0.88)]

    body = detect(client).json()

    for field in ("source", "objects", "fallback_used", "fallback_reason", "description",
                  "reliability_note", "local_candidates", "confidence_threshold"):
        assert field in body
    assert body["confidence_threshold"] == 0.70


def test_model_is_loaded_only_once(monkeypatch, tmp_path: Path):
    loads = []

    class CountingYOLO:
        def __init__(self, path: str) -> None:
            loads.append(path)

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", CountingYOLO)
    monkeypatch.setattr(model_loader, "_cache", {})
    weights = tmp_path / "fake.pt"

    first = model_loader.load_yolo_model(weights)
    for _ in range(5):
        assert model_loader.load_yolo_model(weights) is first
    assert len(loads) == 1
