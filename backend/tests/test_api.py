from __future__ import annotations

from app.ml.detector import RawDetection

from .conftest import make_image_bytes


def upload(client, data: bytes, filename: str = "photo.jpg", content_type: str = "image/jpeg"):
    return client.post("/api/detect", files={"file": (filename, data, content_type)})


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["local_model_loaded"] is True


def test_valid_image_upload(client, detector):
    detector.detections = [RawDetection("chair", 0.91)]
    response = upload(client, make_image_bytes("PNG"), "chair.png", "image/png")
    assert response.status_code == 200
    assert detector.calls == 1


def test_reliable_local_detection_does_not_call_openai(client, detector, fallback):
    detector.detections = [
        RawDetection("chair", 0.91),
        RawDetection("chair", 0.80),
        RawDetection("bottle", 0.75),
        RawDetection("laptop", 0.40),  # below threshold, ignored
    ]
    response = upload(client, make_image_bytes())
    body = response.json()

    assert response.status_code == 200
    assert body["source"] == "local"
    assert body["fallback_used"] is False
    assert body["objects"] == [
        {"label": "chair", "confidence": 0.91, "box": None},
        {"label": "chair", "confidence": 0.80, "box": None},  # separate instance, not merged
        {"label": "bottle", "confidence": 0.75, "box": None},
    ]
    assert fallback.calls == 0


def test_all_local_detections_above_threshold_are_returned(client, detector, fallback):
    detector.detections = [
        RawDetection("table", 0.90, (10.0, 20.0, 300.0, 200.0)),
        RawDetection("sofa", 0.87),
        RawDetection("ceiling fan", 0.72),
        RawDetection("chair", 0.81),
        RawDetection("bed", 0.76),
        RawDetection("cupboard", 0.68),  # below 70%, excluded
    ]
    body = upload(client, make_image_bytes()).json()

    assert body["source"] == "local"
    assert [(o["label"], o["confidence"]) for o in body["objects"]] == [
        ("table", 0.90),
        ("sofa", 0.87),
        ("chair", 0.81),
        ("bed", 0.76),
        ("ceiling fan", 0.72),
    ]
    assert body["objects"][0]["box"] == [10.0, 20.0, 300.0, 200.0]
    assert fallback.calls == 0


def test_low_confidence_triggers_fallback(client, detector, fallback):
    detector.detections = [RawDetection("sofa", 0.42)]
    response = upload(client, make_image_bytes())
    body = response.json()

    assert response.status_code == 200
    assert body["source"] == "openai"
    assert body["fallback_used"] is True
    assert "below the 70% threshold" in body["fallback_reason"]
    assert body["objects"][0]["label"] == "ceiling fan"
    assert body["local_candidates"] == [{"label": "sofa", "confidence": 0.42, "box": None}]
    assert fallback.calls == 1
    assert fallback.last_hints == ["sofa"]
    # Local model always runs first.
    assert detector.calls == 1


def test_no_detection_triggers_fallback(client, detector, fallback):
    detector.detections = []
    body = upload(client, make_image_bytes()).json()
    assert body["fallback_used"] is True
    assert "no household object" in body["fallback_reason"]
    assert fallback.calls == 1


def test_non_household_label_is_not_reliable(client, detector, fallback):
    detector.detections = [RawDetection("airplane", 0.95)]
    body = upload(client, make_image_bytes()).json()
    assert body["source"] == "openai"
    assert fallback.calls == 1


def test_labels_outside_trained_classes_are_never_returned_locally(client, detector, fallback):
    detector.detections = [
        RawDetection("refrigerator", 0.92),
        RawDetection("couch", 0.88),
        RawDetection("potted plant", 0.70),
        RawDetection("sofa", 0.86),
    ]
    body = upload(client, make_image_bytes()).json()

    assert body["source"] == "local"
    assert [o["label"] for o in body["objects"]] == ["refrigerator", "sofa"]
    assert fallback.calls == 0


def test_openai_fallback_failure_returns_clean_error(client, detector, fallback):
    detector.detections = [RawDetection("tv", 0.30)]
    fallback.fail = True
    response = upload(client, make_image_bytes())
    body = response.json()

    assert response.status_code == 502
    assert body["code"] == "fallback_failed"
    assert "Traceback" not in body["detail"]
    assert "test-key" not in response.text


def test_invalid_file_type(client, detector):
    response = upload(client, b"hello world", "notes.txt", "text/plain")
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_file_type"
    assert detector.calls == 0


def test_empty_upload(client):
    response = upload(client, b"")
    assert response.status_code == 400
    assert response.json()["code"] == "empty_upload"


def test_corrupt_image(client):
    response = upload(client, b"\xff\xd8\xff not really a jpeg")
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_image"


def test_missing_file_field(client):
    response = client.post("/api/detect")
    assert response.status_code == 400


def test_stats_track_local_and_fallback(client, detector):
    detector.detections = [RawDetection("chair", 0.9)]
    upload(client, make_image_bytes())
    upload(client, make_image_bytes())
    upload(client, make_image_bytes())
    detector.detections = [RawDetection("chair", 0.2)]
    upload(client, make_image_bytes())

    stats = client.get("/api/stats").json()
    assert stats["total_requests"] == 4
    assert stats["local_detections"] == 3
    assert stats["openai_fallbacks"] == 1
    assert stats["fallback_rate"] == 0.25
