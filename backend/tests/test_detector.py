from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.config import Settings
from app.ml.detector import LocalObjectDetector


class _Tensor:
    def __init__(self, values: list) -> None:
        self._values = values

    def tolist(self) -> list:
        return self._values


class _Boxes:
    cls = _Tensor([1.0, 2.0, 2.0])
    conf = _Tensor([0.9, 0.8, 0.75])
    xyxy = _Tensor([[0, 0, 10, 10], [5, 5, 20, 20], [30, 30, 40, 40]])


class _Result:
    boxes = _Boxes()


class FakeYolo:
    names = {0: "bed", 1: "sofa", 2: "chair"}

    def __init__(self) -> None:
        self.predict_kwargs: dict = {}

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return [_Result()]


def _detector(image_size: int) -> tuple[LocalObjectDetector, FakeYolo]:
    detector = LocalObjectDetector(Path("unused.pt"), image_size=image_size)
    fake = FakeYolo()
    detector._model = fake
    return detector, fake


def test_default_inference_image_size_is_480():
    assert Settings().inference_image_size == 480


def test_detector_passes_image_size_and_keeps_instances():
    detector, fake = _detector(480)
    detections = detector.detect(Image.new("RGB", (64, 64)))

    assert fake.predict_kwargs["imgsz"] == 480
    assert [(d.label, d.confidence) for d in detections] == [("sofa", 0.9), ("chair", 0.8), ("chair", 0.75)]
    assert detections[2].box == (30.0, 30.0, 40.0, 40.0)
