"""ONNX Runtime detector for the exported household YOLO model.

Produces the same RawDetection output as LocalObjectDetector, but runs on
onnxruntime + numpy only, so torch / ultralytics are never imported. That keeps
memory far lower, which matters on small CPU hosts such as Render Free.

Pre- and post-processing mirror Ultralytics' predict path for this model:
letterbox (keep aspect ratio, pad to a multiple of the stride with value 114),
then class-aware NMS (IoU 0.7, max 300 boxes) on the raw (4 + classes, anchors) output.
"""

from __future__ import annotations

import ast
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.errors import LocalInferenceError, ModelLoadError
from app.ml.detector import RawDetection

logger = logging.getLogger(__name__)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS on xyxy boxes; returns kept indices, highest score first."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        w = (np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])).clip(0)
        h = (np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


class OnnxObjectDetector:
    """YOLO (ONNX export) detector. Loads the session once and reuses it for every request."""

    # Same floor as LocalObjectDetector; the reliability threshold is applied later.
    _MIN_CANDIDATE_CONFIDENCE = 0.15
    _IOU_THRESHOLD = 0.7
    _MAX_DETECTIONS = 300
    _PAD_VALUE = 114
    _MAX_WH = 7680  # class offset for class-aware NMS, as in Ultralytics

    def __init__(self, model_path: Path, image_size: int = 480, threads: int | None = None) -> None:
        self._model_path = model_path
        self._image_size = image_size
        self._threads = threads
        self._session = None
        self._input_name = ""
        self._names: dict[int, str] = {}
        self._stride = 32
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    def load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.inter_op_num_threads = 1
            if self._threads:
                options.intra_op_num_threads = self._threads
                cv2.setNumThreads(self._threads)  # resize threads; avoid oversubscribing a tiny CPU
            session = ort.InferenceSession(
                str(self._model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            metadata = session.get_modelmeta().custom_metadata_map
            # Class names come from the model file itself (Ultralytics export metadata).
            names = {int(k): str(v) for k, v in ast.literal_eval(metadata["names"]).items()}
            self._stride = int(ast.literal_eval(metadata.get("stride", "32")))
        except Exception as exc:
            logger.exception("Failed to load ONNX model from %s", self._model_path)
            raise ModelLoadError("The local detection model could not be loaded.") from exc

        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._names = names
        logger.info(
            "Local ONNX model loaded from %s with %d classes (imgsz=%d, threads=%s): %s",
            self._model_path, len(names), self._image_size, self._threads or "default", names,
        )

    def _letterbox(self, image: Image.Image) -> tuple[np.ndarray, float, tuple[int, int]]:
        width, height = image.size
        size = self._image_size
        ratio = min(size / height, size / width)
        new_w, new_h = round(width * ratio), round(height * ratio)
        # Minimum rectangle: pad only up to the next multiple of the stride.
        pad_w, pad_h = (size - new_w) % self._stride / 2, (size - new_h) % self._stride / 2
        top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
        left, right = round(pad_w - 0.1), round(pad_w + 0.1)

        pixels = np.asarray(image, dtype=np.uint8)
        if (new_w, new_h) != (width, height):
            # cv2 INTER_LINEAR matches Ultralytics exactly; PIL's bilinear antialiases when
            # downscaling, which shifts confidences by a few points.
            pixels = cv2.resize(pixels, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((new_h + top + bottom, new_w + left + right, 3), self._PAD_VALUE, dtype=np.uint8)
        canvas[top : top + new_h, left : left + new_w] = pixels
        tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor), ratio, (left, top)

    def _postprocess(
        self, output: np.ndarray, ratio: float, pad: tuple[int, int], image_size: tuple[int, int]
    ) -> list[RawDetection]:
        pred = output[0].T  # (anchors, 4 + classes)
        scores = pred[:, 4:]
        class_ids = scores.argmax(1)
        confidences = scores[np.arange(len(scores)), class_ids]
        mask = confidences > self._MIN_CANDIDATE_CONFIDENCE
        if not mask.any():
            return []
        cxcywh, confidences, class_ids = pred[mask, :4], confidences[mask], class_ids[mask]

        boxes = np.empty_like(cxcywh)
        boxes[:, :2] = cxcywh[:, :2] - cxcywh[:, 2:] / 2
        boxes[:, 2:] = cxcywh[:, :2] + cxcywh[:, 2:] / 2

        keep = _nms(boxes + (class_ids * self._MAX_WH)[:, None], confidences, self._IOU_THRESHOLD)
        keep = keep[: self._MAX_DETECTIONS]

        width, height = image_size
        boxes = boxes[keep]
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad[0]) / ratio).clip(0, width)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad[1]) / ratio).clip(0, height)

        detections = []
        for (x1, y1, x2, y2), conf, cls_id in zip(boxes.tolist(), confidences[keep].tolist(), class_ids[keep].tolist()):
            detections.append(
                RawDetection(
                    label=self._names[int(cls_id)],
                    confidence=float(conf),
                    box=(round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
                )
            )
        return detections

    def detect(self, image: Image.Image) -> list[RawDetection]:
        if self._session is None:
            raise ModelLoadError("The local detection model is not loaded.")

        try:
            with self._lock:
                start = time.perf_counter()
                tensor, ratio, pad = self._letterbox(image)
                preprocessed = time.perf_counter()
                output = self._session.run(None, {self._input_name: tensor})[0]
                inferred = time.perf_counter()
                detections = self._postprocess(output, ratio, pad, image.size)
                done = time.perf_counter()
        except Exception as exc:
            logger.exception("Local ONNX inference failed")
            raise LocalInferenceError("Local detection failed for this image.") from exc

        logger.info(
            "ONNX timing: preprocess_ms=%.1f inference_ms=%.1f postprocess_ms=%.1f input=%dx%d",
            (preprocessed - start) * 1000, (inferred - preprocessed) * 1000, (done - inferred) * 1000,
            tensor.shape[3], tensor.shape[2],
        )
        logger.info(
            "YOLO returned %d detection(s): %s",
            len(detections),
            ", ".join(f"{d.label} {d.confidence:.2f}" for d in detections) or "none",
        )
        return detections
