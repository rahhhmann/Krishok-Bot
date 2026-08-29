"""Unit tests for vision/detector.py.

ultralytics.YOLO is fully mocked -- no real model weights are loaded and
no real inference runs. These tests verify detector.py's own logic:
confidence-tier thresholds, sorting, and the pending-review escalation
log, not YOLO's detection quality.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

import vision.detector as detector_module
from vision.detector import CropDiseaseDetector, _tier


# --- _tier thresholds ------------------------------------------------------


@pytest.mark.parametrize(
    "confidence, expected_tier",
    [
        (0.95, "green"),
        (0.85, "green"),  # boundary: >= GREEN_THRESHOLD
        (0.849, "amber"),
        (0.70, "amber"),
        (0.60, "amber"),  # boundary: >= AMBER_THRESHOLD
        (0.599, "red"),
        (0.10, "red"),
        (0.0, "red"),
    ],
)
def test_tier_thresholds(confidence, expected_tier):
    assert _tier(confidence) == expected_tier


# --- CropDiseaseDetector.__init__ -------------------------------------------


def test_detector_init_raises_if_weights_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.pt"
    with patch.object(detector_module, "YOLO"):
        with pytest.raises(FileNotFoundError):
            CropDiseaseDetector(weights_path=missing_path)


def test_detector_init_loads_model_when_weights_exist(tmp_path):
    weights = tmp_path / "fake_weights.pt"
    weights.write_bytes(b"not a real model, just needs to exist")
    with patch.object(detector_module, "YOLO") as mock_yolo_cls:
        mock_yolo_cls.return_value = MagicMock()
        det = CropDiseaseDetector(weights_path=weights)
    mock_yolo_cls.assert_called_once_with(str(weights))
    assert det.model is mock_yolo_cls.return_value


# --- predict(): detection parsing, sorting, tiering -------------------------


def _make_fake_box(cls_id: int, confidence: float, bbox: list[float]) -> MagicMock:
    box = MagicMock()
    box.cls = [cls_id]
    box.conf = [confidence]
    box.xyxy = [MagicMock(tolist=MagicMock(return_value=bbox))]
    return box


def _make_fake_result(boxes: list[MagicMock]) -> MagicMock:
    result = MagicMock()
    result.boxes = boxes
    return result


def test_predict_parses_detections_into_expected_dict_shape(tmp_path):
    weights = tmp_path / "fake_weights.pt"
    weights.write_bytes(b"stub")
    with patch.object(detector_module, "YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.names = {0: "potato_late_blight", 1: "potato_healthy"}
        box = _make_fake_box(0, 0.92, [10.0, 20.0, 100.0, 200.0])
        mock_model.predict.return_value = [_make_fake_result([box])]
        mock_yolo_cls.return_value = mock_model

        det = CropDiseaseDetector(weights_path=weights)
        detections = det.predict("fake_image.jpg", log_low_confidence=False)

    assert len(detections) == 1
    d = detections[0]
    assert d["class_name"] == "potato_late_blight"
    assert d["confidence"] == pytest.approx(0.92)
    assert d["tier"] == "green"
    assert d["bbox"] == [10.0, 20.0, 100.0, 200.0]


def test_module_level_predict_sorts_by_confidence_descending(tmp_path):
    weights = tmp_path / "fake_weights.pt"
    weights.write_bytes(b"stub")
    with patch.object(detector_module, "YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.names = {0: "rice_blast", 1: "rice_healthy"}
        low_box = _make_fake_box(1, 0.30, [0, 0, 1, 1])
        high_box = _make_fake_box(0, 0.91, [0, 0, 1, 1])
        mock_model.predict.return_value = [_make_fake_result([low_box, high_box])]
        mock_yolo_cls.return_value = mock_model

        # Reset the module-level singleton so this test gets our mocked model.
        detector_module._detector_singleton = None
        detections = detector_module.predict(
            "fake_image.jpg", image_ref="test", log_low_confidence=False
        )

    assert [d["confidence"] for d in detections] == [pytest.approx(0.91), pytest.approx(0.30)]
    detector_module._detector_singleton = None  # don't leak into other tests


def test_predict_logs_red_tier_to_pending_review(tmp_path, monkeypatch):
    weights = tmp_path / "fake_weights.pt"
    weights.write_bytes(b"stub")
    pending_path = tmp_path / "pending_review.jsonl"
    monkeypatch.setattr(detector_module, "PENDING_REVIEW_PATH", pending_path)

    with patch.object(detector_module, "YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.names = {0: "tomato_early_blight"}
        red_box = _make_fake_box(0, 0.35, [0, 0, 1, 1])
        mock_model.predict.return_value = [_make_fake_result([red_box])]
        mock_yolo_cls.return_value = mock_model

        det = CropDiseaseDetector(weights_path=weights)
        det.predict("fake_image.jpg", image_ref="img_001", log_low_confidence=True)

    assert pending_path.exists()
    lines = pending_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["image_ref"] == "img_001"
    assert entry["reason_flagged"] == "red_tier_confidence_below_0.60"
    assert entry["model_output"]["tier"] == "red"


def test_predict_does_not_log_when_log_low_confidence_false(tmp_path, monkeypatch):
    weights = tmp_path / "fake_weights.pt"
    weights.write_bytes(b"stub")
    pending_path = tmp_path / "pending_review.jsonl"
    monkeypatch.setattr(detector_module, "PENDING_REVIEW_PATH", pending_path)

    with patch.object(detector_module, "YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.names = {0: "tomato_early_blight"}
        red_box = _make_fake_box(0, 0.35, [0, 0, 1, 1])
        mock_model.predict.return_value = [_make_fake_result([red_box])]
        mock_yolo_cls.return_value = mock_model

        det = CropDiseaseDetector(weights_path=weights)
        det.predict("fake_image.jpg", log_low_confidence=False)

    assert not pending_path.exists()


def test_predict_no_detections_returns_empty_list(tmp_path):
    weights = tmp_path / "fake_weights.pt"
    weights.write_bytes(b"stub")
    with patch.object(detector_module, "YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.names = {}
        mock_model.predict.return_value = [_make_fake_result([])]
        mock_yolo_cls.return_value = mock_model

        det = CropDiseaseDetector(weights_path=weights)
        detections = det.predict("fake_image.jpg", log_low_confidence=False)

    assert detections == []
