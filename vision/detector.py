"""
vision/detector.py — KrishokBot crop-disease YOLO inference wrapper.

Mirrors the confidence-tier pattern used in the BD License Plate Detector:
    green  >= 0.85   high confidence
    amber  0.60-0.85 medium confidence
    red    <  0.60   low confidence -> logged to eval/results/pending_review.jsonl
                      as an escalation-queue hook (no active human reviewer in
                      this version -- see krishokbot_addendum_plan.md Section 5).
"""
import json
import time
from pathlib import Path

from ultralytics import YOLO

WEIGHTS_PATH = Path(__file__).parent / "weights" / "crop_disease_best.pt"
PENDING_REVIEW_PATH = Path(__file__).parent.parent / "eval" / "results" / "pending_review.jsonl"

GREEN_THRESHOLD = 0.85
AMBER_THRESHOLD = 0.60


def _tier(confidence: float) -> str:
    if confidence >= GREEN_THRESHOLD:
        return "green"
    if confidence >= AMBER_THRESHOLD:
        return "amber"
    return "red"


class CropDiseaseDetector:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        if not weights_path.exists():
            raise FileNotFoundError(
                f"{weights_path} not found -- place crop_disease_best.pt "
                f"in vision/weights/ before using this detector."
            )
        self.model = YOLO(str(weights_path))

    def predict(self, image, image_ref: str = "unknown", log_low_confidence: bool = True):
        """
        image: path, PIL Image, or numpy array (anything ultralytics.YOLO.predict accepts)
        image_ref: identifier for the image, used only in the pending-review log

        Returns a list of dicts:
            {"class_name": str, "confidence": float, "tier": "green"|"amber"|"red", "bbox": [x1,y1,x2,y2]}
        """
        results = self.model.predict(image, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detections.append({
                    "class_name": self.model.names[cls_id],
                    "confidence": confidence,
                    "tier": _tier(confidence),
                    "bbox": [float(v) for v in box.xyxy[0].tolist()],
                })

        if log_low_confidence:
            for det in detections:
                if det["tier"] == "red":
                    self._log_pending_review(image_ref, det)

        return detections

    def _log_pending_review(self, image_ref: str, detection: dict):
        PENDING_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.time(),
            "input_type": "image",
            "image_ref": image_ref,
            "model_output": detection,
            "confidence": detection["confidence"],
            "reason_flagged": "red_tier_confidence_below_0.60",
        }
        with open(PENDING_REVIEW_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


_detector_singleton: "CropDiseaseDetector | None" = None


def get_detector() -> "CropDiseaseDetector":
    """Lazy singleton -- avoids reloading the YOLO model on every call."""
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = CropDiseaseDetector()
    return _detector_singleton


def predict(image, image_ref: str = "unknown", log_low_confidence: bool = True) -> list[dict]:
    """Module-level entry point used by agent/graph.py::vision_node.

    graph.py does `from vision.detector import predict; predict(image_path)`
    and expects detections sorted by confidence descending (it takes
    detections[0] as the "top" result).
    """
    detections = get_detector().predict(image, image_ref=image_ref, log_low_confidence=log_low_confidence)
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if image_path:
        for det in predict(image_path, image_ref=image_path):
            print(det)
