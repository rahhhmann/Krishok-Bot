"""eval/vision_eval.py

Two independent things, per Section 24/25 -- do not conflate them:

1. STANDALONE DEPLOYMENT CHECK (--check-deploy, default)
   Loads crop_disease_best.pt OUTSIDE the training notebook, through
   vision.detector, exactly as the Streamlit app will. Confirms the
   .pt file exists, loads, and returns valid detections on a real
   image. This is a pass/fail smoke test, not a metrics run -- the
   formal mAP/precision/recall numbers already live in
   vision/output/eval_report.json (produced by YOLO's own val() during
   training) and are NOT recomputed here to avoid two competing
   sources of truth. This script prints those numbers alongside the
   deployment check for a single combined report.

2. QUALITATIVE EXTERNAL TEST (--external-dir path/)
   Runs inference on images OUTSIDE the train/val/test split (e.g.
   photos you took or downloaded separately) and records predictions
   for manual review. Per Section 24: these are NEVER added to
   training or folded into the formal metrics -- output is written to
   eval/results/vision_qualitative.json for you to eyeball, not scored
   automatically, since there's no ground-truth label for images
   outside the dataset.

Run:
    python -m eval.vision_eval
    python -m eval.vision_eval --external-dir /path/to/some/photos
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
VISION_OUTPUT_DIR = EVAL_DIR.parent / "vision" / "output"
EVAL_REPORT_PATH = VISION_OUTPUT_DIR / "eval_report.json"
WEIGHTS_PATH = EVAL_DIR.parent / "vision" / "weights" / "crop_disease_best.pt"


def print_training_metrics() -> dict | None:
    """Prints the ALREADY-MEASURED metrics from eval_report.json (produced
    during training/val, see krishok-bot-vision-pipeline.ipynb). Does not
    recompute or alter them -- this is the single source of truth for
    formal vision metrics per Section 23."""
    if not EVAL_REPORT_PATH.exists():
        logger.warning(
            "%s not found -- no formal metrics to report. Run the training "
            "notebook's val() step first.", EVAL_REPORT_PATH
        )
        return None

    with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    overall = report.get("overall", {})
    print("\n=== Formal metrics (from vision/output/eval_report.json) ===")
    print(f"  mAP50:     {overall.get('mAP50')}")
    print(f"  mAP50-95:  {overall.get('mAP50-95')}")
    print(f"  precision: {overall.get('precision')}")
    print(f"  recall:    {overall.get('recall')}")
    print("\n  Per-class AP50 (lowest first -- worst performers surfaced, not hidden):")
    per_class = report.get("per_class", {})
    for cls, m in sorted(per_class.items(), key=lambda kv: kv[1].get("ap50", 0)):
        print(f"    {cls:28s} ap50={m.get('ap50'):.4f}  precision={m.get('precision'):.4f}  recall={m.get('recall'):.4f}")

    return report


def check_deployment_readiness(test_image: str | None) -> bool:
    """Loads the model exactly as the Streamlit app would (via
    vision.detector, not the training notebook) and runs one real
    inference. Returns True only if this actually succeeds."""
    print("\n=== Standalone deployment check ===")

    if not WEIGHTS_PATH.exists():
        print(f"FAIL: weights not found at {WEIGHTS_PATH}")
        return False
    print(f"OK: weights file present ({WEIGHTS_PATH.stat().st_size / 1_000_000:.1f} MB)")

    try:
        from vision.detector import get_detector
    except ImportError as exc:
        print(f"FAIL: could not import vision.detector: {exc}")
        return False

    try:
        start = time.monotonic()
        detector = get_detector()
        load_seconds = round(time.monotonic() - start, 2)
        print(f"OK: model loaded standalone in {load_seconds}s")
    except Exception as exc:
        print(f"FAIL: model failed to load: {exc}")
        return False

    if not test_image:
        print(
            "No --test-image given -- skipping live inference call. "
            "Model load succeeded but end-to-end inference is UNVERIFIED. "
            "Re-run with --test-image path/to/leaf.jpg to fully confirm."
        )
        return True

    if not Path(test_image).exists():
        print(f"FAIL: --test-image path does not exist: {test_image}")
        return False

    try:
        start = time.monotonic()
        detections = detector.predict(test_image, image_ref=test_image, log_low_confidence=False)
        infer_seconds = round(time.monotonic() - start, 2)
    except Exception as exc:
        print(f"FAIL: inference raised: {exc}")
        return False

    print(f"OK: inference ran in {infer_seconds}s, {len(detections)} detection(s)")
    for d in detections[:5]:
        print(f"    {d['class_name']:24s} conf={d['confidence']:.3f} tier={d['tier']}")
    return True


def run_qualitative_external(external_dir: str) -> None:
    """Runs inference on external/unseen images. Results are written for
    manual review ONLY -- never merged into train/val/test or the formal
    metrics (Section 24)."""
    from vision.detector import get_detector

    ext_path = Path(external_dir)
    if not ext_path.is_dir():
        print(f"FAIL: {external_dir} is not a directory")
        return

    images = [
        p for p in sorted(ext_path.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    if not images:
        print(f"No image files found in {external_dir}")
        return

    detector = get_detector()
    records = []
    print(f"\n=== Qualitative external test: {len(images)} image(s) ===")
    for img_path in images:
        try:
            detections = detector.predict(str(img_path), image_ref=str(img_path), log_low_confidence=False)
            top = detections[0] if detections else None
            print(
                f"  {img_path.name:40s} -> "
                f"{top['class_name'] if top else 'NO DETECTION'}"
                f" ({top['confidence']:.3f}, {top['tier']})" if top else f"  {img_path.name:40s} -> NO DETECTION"
            )
            records.append({
                "image": str(img_path),
                "detections": detections,
                "note": "external/unseen image -- NOT part of train/val/test, qualitative only",
            })
        except Exception as exc:
            print(f"  {img_path.name:40s} -> ERROR: {exc}")
            records.append({"image": str(img_path), "error": str(exc)})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "vision_qualitative.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nQualitative results written to {out_path} -- review manually, not auto-scored.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="KrishokBot vision evaluation / deployment check.")
    parser.add_argument("--test-image", type=str, default=None, help="Single image for the deployment smoke test.")
    parser.add_argument("--external-dir", type=str, default=None, help="Directory of external/unseen images for qualitative review.")
    args = parser.parse_args()

    print_training_metrics()
    ok = check_deployment_readiness(args.test_image)

    if args.external_dir:
        run_qualitative_external(args.external_dir)

    print(f"\nDeployment check: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
