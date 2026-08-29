"""vision/class_mapping.py

Maps raw YOLO class names (English, e.g. "rice_leaf_scald") to:
  - bn_name:    Bengali display name for the UI/synthesis answer
  - rag_query:  Bengali search phrase to feed rag.retriever for enrichment
  - crop_bn:    Bengali crop name alone (used in badges/headers)

Why this exists (separate from detector.py):
  detector.py returns raw training class_name strings -- those are English,
  underscore-joined, and not meant for end users. Without this mapping the
  UI would either show "rice_leaf_scald" directly to a farmer, or the
  synthesis LLM would have to guess a Bengali translation each time
  (inconsistent, unverifiable). This gives one fixed, reviewed mapping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassInfo:
    crop_bn: str
    bn_name: str       # full Bengali disease/status name for display
    rag_query: str      # Bengali phrase used to query rag.retriever for related advice


CLASS_MAP: dict[str, ClassInfo] = {
    "rice_brown_spot": ClassInfo(
        crop_bn="ধান",
        bn_name="বাদামী দাগ রোগ",
        rag_query="ধানের বাদামী দাগ রোগ প্রতিকার",
    ),
    "rice_leaf_scald": ClassInfo(
        crop_bn="ধান",
        bn_name="পাতা ঝলসানো রোগ (লিফ স্কাল্ড)",
        rag_query="ধানের পাতা ঝলসানো রোগ প্রতিকার",
    ),
    "rice_blast": ClassInfo(
        crop_bn="ধান",
        bn_name="ব্লাস্ট রোগ",
        rag_query="ধান ব্লাস্ট রোগ প্রতিকার",
    ),
    "rice_tungro": ClassInfo(
        crop_bn="ধান",
        bn_name="টুংরো রোগ",
        rag_query="ধানের টুংরো রোগ প্রতিকার",
    ),
    "rice_bacterial_blight": ClassInfo(
        crop_bn="ধান",
        bn_name="ব্যাকটেরিয়াল ব্লাইট (পাতা পোড়া রোগ)",
        rag_query="ধানের ব্যাকটেরিয়াল ব্লাইট রোগ প্রতিকার",
    ),
    "potato_early_blight": ClassInfo(
        crop_bn="আলু",
        bn_name="আগাম ধ্বসা রোগ",
        rag_query="আলুর আগাম ধ্বসা রোগ প্রতিকার",
    ),
    "potato_late_blight": ClassInfo(
        crop_bn="আলু",
        bn_name="নাবী ধ্বসা রোগ",
        rag_query="আলুর নাবী ধ্বসা রোগ প্রতিকার",
    ),
    "potato_healthy": ClassInfo(
        crop_bn="আলু",
        bn_name="সুস্থ গাছ",
        rag_query="আলু গাছের যত্ন ও পরিচর্যা",
    ),
    "tomato_early_blight": ClassInfo(
        crop_bn="টমেটো",
        bn_name="আগাম ধ্বসা রোগ",
        rag_query="টমেটোর আগাম ধ্বসা রোগ প্রতিকার",
    ),
    "tomato_late_blight": ClassInfo(
        crop_bn="টমেটো",
        bn_name="নাবী ধ্বসা রোগ",
        rag_query="টমেটোর নাবী ধ্বসা রোগ প্রতিকার",
    ),
    "tomato_healthy": ClassInfo(
        crop_bn="টমেটো",
        bn_name="সুস্থ গাছ",
        rag_query="টমেটো গাছের যত্ন ও পরিচর্যা",
    ),
}


def get_class_info(class_name: str | None) -> ClassInfo:
    """Always returns a usable ClassInfo, even for None/unknown class_name
    (e.g. no detection, or a future class not yet added here) -- callers
    (UI, synthesis_node) should never crash on a missing mapping."""
    if class_name and class_name in CLASS_MAP:
        return CLASS_MAP[class_name]
    return ClassInfo(
        crop_bn="অজানা",
        bn_name="সনাক্ত করা যায়নি",
        rag_query="ফসলের রোগ সনাক্তকরণ ও প্রতিকার",
    )
