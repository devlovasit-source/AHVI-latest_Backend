from services.hybrid_detection_service import MAX_ITEMS, filter_and_limit


def test_hybrid_detection_allows_six_items():
    detections = [
        {"label": f"shirt {idx}", "bbox": [idx * 40, 10, idx * 40 + 35, 80], "score": 1.0 - idx * 0.01}
        for idx in range(6)
    ]

    kept = filter_and_limit(detections, width=320, height=120)

    assert MAX_ITEMS >= 6
    assert len(kept) == 6
