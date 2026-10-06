from __future__ import annotations


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def text_score(metrics: dict[str, float | bool]) -> float:
    """Calculate a 0-100 text score from the baseline OCR metrics."""

    character_accuracy = 1.0 - _clamp(float(metrics["cer"]))
    word_accuracy = 1.0 - _clamp(float(metrics["wer"]))
    exact_accuracy = 1.0 if metrics["exact_match"] else 0.0

    return round(
        100.0
        * (
            0.60 * character_accuracy
            + 0.30 * word_accuracy
            + 0.10 * exact_accuracy
        ),
        2,
    )


def overall_score(
    accuracy_score: float,
    median_seconds: float,
    fastest_median_seconds: float,
) -> dict[str, float]:
    """Combine accuracy with speed relative to the fastest tested engine."""

    if median_seconds < 0 or fastest_median_seconds < 0:
        raise ValueError("Runtime values cannot be negative")
    if median_seconds == 0:
        speed_score = 100.0
    else:
        speed_score = 100.0 * _clamp(fastest_median_seconds / median_seconds)

    total = 0.90 * _clamp(accuracy_score / 100.0) * 100.0 + 0.10 * speed_score
    return {
        "accuracy_score": round(_clamp(accuracy_score / 100.0) * 100.0, 2),
        "speed_score": round(speed_score, 2),
        "overall_score": round(total, 2),
    }
