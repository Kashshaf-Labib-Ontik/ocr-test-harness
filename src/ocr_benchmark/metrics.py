from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def normalize_text(text: str) -> str:
    """Apply the shared normalization used for baseline text metrics."""

    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def edit_distance(reference: Sequence[T], prediction: Sequence[T]) -> int:
    """Compute Levenshtein distance using one row of memory."""

    if len(reference) < len(prediction):
        reference, prediction = prediction, reference

    previous = list(range(len(prediction) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for pred_index, pred_item in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[pred_index] + 1,
                    previous[pred_index - 1] + (ref_item != pred_item),
                )
            )
        previous = current
    return previous[-1]


def _error_rate(distance: int, reference_length: int) -> float:
    if reference_length:
        return distance / reference_length
    return 0.0 if distance == 0 else 1.0


def evaluate_text(reference: str, prediction: str) -> dict[str, float | bool]:
    """Return the small baseline metric set for one OCR prediction."""

    ref = normalize_text(reference)
    pred = normalize_text(prediction)

    char_distance = edit_distance(ref, pred)
    ref_words = ref.split()
    pred_words = pred.split()
    word_distance = edit_distance(ref_words, pred_words)
    longest = max(len(ref), len(pred))

    return {
        "cer": _error_rate(char_distance, len(ref)),
        "wer": _error_rate(word_distance, len(ref_words)),
        "exact_match": ref == pred,
        "edit_similarity": 1.0 if longest == 0 else 1.0 - char_distance / longest,
    }
