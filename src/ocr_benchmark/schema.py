from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class OCRBlock:
    """One text region returned by an OCR engine."""

    text: str
    bbox: BBox | None = None
    confidence: float | None = None
    block_type: str = "text"
    language: str | None = None
    reading_order: int | None = None


@dataclass(slots=True)
class OCRResult:
    """Engine-independent OCR output for one image."""

    engine: str
    image_path: str
    text: str
    elapsed_seconds: float
    blocks: list[OCRBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
