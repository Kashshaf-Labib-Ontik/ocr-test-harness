from __future__ import annotations

from pathlib import Path
from typing import Any

from ocr_benchmark.adapters.base import OCRAdapter
from ocr_benchmark.schema import OCRBlock


class EasyOCRAdapter(OCRAdapter):
    """CPU adapter for EasyOCR's combined Bengali and English model."""

    name = "easyocr"

    def __init__(
        self,
        model_directory: str | Path = "models/easyocr",
        reader: Any | None = None,
    ) -> None:
        self.languages = ("bn", "en")
        if reader is not None:
            self.reader = reader
            return

        try:
            import easyocr
        except ImportError as error:
            raise RuntimeError("EasyOCR is not installed") from error

        self.reader = easyocr.Reader(
            list(self.languages),
            gpu=False,
            model_storage_directory=str(Path(model_directory)),
        )

    def _recognize(
        self, image_path: Path
    ) -> tuple[str, list[OCRBlock], dict[str, object]]:
        detections = self.reader.readtext(str(image_path), detail=1, paragraph=False)
        blocks = []
        for polygon, text, confidence in detections:
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            blocks.append(
                OCRBlock(
                    text=str(text),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    confidence=float(confidence),
                )
            )

        blocks.sort(
            key=lambda block: (
                block.bbox[1] if block.bbox else 0,
                block.bbox[0] if block.bbox else 0,
            )
        )
        for order, block in enumerate(blocks):
            block.reading_order = order

        return (
            "\n".join(block.text for block in blocks),
            blocks,
            {"device": "cpu", "languages": list(self.languages)},
        )
