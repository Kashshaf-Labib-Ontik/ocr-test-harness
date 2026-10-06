from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from time import perf_counter

from ocr_benchmark.schema import OCRBlock, OCRResult


class OCRAdapter(ABC):
    """Common contract implemented by every OCR engine adapter."""

    name: str

    def recognize(self, image_path: str | Path) -> OCRResult:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"OCR input does not exist: {path}")

        started = perf_counter()
        text, blocks, metadata = self._recognize(path)
        elapsed = perf_counter() - started

        return OCRResult(
            engine=self.name,
            image_path=str(path),
            text=text,
            elapsed_seconds=elapsed,
            blocks=blocks,
            metadata=metadata,
        )

    @abstractmethod
    def _recognize(
        self, image_path: Path
    ) -> tuple[str, list[OCRBlock], dict[str, object]]:
        """Run an engine and return normalized text, blocks, and metadata."""
