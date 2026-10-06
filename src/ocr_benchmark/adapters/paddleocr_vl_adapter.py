from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any

from ocr_benchmark.adapters.base import OCRAdapter
from ocr_benchmark.schema import OCRBlock


def _plain_text(value: object) -> str:
    text = html.unescape(str(value))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


class PaddleOCRVLAdapter(OCRAdapter):
    """CPU adapter for PaddleOCR-VL structured document recognition."""

    name = "paddleocr-vl-1.5"

    def __init__(
        self,
        cache_directory: str | Path = "models/paddlex",
        pipeline: Any | None = None,
    ) -> None:
        if pipeline is not None:
            self.pipeline = pipeline
            return

        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path(cache_directory)))
        from paddleocr import PaddleOCRVL

        self.pipeline = PaddleOCRVL(
            pipeline_version="v1.5",
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )

    def _recognize(
        self, image_path: Path
    ) -> tuple[str, list[OCRBlock], dict[str, object]]:
        pages = list(self.pipeline.predict(str(image_path)))
        blocks: list[OCRBlock] = []

        for page in pages:
            raw = page.json["res"]
            for item in raw.get("parsing_res_list", []):
                bbox = item.get("block_bbox")
                blocks.append(
                    OCRBlock(
                        text=_plain_text(item.get("block_content", "")),
                        bbox=tuple(float(value) for value in bbox) if bbox else None,
                        block_type=str(item.get("block_label", "text")),
                        reading_order=item.get("block_order"),
                    )
                )

        blocks.sort(
            key=lambda block: (
                block.reading_order is None,
                block.reading_order if block.reading_order is not None else 0,
            )
        )
        return (
            "\n".join(block.text for block in blocks if block.text),
            blocks,
            {"device": "cpu", "pipeline_version": "v1.5"},
        )
