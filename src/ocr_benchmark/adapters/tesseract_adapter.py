from __future__ import annotations

import csv
import io
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Callable

from ocr_benchmark.adapters.base import OCRAdapter
from ocr_benchmark.schema import OCRBlock


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


class TesseractAdapter(OCRAdapter):
    """Tesseract adapter using official Bengali and English trained data."""

    name = "tesseract-fast"

    def __init__(
        self,
        executable: str | Path = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        tessdata_directory: str | Path = "models/tesseract",
        languages: str = "ben+eng",
        page_segmentation_mode: int = 3,
        runner: CommandRunner | None = None,
    ) -> None:
        self.executable = Path(executable)
        self.tessdata_directory = Path(tessdata_directory)
        self.languages = languages
        self.page_segmentation_mode = page_segmentation_mode
        self.runner = runner or self._run_command

    @staticmethod
    def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def _recognize(
        self, image_path: Path
    ) -> tuple[str, list[OCRBlock], dict[str, object]]:
        command = [
            str(self.executable),
            str(image_path),
            "stdout",
            "--tessdata-dir",
            str(self.tessdata_directory),
            "-l",
            self.languages,
            "--oem",
            "1",
            "--psm",
            str(self.page_segmentation_mode),
            "tsv",
        ]
        completed = self.runner(command)
        lines: dict[tuple[int, int, int, int], list[dict[str, str]]] = defaultdict(list)

        for row in csv.DictReader(io.StringIO(completed.stdout), delimiter="\t"):
            if row.get("level") == "5" and row.get("text", "").strip():
                key = tuple(
                    int(row[field])
                    for field in ("page_num", "block_num", "par_num", "line_num")
                )
                lines[key].append(row)

        blocks = []
        for order, key in enumerate(sorted(lines)):
            words = sorted(lines[key], key=lambda row: int(row["word_num"]))
            left = min(int(row["left"]) for row in words)
            top = min(int(row["top"]) for row in words)
            right = max(int(row["left"]) + int(row["width"]) for row in words)
            bottom = max(int(row["top"]) + int(row["height"]) for row in words)
            confidences = [float(row["conf"]) / 100.0 for row in words]
            blocks.append(
                OCRBlock(
                    text=" ".join(row["text"].strip() for row in words),
                    bbox=(float(left), float(top), float(right), float(bottom)),
                    confidence=statistics.fmean(confidences),
                    block_type="line",
                    reading_order=order,
                )
            )

        return (
            "\n".join(block.text for block in blocks),
            blocks,
            {
                "device": "cpu",
                "languages": self.languages,
                "page_segmentation_mode": self.page_segmentation_mode,
            },
        )
