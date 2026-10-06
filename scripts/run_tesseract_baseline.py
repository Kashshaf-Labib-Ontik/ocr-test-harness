from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_benchmark.adapters import TesseractAdapter
from ocr_benchmark.metrics import evaluate_text
from ocr_benchmark.scoring import overall_score, text_score


DATASET = ROOT / "data" / "synthetic" / "v0.1"
GROUND_TRUTH = DATASET / "ground_truth" / "pages.jsonl"
OUTPUT = ROOT / "results" / "tesseract_baseline.json"


def run() -> None:
    pages = [
        json.loads(line)
        for line in GROUND_TRUTH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    adapter = TesseractAdapter(tessdata_directory=ROOT / "models" / "tesseract")
    results = []

    for index, page in enumerate(pages, start=1):
        prediction = adapter.recognize(DATASET / page["image"])
        prediction_data = prediction.to_dict()
        prediction_data["image_path"] = page["image"]
        metrics = evaluate_text(page["text_nfc"], prediction.text)
        results.append(
            {
                "id": page["id"],
                "language": page["language"],
                "degradation": page["degradation"],
                "prediction": prediction_data,
                "metrics": metrics,
                "text_score": text_score(metrics),
            }
        )
        print(f"[{index}/{len(pages)}] {page['id']}: CER={metrics['cer']:.3f}")

    mean_text_score = statistics.fmean(item["text_score"] for item in results)
    median_seconds = statistics.median(
        item["prediction"]["elapsed_seconds"] for item in results
    )
    summary = {
        "engine": adapter.name,
        "pages": len(results),
        "mean_cer": statistics.fmean(item["metrics"]["cer"] for item in results),
        "mean_wer": statistics.fmean(item["metrics"]["wer"] for item in results),
        "mean_text_score": mean_text_score,
        "median_seconds": median_seconds,
        **overall_score(mean_text_score, median_seconds, median_seconds),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps({"summary": summary, "pages": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    run()
