from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_benchmark.adapters import PaddleOCRVLAdapter
from ocr_benchmark.metrics import evaluate_text
from ocr_benchmark.scoring import text_score


def run() -> None:
    dataset = ROOT / "data" / "synthetic" / "v0.1"
    records = [
        json.loads(line)
        for line in (dataset / "ground_truth" / "pages.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    page = next(record for record in records if record["id"] == "bn_policy_01_clean")
    adapter = PaddleOCRVLAdapter(cache_directory=ROOT / "models" / "paddlex")
    prediction = adapter.recognize(dataset / page["image"])
    metrics = evaluate_text(page["text_nfc"], prediction.text)
    result = {
        "id": page["id"],
        "prediction": prediction.to_dict(),
        "metrics": metrics,
        "text_score": text_score(metrics),
    }
    output = ROOT / "results" / "paddleocr_vl_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "text_score": result["text_score"]}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    run()
