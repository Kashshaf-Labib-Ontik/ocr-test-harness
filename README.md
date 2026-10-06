# Multilingual OCR Test Harness

A small, reproducible, CPU-first benchmark for evaluating open-source OCR engines on Bangla, English, and mixed-language policy documents.

The project focuses on document types relevant to Bangladeshi organizations, including policy pages containing paragraphs, headings, bullet lists, tables, dates, amounts, and other structured content. It is intentionally small so the complete pipeline can be validated before expanding the dataset or installing additional models.

## Current status

The repository currently provides:

- A common OCR adapter interface and engine-independent result schema.
- Lightweight document preprocessing variants.
- Unicode-aware text normalization and OCR evaluation metrics.
- A transparent provisional scoring formula.
- A deterministic synthetic Bangla-English policy-document generator.
- Six generated test images with JSONL ground truth.
- Adapters for Tesseract, EasyOCR, and PaddleOCR-VL.
- A completed Tesseract `ben+eng` CPU benchmark.
- A completed one-page PaddleOCR-VL 1.5 CPU smoke test.
- Unit tests for schemas, adapters, preprocessing, metrics, and scoring.

Tesseract is the first engine with a completed six-page benchmark. PaddleOCR-VL has completed a one-page real-inference smoke test, while EasyOCR remains contract-tested only in the current Windows environment. See [Engine status](#engine-status) for details.

## Benchmark pipeline

```text
Synthetic policy source
        |
        v
Clean and degraded page images + JSONL ground truth
        |
        v
Optional preprocessing
        |
        v
Common OCR adapter
        |
        v
Normalized text and region output
        |
        v
CER, WER, exact match, similarity, runtime, and scores
        |
        v
Machine-readable JSON result
```

## Dataset

Dataset version `v0.1` contains three canonical policy pages:

| Source page | Language | Content |
|---|---|---|
| `bn_policy_01` | Bangla | Heading, paragraphs, bullets, and a retention table |
| `en_policy_01` | English | Complaint-handling policy and response table |
| `mixed_policy_01` | Bangla-English | Loan-review policy with bilingual fields |

Each source page has a clean digitally rendered version and a deterministic blurred, noisy, JPEG-compressed scan. This produces six images. Ground truth is stored in `data/synthetic/v0.1/ground_truth/pages.jsonl` and includes exact text, Unicode NFC text, language, block type, bounding box, and reading order.

The included dataset is a pipeline-validation fixture, not a statistically sufficient production benchmark. A larger dataset with licensed real documents and genuine handwriting will be required before making deployment decisions.

## Evaluation metrics

All baseline text metrics are calculated after Unicode NFC normalization, whitespace collapsing, and trimming. Punctuation and Bangla digits are preserved.

### Character Error Rate

Character Error Rate measures character-level transcription errors:

```text
CER = character Levenshtein distance / number of reference characters
```

Lower is better. `0` represents a perfect transcription. Insertions can cause CER to exceed `1`; the scoring formula caps the error contribution at `1`.

### Word Error Rate

Word Error Rate measures word insertions, deletions, and substitutions:

```text
WER = word-level Levenshtein distance / number of reference words
```

Lower is better. WER is useful for human-readable documents but is sensitive to OCR spacing and tokenization errors.

### Exact match

Exact match is `true` only when the complete normalized prediction equals the complete normalized reference. It is deliberately strict and therefore receives a relatively small score weight.

### Normalized edit similarity

```text
similarity = 1 - character distance / max(reference length, prediction length)
```

Higher is better. A perfect match scores `1`.

### Runtime

Each adapter records wall-clock inference time per page. Model download and adapter initialization are excluded. The benchmark reports median seconds per page to reduce sensitivity to outliers.

## Provisional scores

The current text score is:

```text
character accuracy = 1 - clamp(CER, 0, 1)
word accuracy      = 1 - clamp(WER, 0, 1)

text score = 100 * (
    0.60 * character accuracy
  + 0.30 * word accuracy
  + 0.10 * exact match
)
```

When multiple engines are available, speed is calculated relative to the fastest median runtime:

```text
speed score = 100 * clamp(fastest median / engine median, 0, 1)
overall score = 0.90 * text score + 0.10 * speed score
```

This formula is provisional. Table structure, layout accuracy, reading order, robustness, and field-level accuracy will receive separate weights after the structured dataset is expanded. With only one completed engine, Tesseract's speed score is automatically `100` and should not yet be interpreted as a comparative result.

## Tesseract baseline results

Configuration:

- Tesseract 5.4.0 with official `tessdata_fast` Bengali and English models.
- Languages: `ben+eng`.
- LSTM OCR engine mode: `--oem 1`.
- Automatic page segmentation: `--psm 3`.
- CPU-only execution.

| Page | CER | WER | Text score | Runtime |
|---|---:|---:|---:|---:|
| Bangla clean | 7.44% | 10.81% | 82.29 | 0.398 s |
| Bangla scan | 18.14% | 18.92% | 73.44 | 0.324 s |
| English clean | 0.66% | 4.11% | 88.37 | 0.350 s |
| English scan | 0.88% | 6.85% | 87.41 | 0.411 s |
| Mixed clean | 1.16% | 7.95% | 86.92 | 0.500 s |
| Mixed scan | 0.78% | 4.55% | 88.17 | 0.464 s |

Aggregate result:

| Metric | Value |
|---|---:|
| Mean CER | 4.84% |
| Mean WER | 8.86% |
| Mean text score | 84.43 |
| Median runtime | 0.40 seconds/page |
| Provisional overall score | 85.99 |

Complete predictions, boxes, confidence values, metrics, and timings are stored in `results/tesseract_baseline.json`. The current fixture indicates that degraded Bangla is the main Tesseract weakness, but the dataset is too small to generalize this result to real policy documents.

### PaddleOCR-VL CPU smoke test

PaddleOCR-VL 1.5 was run on the clean synthetic Bangla policy page using CPU inference. It reproduced the normalized reference text exactly: CER 0%, WER 0%, exact match true, and text score 100. Inference took 370.68 seconds (about 6 minutes 11 seconds) and used approximately 3.9 GB of working memory during the observed run. This is a one-page smoke test, not a result directly comparable with the six-page Tesseract benchmark. The detailed output is written locally to `results/paddleocr_vl_smoke.json`.

## Engine status

| Engine | Status | Notes |
|---|---|---|
| Tesseract | Completed | CPU baseline completed with official Bengali and English models |
| EasyOCR | Adapter tested | Real inference was blocked by the host's enterprise policy rejecting PyTorch native extensions |
| PaddleOCR-VL | Smoke test completed | Clean Bangla page achieved 0% CER and 0% WER in 370.68 seconds on CPU; full dataset benchmark remains pending |
| Kraken | Planned | Requires a suitable Bengali model or separately licensed training data |
| TrOCR | Planned | Requires validation of the Bengali checkpoint and training-data provenance |

The EasyOCR environment failure is not an accuracy conclusion about that engine.

## Local client demo

The repository includes a dependency-free local web interface for demonstrating OCR to clients. It supports image upload and the six prepared samples, lets the presenter switch between Tesseract and PaddleOCR-VL, overlays detected regions, and provides extracted text plus downloadable structured JSON. Documents are processed on the local machine; temporary uploads are deleted immediately after inference.

Prerequisites:

- Complete the core setup and Tesseract setup below.
- Install `requirements-paddle.txt` only when PaddleOCR-VL will be demonstrated.
- Allow roughly 4 GB of available memory for PaddleOCR-VL. Its model is downloaded on first use and cached under `models/paddlex/`; model files are not committed.

Start the demo from PowerShell:

```powershell
cd D:\ocr-demo
.\venv\Scripts\Activate.ps1
python scripts\run_local_demo.py
```

If PowerShell activation is disabled, run the environment's interpreter directly:

```powershell
.\venv\Scripts\python.exe scripts\run_local_demo.py
```

The browser opens at `http://127.0.0.1:8000`. To present the demo:

1. Upload a PNG, JPEG, WebP, TIFF, or BMP image up to 15 MB, or select one of the prepared Bangla, English, or mixed-language policy samples.
2. Choose Tesseract for a result that typically completes in under one second on the test machine.
3. Choose PaddleOCR-VL for structured document parsing; the clean Bangla sample took about six minutes on the current CPU-only machine.
4. Review the extracted text, processing time, text-region overlay, and reading-order blocks.
5. Copy the text or download the engine-independent JSON result.
6. Press `Ctrl+C` in the terminal to stop the local server.

The server binds to `127.0.0.1` by default, so the interface is available only on the local computer. No frontend package manager, cloud account, API key, or paid service is required.

## Repository structure

```text
assets/fonts/                   Open-licensed Noto fonts
demo/                           Dependency-free local client interface
data/synthetic/v0.1/           Tiny generated dataset and ground truth
models/tesseract/configs/       Tracked Tesseract TSV configuration
results/                        Selected reproducible benchmark results
scripts/                        Dataset generation and engine runners
src/ocr_benchmark/              Schemas, metrics, scoring, preprocessing, adapters
tests/                          Unit and adapter-contract tests
```

Model weights and trained data are intentionally not committed.

## Setup

### 1. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The project was developed with Python 3.12 on Windows 11.

### 2. Install Tesseract on Windows

```powershell
winget install --id UB-Mannheim.TesseractOCR --exact
```

The adapter expects `C:\Program Files\Tesseract-OCR\tesseract.exe`.

### 3. Download Bengali and English models

```powershell
New-Item ".\models\tesseract" -ItemType Directory -Force

curl.exe -L `
  https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/ben.traineddata `
  -o ".\models\tesseract\ben.traineddata"

curl.exe -L `
  https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata `
  -o ".\models\tesseract\eng.traineddata"
```

The required TSV configuration is tracked at `models/tesseract/configs/tsv`.

## Usage

### Regenerate the tiny dataset

The generator uses local headless Chrome for correct Bengali shaping.

```powershell
.\venv\Scripts\python.exe .\scripts\generate_tiny_dataset.py
```

### Run the Tesseract benchmark

```powershell
.\venv\Scripts\python.exe .\scripts\run_tesseract_baseline.py
```

### Manually inspect Tesseract output

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" `
  ".\data\synthetic\v0.1\images\test\bn_policy_01_clean.png" stdout `
  --tessdata-dir ".\models\tesseract" `
  -l ben+eng --oem 1 --psm 3
```

### Run tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The current suite contains 12 tests.

## Preprocessing variants

The project currently provides three dimension-preserving variants:

- `original`: unchanged BGR image.
- `enhance`: grayscale with CLAHE local contrast enhancement.
- `binary`: enhanced grayscale followed by Gaussian blur and Otsu thresholding.

Dimension preservation keeps ground-truth coordinates valid. Preprocessing variants have not yet been included in the published Tesseract baseline.

## Limitations and next steps

- Expand beyond six synthetic images.
- Add licensed real Bangla and English policy documents.
- Add genuine handwritten Bangla samples as a separate evaluation track.
- Complete at least four additional engine integrations.
- Add table cell, layout, reading-order, and field-level metrics.
- Add confidence intervals once the dataset is large enough.
- Record package, model, hardware, and operating-system metadata automatically.
- Compare preprocessing variants without giving any engine an unfair advantage.

## Licensing and data safety

- Tesseract and its official trained data are distributed under Apache-2.0.
- Included Noto fonts are distributed under the SIL Open Font License 1.1.
- Generated policy content is synthetic and contains no client or banking data.
- Third-party engines, model weights, and future datasets retain their own licenses.
- A repository-level software license has not yet been selected.
