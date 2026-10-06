from __future__ import annotations

import json
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data" / "synthetic" / "v0.1"
SOURCE_DIR = DATASET / "source_documents"
IMAGE_DIR = DATASET / "images" / "test"
GROUND_TRUTH = DATASET / "ground_truth" / "pages.jsonl"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


@dataclass(frozen=True)
class Block:
    block_type: str
    language: str
    bbox: tuple[int, int, int, int]
    text: str


@dataclass(frozen=True)
class Page:
    page_id: str
    language: str
    font: str
    blocks: tuple[Block, ...]


PAGES = (
    Page(
        "bn_policy_01",
        "bn",
        "Noto Bengali",
        (
            Block("heading", "bn", (90, 85, 1150, 175), "গ্রাহক তথ্য সুরক্ষা নীতিমালা"),
            Block("paragraph", "bn", (90, 215, 1150, 390), "১. উদ্দেশ্য\nএই নীতিমালার উদ্দেশ্য হলো গ্রাহকের ব্যক্তিগত তথ্য নিরাপদ রাখা এবং অনুমোদিত কাজে ব্যবহার নিশ্চিত করা।"),
            Block("list_item", "bn", (110, 430, 1130, 665), "২. তথ্য ব্যবহারের নিয়ম\n• গ্রাহকের সম্মতি ছাড়া তথ্য প্রকাশ করা যাবে না।\n• কর্মীরা কেবল প্রয়োজনীয় তথ্য ব্যবহার করবেন।\n• ভুল তথ্য দ্রুত সংশোধন করতে হবে।"),
            Block("table", "bn", (90, 735, 1150, 1085), "তথ্যের ধরন | সংরক্ষণের সময়\nআবেদনপত্র | ৫ বছর\nযোগাযোগের রেকর্ড | ২ বছর\nঅভিযোগের নথি | ৩ বছর"),
            Block("footer", "bn", (90, 1570, 1150, 1640), "সিনথেটিক পরীক্ষা নথি — বাস্তব নীতিমালা নয়"),
        ),
    ),
    Page(
        "en_policy_01",
        "en",
        "Noto Sans",
        (
            Block("heading", "en", (90, 85, 1150, 175), "Customer Complaint Policy"),
            Block("paragraph", "en", (90, 215, 1150, 390), "1. Purpose\nThis policy defines how customer complaints are recorded, reviewed, and resolved fairly."),
            Block("list_item", "en", (110, 430, 1130, 665), "2. Service rules\n• A complaint must receive a reference number.\n• Urgent cases must be reviewed within one business day.\n• Customers must receive a written decision."),
            Block("table", "en", (90, 735, 1150, 1085), "Priority | Response target\nUrgent | 1 business day\nNormal | 5 business days\nInformation request | 7 business days"),
            Block("footer", "en", (90, 1570, 1150, 1640), "SYNTHETIC TEST DOCUMENT — NOT AN ACTIVE POLICY"),
        ),
    ),
    Page(
        "mixed_policy_01",
        "mixed",
        "Noto Bengali",
        (
            Block("heading", "mixed", (90, 85, 1150, 175), "Loan Review Policy / ঋণ পর্যালোচনা নীতিমালা"),
            Block("paragraph", "mixed", (90, 215, 1150, 410), "1. Application review / আবেদন পর্যালোচনা\nপ্রতিটি আবেদন সম্পূর্ণতা, পরিচয় এবং পরিশোধ সক্ষমতার ভিত্তিতে যাচাই করা হবে।"),
            Block("list_item", "mixed", (110, 450, 1130, 690), "Required checks / প্রয়োজনীয় যাচাই\n• Identity document / পরিচয়পত্র\n• Declared income / ঘোষিত আয়\n• Contact verification / যোগাযোগ যাচাই"),
            Block("table", "mixed", (90, 760, 1150, 1110), "Decision / সিদ্ধান্ত | Review period / সময়\nApproved / অনুমোদিত | 3 days / ৩ দিন\nFurther review / পুনরায় যাচাই | 7 days / ৭ দিন\nDeclined / প্রত্যাখ্যাত | Written notice / লিখিত নোটিশ"),
            Block("footer", "mixed", (90, 1570, 1150, 1640), "SYNTHETIC / কৃত্রিম পরীক্ষার নথি"),
        ),
    ),
)


def _html(page: Page) -> str:
    block_html = []
    for block in page.blocks:
        x1, y1, x2, y2 = block.bbox
        content = (
            block.text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        size = 42 if block.block_type == "heading" else 27
        if block.block_type == "footer":
            size = 19
        style = (
            f"left:{x1}px;top:{y1}px;width:{x2-x1}px;height:{y2-y1}px;"
            f"font-size:{size}px;line-height:1.55;"
        )
        css_class = "table" if block.block_type == "table" else "block"
        block_html.append(f'<div class="{css_class}" style="{style}">{content}</div>')

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
@font-face {{ font-family: 'Noto Bengali'; src: url('../../../../assets/fonts/NotoSansBengali-Regular.ttf'); }}
@font-face {{ font-family: 'Noto Sans'; src: url('../../../../assets/fonts/NotoSans-Regular.ttf'); }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; width: 1240px; height: 1754px; overflow: hidden; }}
body {{ position: relative; background: white; color: #111; font-family: '{page.font}'; }}
.block, .table {{ position: absolute; white-space: normal; }}
.table {{ border: 2px solid #444; padding: 20px; line-height: 1.8 !important; background: #fafafa; }}
</style></head><body>{''.join(block_html)}</body></html>"""


def _render(html_path: Path, output_path: Path) -> None:
    command = [
        str(CHROME),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        "--window-size=1240,1754",
        f"--screenshot={output_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _degrade(clean_path: Path, output_path: Path, seed: int) -> None:
    image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read rendered page: {clean_path}")

    rng = np.random.default_rng(seed)
    blurred = cv2.GaussianBlur(image, (3, 3), 0.8)
    noise = rng.normal(0, 5, blurred.shape).astype(np.int16)
    degraded = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    encoded = cv2.imencode(".jpg", degraded, [cv2.IMWRITE_JPEG_QUALITY, 68])[1]
    output_path.write_bytes(encoded.tobytes())


def generate() -> None:
    if not CHROME.is_file():
        raise FileNotFoundError(f"Chrome was not found at {CHROME}")

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    records = []

    for seed, page in enumerate(PAGES, start=1):
        html_path = SOURCE_DIR / f"{page.page_id}.html"
        clean_path = IMAGE_DIR / f"{page.page_id}_clean.png"
        scan_path = IMAGE_DIR / f"{page.page_id}_scan.jpg"
        html_path.write_text(_html(page), encoding="utf-8")
        _render(html_path, clean_path)
        _degrade(clean_path, scan_path, seed)

        full_text = "\n".join(block.text for block in page.blocks)
        blocks = [
            {
                "type": block.block_type,
                "language": block.language,
                "bbox": block.bbox,
                "text": block.text,
                "reading_order": order,
            }
            for order, block in enumerate(page.blocks)
        ]
        for variant, image_path in (("clean", clean_path), ("scan", scan_path)):
            records.append(
                {
                    "id": f"{page.page_id}_{variant}",
                    "source_id": page.page_id,
                    "split": "test",
                    "language": page.language,
                    "image": image_path.relative_to(DATASET).as_posix(),
                    "degradation": variant,
                    "text_raw": full_text,
                    "text_nfc": unicodedata.normalize("NFC", full_text),
                    "blocks": blocks,
                    "tables": [],
                }
            )

    with GROUND_TRUTH.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} images and {GROUND_TRUTH}")


if __name__ == "__main__":
    generate()
