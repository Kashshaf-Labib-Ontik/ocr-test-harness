import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ocr_benchmark.adapters import PaddleOCRVLAdapter


class FakeResult:
    json = {
        "res": {
            "parsing_res_list": [
                {
                    "block_label": "text",
                    "block_content": "Policy &amp; নিয়ম",
                    "block_bbox": [10, 20, 100, 50],
                    "block_order": 1,
                }
            ]
        }
    }


class FakePipeline:
    def predict(self, image_path):
        return [FakeResult()]


class PaddleOCRVLAdapterTest(unittest.TestCase):
    def test_structured_result_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.png"
            image.touch()
            result = PaddleOCRVLAdapter(pipeline=FakePipeline()).recognize(image)

        self.assertEqual(result.text, "Policy & নিয়ম")
        self.assertEqual(result.blocks[0].bbox, (10.0, 20.0, 100.0, 50.0))
        self.assertEqual(result.blocks[0].block_type, "text")


if __name__ == "__main__":
    unittest.main()
