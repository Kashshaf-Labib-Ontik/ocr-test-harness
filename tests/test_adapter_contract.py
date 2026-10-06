import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ocr_benchmark.adapters import OCRAdapter
from ocr_benchmark.schema import OCRBlock


class FakeAdapter(OCRAdapter):
    name = "fake"

    def _recognize(self, image_path: Path):
        block = OCRBlock(text="বাংলা English", bbox=(0, 0, 10, 10), confidence=1.0)
        return block.text, [block], {"device": "cpu"}


class AdapterContractTest(unittest.TestCase):
    def test_adapter_returns_serializable_common_result(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.png"
            image.touch()
            result = FakeAdapter().recognize(image).to_dict()

        self.assertEqual(result["engine"], "fake")
        self.assertEqual(result["text"], "বাংলা English")
        self.assertEqual(result["blocks"][0]["bbox"], (0, 0, 10, 10))
        self.assertGreaterEqual(result["elapsed_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
