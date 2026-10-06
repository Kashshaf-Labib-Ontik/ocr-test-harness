import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ocr_benchmark.adapters import EasyOCRAdapter


class FakeReader:
    def readtext(self, image_path, detail, paragraph):
        return [
            ([[20, 30], [90, 30], [90, 50], [20, 50]], "Policy", 0.9),
            ([[10, 5], [100, 5], [100, 25], [10, 25]], "নীতিমালা", 0.8),
        ]


class EasyOCRAdapterTest(unittest.TestCase):
    def test_output_is_sorted_and_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.png"
            image.touch()
            result = EasyOCRAdapter(reader=FakeReader()).recognize(image)

        self.assertEqual(result.text, "নীতিমালা\nPolicy")
        self.assertEqual(result.blocks[0].bbox, (10.0, 5.0, 100.0, 25.0))
        self.assertEqual(result.metadata["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
