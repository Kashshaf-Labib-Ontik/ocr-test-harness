import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ocr_benchmark.preprocessing import PREPROCESS_MODES, preprocess_image


class PreprocessingTest(unittest.TestCase):
    def test_all_modes_preserve_page_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            page = np.full((80, 120, 3), 255, dtype=np.uint8)
            cv2.putText(page, "Policy", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.imwrite(str(path), page)

            outputs = {mode: preprocess_image(path, mode) for mode in PREPROCESS_MODES}

        self.assertEqual(outputs["original"].shape[:2], (80, 120))
        self.assertEqual(outputs["enhance"].shape, (80, 120))
        self.assertEqual(outputs["binary"].shape, (80, 120))


if __name__ == "__main__":
    unittest.main()
