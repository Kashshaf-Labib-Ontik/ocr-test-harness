import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ocr_benchmark.adapters import TesseractAdapter


TSV = """level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
5\t1\t1\t1\t1\t1\t10\t20\t50\t20\t90\tগ্রাহক
5\t1\t1\t1\t1\t2\t70\t20\t45\t20\t80\tPolicy
"""


def fake_runner(command):
    return subprocess.CompletedProcess(command, 0, stdout=TSV, stderr="")


class TesseractAdapterTest(unittest.TestCase):
    def test_tsv_words_are_grouped_into_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.png"
            image.touch()
            result = TesseractAdapter(runner=fake_runner).recognize(image)

        self.assertEqual(result.text, "গ্রাহক Policy")
        self.assertEqual(result.blocks[0].bbox, (10.0, 20.0, 115.0, 40.0))
        self.assertAlmostEqual(result.blocks[0].confidence, 0.85)


if __name__ == "__main__":
    unittest.main()
