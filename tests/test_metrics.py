import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ocr_benchmark.metrics import evaluate_text, normalize_text


class TextMetricsTest(unittest.TestCase):
    def test_exact_bangla_and_english_match(self):
        result = evaluate_text("নীতিমালা Policy", "নীতিমালা Policy")
        self.assertEqual(result["cer"], 0.0)
        self.assertEqual(result["wer"], 0.0)
        self.assertTrue(result["exact_match"])

    def test_whitespace_is_normalized(self):
        self.assertEqual(normalize_text("Policy\n\t text"), "Policy text")

    def test_single_word_substitution(self):
        result = evaluate_text("loan policy", "loan rules")
        self.assertEqual(result["wer"], 0.5)
        self.assertFalse(result["exact_match"])

    def test_empty_reference_is_well_defined(self):
        self.assertEqual(evaluate_text("", "")["cer"], 0.0)
        self.assertEqual(evaluate_text("", "unexpected")["cer"], 1.0)


if __name__ == "__main__":
    unittest.main()
