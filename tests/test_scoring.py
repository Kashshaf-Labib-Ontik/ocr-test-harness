import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ocr_benchmark.scoring import overall_score, text_score


class ScoringTest(unittest.TestCase):
    def test_perfect_text_scores_one_hundred(self):
        metrics = {"cer": 0.0, "wer": 0.0, "exact_match": True}
        self.assertEqual(text_score(metrics), 100.0)

    def test_text_score_weights_are_explicit(self):
        metrics = {"cer": 0.10, "wer": 0.20, "exact_match": False}
        self.assertEqual(text_score(metrics), 78.0)

    def test_speed_is_relative_to_fastest_engine(self):
        score = overall_score(80.0, median_seconds=2.0, fastest_median_seconds=1.0)
        self.assertEqual(score["speed_score"], 50.0)
        self.assertEqual(score["overall_score"], 77.0)


if __name__ == "__main__":
    unittest.main()
