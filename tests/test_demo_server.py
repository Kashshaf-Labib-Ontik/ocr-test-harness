import base64
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_local_demo.py"
SPEC = importlib.util.spec_from_file_location("run_local_demo", SCRIPT)
demo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(demo)


class DemoServerTest(unittest.TestCase):
    def test_sample_catalog_only_returns_existing_images(self):
        samples = demo.sample_catalog()
        self.assertEqual(len(samples), 6)
        self.assertTrue(all(item["url"].startswith("/samples/") for item in samples))

    def test_decode_upload_accepts_image_payload(self):
        content, suffix = demo.decode_upload(
            {
                "filename": "policy.png",
                "content_base64": base64.b64encode(b"not-verified-here").decode("ascii"),
            }
        )
        self.assertEqual(content, b"not-verified-here")
        self.assertEqual(suffix, ".png")

    def test_decode_upload_rejects_unsupported_extension(self):
        with self.assertRaisesRegex(ValueError, "PNG"):
            demo.decode_upload({"filename": "policy.pdf", "content_base64": "YWJj"})


if __name__ == "__main__":
    unittest.main()
