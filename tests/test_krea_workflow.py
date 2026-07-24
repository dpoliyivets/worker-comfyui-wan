import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import krea_workflow


VALID_REQUEST = {
    "cfg": 1,
    "loras": [
        {"source": "r2://loras/krea2/refusal-reduction-v1.safetensors", "strength": 1.0},
        {"source": "r2://loras/krea2/mystic-xxx-v3.safetensors", "strength": 0.5},
    ],
    "negatives": "",
    "prompt": "a test prompt",
    "resolution": {"height": 704, "width": 1216},
    "sampler": "euler",
    "scheduler": "beta",
    "seed": 7,
    "steps": 12,
}


class TestBuildKreaWorkflow(unittest.TestCase):
    def test_builds_bench_graph_shape(self):
        wf = krea_workflow.build_krea_workflow(VALID_REQUEST, lora_filenames=[])
        self.assertEqual(wf["1"]["class_type"], "UNETLoader")
        self.assertEqual(wf["2"]["class_type"], "CLIPLoader")
        self.assertEqual(wf["2"]["inputs"]["type"], "krea2")
        self.assertEqual(wf["3"]["class_type"], "VAELoader")
        self.assertEqual(wf["6"]["inputs"]["width"], 1216)
        self.assertEqual(wf["6"]["inputs"]["height"], 704)
        sampler = wf["7"]["inputs"]
        self.assertEqual(sampler["seed"], 7)
        self.assertEqual(sampler["steps"], 12)
        self.assertEqual(sampler["cfg"], 1)
        self.assertEqual(sampler["sampler_name"], "euler")
        self.assertEqual(sampler["scheduler"], "beta")
        self.assertEqual(wf["8"]["class_type"], "VAEDecode")
        self.assertEqual(wf["9"]["class_type"], "SaveImage")

    def test_no_loras_wires_sampler_to_unet(self):
        wf = krea_workflow.build_krea_workflow(VALID_REQUEST, lora_filenames=[])
        self.assertEqual(wf["7"]["inputs"]["model"], ["1", 0])

    def test_lora_chain_wires_in_order(self):
        wf = krea_workflow.build_krea_workflow(
            VALID_REQUEST,
            lora_filenames=["refusal-reduction-v1.safetensors", "mystic-xxx-v3.safetensors"],
        )
        first = wf["10"]
        second = wf["11"]
        self.assertEqual(first["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(first["inputs"]["model"], ["1", 0])
        self.assertEqual(first["inputs"]["lora_name"], "refusal-reduction-v1.safetensors")
        self.assertEqual(first["inputs"]["strength_model"], 1.0)
        self.assertEqual(second["inputs"]["model"], ["10", 0])
        self.assertEqual(second["inputs"]["strength_model"], 0.5)
        self.assertEqual(wf["7"]["inputs"]["model"], ["11", 0])

    def test_prompt_and_negatives_encoded(self):
        wf = krea_workflow.build_krea_workflow(VALID_REQUEST, lora_filenames=[])
        self.assertEqual(wf["4"]["inputs"]["text"], "a test prompt")
        self.assertEqual(wf["5"]["inputs"]["text"], "")


class TestEnsureLoras(unittest.TestCase):
    def test_downloads_missing_and_skips_cached(self):
        client = MagicMock()

        def fake_download(bucket, key, path):
            with open(path, "wb") as f:
                f.write(b"weights")

        client.download_file.side_effect = fake_download
        with tempfile.TemporaryDirectory() as tmp:
            cached = os.path.join(tmp, "mystic-xxx-v3.safetensors")
            with open(cached, "wb") as f:
                f.write(b"x")
            names = krea_workflow.ensure_loras(
                VALID_REQUEST["loras"], lora_dir=tmp, s3_client=client, bucket="bkt"
            )
        self.assertEqual(
            names, ["refusal-reduction-v1.safetensors", "mystic-xxx-v3.safetensors"]
        )
        client.download_file.assert_called_once()
        args = client.download_file.call_args[0]
        self.assertEqual(args[0], "bkt")
        self.assertEqual(args[1], "loras/krea2/refusal-reduction-v1.safetensors")

    def test_rejects_non_r2_sources(self):
        with self.assertRaises(ValueError):
            krea_workflow.ensure_loras(
                [{"source": "http://evil/x.safetensors", "strength": 1}],
                lora_dir="/tmp",
                s3_client=MagicMock(),
                bucket="bkt",
            )


class TestExpandKreaInput(unittest.TestCase):
    def test_missing_fields_raise(self):
        with self.assertRaises(ValueError):
            krea_workflow.expand_krea_input({"prompt": "x"}, ensure=lambda loras: [])

    def test_valid_input_returns_workflow_job(self):
        expanded = krea_workflow.expand_krea_input(
            VALID_REQUEST, ensure=lambda loras: ["a.safetensors", "b.safetensors"]
        )
        self.assertIn("workflow", expanded)
        self.assertEqual(expanded["workflow"]["7"]["inputs"]["model"], ["11", 0])


if __name__ == "__main__":
    unittest.main()
