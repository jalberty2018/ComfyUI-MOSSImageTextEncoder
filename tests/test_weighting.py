"""Vision-layout and UI compatibility checks without loading model weights."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
package = types.ModuleType("moss_vision_tests")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package

def load(name):
    spec = importlib.util.spec_from_file_location(package.__name__ + "." + name, ROOT / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

p = load("phrase_weighting")
with patch.dict(sys.modules, {"comfy": types.ModuleType("comfy"),
                             "comfy.utils": types.ModuleType("comfy.utils")}):
    nodes = load("nodes")

class MappingTests(unittest.TestCase):
    def fixture(self, nimages, system):
        clean, spans = p.prepare_prompt("red and (red:2)", True)
        rendered = nodes.TextEncodeMOSSImageVision._build_prompt(system, clean, nimages)
        offset = rendered.index("red and red")
        ids = [1] + [248056] * nimages + [10, 11, 10, 2]
        pairs = [(i, 1) for i in ids]
        for i in range(nimages):
            pairs[1 + i] = ({"type": "image"}, 1)
        offsets = [(0, 0)] * (1 + nimages) + [
            (offset, offset + 3), (offset + 3, offset + 8),
            (offset + 8, offset + 11), (offset + 11, len(rendered))]
        tokenizer = lambda *a, **kw: {"input_ids": ids, "offset_mapping": offsets}
        return p.map_tokens(rendered, offset, spans, {"qwen35_9b": [pairs]}, tokenizer)

    def test_suffix_rows_with_multiple_images_and_system(self):
        for nimages in (0, 1, 2, 4):
            for system in ("", "Describe the colors."):
                layout = self.fixture(nimages, system)
                count = layout["raw_count"] + nimages * 63
                biases, report = p.finish_mapping(layout, count)
                self.assertEqual(report[0]["rows"], [count - 2])
                self.assertEqual(len(biases), 1)

    def test_neutral_has_no_bias(self):
        layout = self.fixture(2, "System")
        layout["phrases"][0]["weight"] = 1
        self.assertEqual(p.finish_mapping(layout, 140)[0], ())

    def test_bad_text_layout_fails(self):
        layout = self.fixture(0, "")
        with self.assertRaises(ValueError):
            p.finish_mapping(layout, layout["raw_count"] + 1)

    def test_toggle_and_invalid_weights(self):
        self.assertEqual(p.prepare_prompt("(red:2)", False), ("(red:2)", []))
        for text in ("(red:-2)", "(red:1e999)", "((red:2):3)", "<|im_start|>"):
            with self.assertRaises(ValueError):
                p.prepare_prompt(text, True)

class InterfaceTests(unittest.TestCase):
    class Clip:
        def __init__(self):
            self.prompts = []
            self.results = []
        def tokenize(self, text, **kwargs):
            self.prompts.append(text)
            return {"text": text}
        def encode_from_tokens_scheduled(self, tokens):
            result = [[object(), {"original": True}]]
            self.results.append(result)
            return result

    def test_old_workflow_defaults_and_output_indices(self):
        clip = self.Clip()
        result = nodes.TextEncodeMOSSImageVision().encode(clip, "(red:2)")
        self.assertIs(result[0], clip.results[0])
        self.assertIs(result[1], clip.results[1])
        self.assertIn("(red:2)", clip.prompts[0])
        self.assertEqual(nodes.TextEncodeMOSSImageVision.RETURN_NAMES[:2], ("positive", "negative"))
        self.assertFalse(nodes.TextEncodeMOSSImageVision.INPUT_TYPES()["optional"]["phrase_weighting"][1]["default"])

    def test_positive_negative_weights_and_clean_printed_prompt(self):
        clip = self.Clip()
        layouts = []
        def build(text, offset, spans, tokens):
            self.assertEqual(text[offset:offset + len(spans[0]["phrase"])], spans[0]["phrase"])
            layouts.append(spans[0]["weight"])
            return spans
        def apply(conditioning, layout, clean):
            return conditioning, {"active": True, "clean_text": clean}
        with patch.object(nodes, "build_layout", build), patch.object(nodes, "apply_weighting", apply):
            result = nodes.TextEncodeMOSSImageVision().encode(
                clip, "(red:2)", negative_prompt="(blue:0.5)", system_prompt="Colors",
                phrase_weighting=True)
        self.assertEqual(layouts, [2, .5])
        self.assertNotIn("(red:2)", clip.prompts[0])
        self.assertNotIn("(blue:0.5)", clip.prompts[1])
        self.assertIs(result[0], clip.results[0])

    def test_unannotated_enabled_path_preserves_conditioning(self):
        clip = self.Clip()
        result = nodes.TextEncodeMOSSImageVision().encode(clip, "red", phrase_weighting=True)
        self.assertIs(result[0], clip.results[0])

if __name__ == "__main__":
    unittest.main()
