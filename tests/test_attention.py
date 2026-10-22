"""CPU contract checks using NumPy; not an end-to-end ComfyUI/GPU test."""
import ast
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).parents[1]

class Tensor(np.ndarray):
    def new_zeros(self, shape):
        return np.zeros(shape, dtype=self.dtype).view(Tensor)

def tensor(data):
    return np.asarray(data, dtype=np.float32).view(Tensor)

class MOSSImage:
    pass

module = ast.parse((ROOT / "phrase_attention.py").read_text(encoding="utf-8"))
wrapper_ast = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "PhraseWrapper")
namespace = {"torch": types.SimpleNamespace(cat=lambda xs, dim: np.concatenate(xs, axis=dim), bool=np.bool_)}
exec(compile(ast.Module(body=[wrapper_ast], type_ignores=[]), "phrase_attention.py", "exec"), namespace)
Wrapper = namespace["PhraseWrapper"]

def attention(function_unused, q, k, v, heads, mask=None, **kwargs):
    scores = q @ k.transpose(0, 2, 1)
    if mask is not None:
        scores = scores + mask[:, 0]
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return (scores / scores.sum(axis=-1, keepdims=True)) @ v

class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {
            "comfy.ldm.moss_image.model": types.SimpleNamespace(MOSSImage=MOSSImage)})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def execute(self, wrapper, key_count=4, previous=None, model=None):
        options = {} if previous is None else {"optimized_attention_override": previous}
        class Executor:
            class_obj = model or MOSSImage()
            def __call__(self, x, timestep, context, transformer_options, **kwargs):
                q = tensor(np.zeros((1, key_count, 4)))
                v = tensor(np.eye(key_count)[None])
                return transformer_options["optimized_attention_override"](
                    lambda *a, **kw: attention(None, *a, **kw), q, q, v, 1)
        result = wrapper(Executor(), tensor(np.zeros((1, 128, 1, 2))), None,
                         tensor(np.zeros((1, 2, 4096))), options)
        self.assertEqual(set(options), set() if previous is None else {"optimized_attention_override"})
        return result

    def test_image_queries_only_and_exact_odds(self):
        result = self.execute(Wrapper(2, [(0, math.log(3))]))
        np.testing.assert_allclose(result[0, :2], [[1/6, 1/6, 1/2, 1/6]] * 2)
        np.testing.assert_allclose(result[0, 2:], np.full((2, 4), .25))

    def test_text_embedder_is_untouched(self):
        result = self.execute(Wrapper(2, [(0, math.log(20))]), key_count=2)
        np.testing.assert_allclose(result, np.full((1, 2, 2), .5))

    def test_zero_suppresses_selected_key(self):
        result = self.execute(Wrapper(2, [(1, -80)]))
        self.assertLess(result[0, 0, 3], 1e-30)

    def test_previous_override_is_preserved(self):
        calls = []
        def previous(function, *args, **kwargs):
            calls.append(args[0].shape[1])
            return function(*args, **kwargs)
        self.execute(Wrapper(2, [(0, 1)]), previous=previous)
        self.assertEqual(calls, [2, 2])

    def test_wrong_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "native MOSS"):
            self.execute(Wrapper(2, [(0, 1)]), model=object())

    def test_changed_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "Text rows changed"):
            self.execute(Wrapper(3, [(0, 1)]))

if __name__ == "__main__":
    unittest.main()

