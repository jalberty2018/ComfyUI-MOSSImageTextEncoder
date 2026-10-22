"""Map weighted text after Qwen3.5 visual-token expansion; retain native conditioning."""
import math
import numbers
import re
from pathlib import Path

from .phrases import parse_phrases, ZERO_WEIGHT_BIAS

HOOK_ID = "moss_image_phrase_weights"


def prepare_prompt(text, enabled):
    if not enabled:
        return text, []
    if "<|" in text or re.search(r"\([^()]*\(", text):
        raise ValueError("Use plain text with non-nested (phrase:weight) annotations.")
    return parse_phrases(text)


def map_tokens(rendered, text_offset, spans, tokens, tokenizer):
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    batches = tokens.get("qwen35_9b")
    if batches is None or len(batches) != 1 or len(batches[0]) != len(ids):
        raise ValueError("Phrase weights require the native MOSS Qwen3.5-9B token layout.")
    images = []
    for i, (pair, expected) in enumerate(zip(batches[0], ids)):
        value = pair[0]
        if isinstance(value, dict):
            if value.get("type") != "image" or expected != 248056:
                raise ValueError("Unsupported embedded token in MOSS phrase mapping.")
            images.append(i)
        elif not isinstance(value, numbers.Integral) or value != expected:
            raise ValueError("Native tokens differ from the MOSS phrase map.")
    owners, mapped = set(), []
    for span in spans:
        start, end = text_offset + span["start"], text_offset + span["end"]
        rows = []
        for i, (a, b) in enumerate(offsets):
            if a >= end or b <= start:
                continue
            if rendered[a:start].strip() or rendered[end:b].strip():
                raise ValueError("Mark whole tokenizer pieces in weighted phrases.")
            if i in owners or (images and i <= images[-1]):
                raise ValueError("Weighted text must follow all images, without overlapping tokens.")
            owners.add(i)
            rows.append(i)
        if not rows:
            raise ValueError("Weighted phrase contains no tokens.")
        mapped.append({**span, "raw_rows": rows})
    return {"raw_count": len(ids), "images": images, "phrases": mapped}


def finish_mapping(layout, count):
    raw_count = layout["raw_count"]
    if count < raw_count or (not layout["images"] and count != raw_count):
        raise ValueError("Encoded rows differ from the native MOSS layout.")
    # Qwen expands image placeholders before the instruction. It preserves the
    # remaining text suffix and does not trim any prefix for MOSS.
    expansion = count - raw_count
    biases, report = [], []
    for span in layout["phrases"]:
        rows = [i + expansion for i in span["raw_rows"]]
        if any(i < 0 or i >= count for i in rows):
            raise ValueError("Weighted phrase row lies outside conditioning.")
        if span["weight"] != 1:
            value = math.log(span["weight"]) if span["weight"] else ZERO_WEIGHT_BIAS
            biases.extend((i, value) for i in rows)
        report.append({k: v for k, v in span.items() if k != "raw_rows"} | {"rows": rows})
    return tuple(biases), report


def build_layout(rendered, text_offset, spans, tokens):
    from transformers import Qwen2TokenizerFast
    from comfy.text_encoders import qwen35
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        Path(qwen35.__file__).parent / "qwen35_tokenizer", local_files_only=True)
    return map_tokens(rendered, text_offset, spans, tokens, tokenizer)


def apply_weighting(conditioning, layout, clean):
    import comfy.hooks
    from comfy.patcher_extension import WrappersMP
    from .phrase_attention import PhraseWrapper

    output, reports = [], []
    for tensor, metadata in conditioning:
        if tensor.ndim != 3 or tensor.shape[-1] != 4096:
            raise ValueError("Expected native MOSS conditioning with 4096 features.")
        count = tensor.shape[1]
        biases, phrases = finish_mapping(layout, count)
        item = [[tensor, metadata]]
        if biases:
            hook = comfy.hooks.TransformerOptionsHook(
                {"wrappers": {WrappersMP.DIFFUSION_MODEL: {
                    HOOK_ID: [PhraseWrapper(count, biases)]}}},
                hook_scope=comfy.hooks.EnumHookScope.HookedOnly)
            hook.hook_id = HOOK_ID
            group = comfy.hooks.HookGroup()
            group.add(hook)
            item = comfy.hooks.set_hooks_for_conditioning(item, group)
        output.extend(item)
        reports.append({"active": bool(biases), "token_count": count, "phrases": phrases})
    return output, {"active": any(r["active"] for r in reports),
                    "clean_text": clean, "entries": reports}
