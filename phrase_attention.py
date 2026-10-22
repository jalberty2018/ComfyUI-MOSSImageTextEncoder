"""Attention implementation from ComfyUI-Phrase-Weight-Conditioning (MIT)."""
import torch

class PhraseWrapper:
    def __init__(self, count, biases):
        self.count, self.biases = count, tuple(biases)

    def __call__(self, executor, x, timestep, context=None, transformer_options=None, **kwargs):
        from comfy.ldm.moss_image.model import MOSSImage
        if not isinstance(executor.class_obj, MOSSImage):
            raise ValueError("Phrase Weights requires the native MOSS-Image model.")
        if context is None or context.shape[1] != self.count:
            raise ValueError("Text rows changed after phrase encoding.")
        options = dict(transformer_options or {})
        previous = options.get("optimized_attention_override")
        target = x.shape[-2] * x.shape[-1]
        ref_count = 0
        total = target + ref_count + self.count

        def attention(function, q, k, v, heads, mask=None, **extra):
            # Skip the two text-only embedder blocks; only joint blocks have this length.
            if k.shape[1] == total:
                if q.ndim != 3 or q.shape[1] != total:
                    raise ValueError("Unsupported MOSS-Image attention layout.")
                # Split query rows to avoid a full quadratic bias allocation.
                bias = q.new_zeros((1, 1, 1, total))
                for row, value in self.biases:
                    bias[..., target + ref_count + row] = value
                def run(a, b, weighted):
                    part_mask = mask
                    if part_mask is not None:
                        if part_mask.ndim == 2:
                            part_mask = part_mask[None, None]
                        elif part_mask.ndim == 3:
                            part_mask = part_mask[:, None]
                        if part_mask.shape[-2] != 1:
                            part_mask = part_mask[..., a:b, :]
                    if weighted:
                        if part_mask is None:
                            part_mask = bias
                        elif part_mask.dtype == torch.bool:
                            part_mask = bias.masked_fill(~part_mask, float("-inf"))
                        else:
                            part_mask = part_mask + bias
                    if previous:
                        return previous(function, q[:, a:b], k, v, heads, mask=part_mask, **extra)
                    return function(q[:, a:b], k, v, heads, mask=part_mask, **extra)
                return torch.cat((run(0, target, True), run(target, total, False)), dim=1)
            if previous:
                return previous(function, q, k, v, heads, mask=mask, **extra)
            return function(q, k, v, heads, mask=mask, **extra)

        options["optimized_attention_override"] = attention
        return executor(x, timestep, context, options, **kwargs)



