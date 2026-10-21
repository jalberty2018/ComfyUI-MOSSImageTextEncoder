"""MOSS-Image vision conditioning — reference image embedded into the prompt.

``TextEncodeMOSSImageVision`` pushes a reference image through the Qwen3.5-9B vision tower
and leaves its embeddings in the text conditioning, so the image is part of the prompt
instead of a VAE latent. MOSS-Image conditions on the last hidden state of the text encoder,
which is exactly what the core ``CLIPTextEncode`` produces for text-only prompts.
"""

import math

import comfy.utils

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# MOSS-Image conditions with the assistant turn inside an open think block.
THINK_OPEN = "<think>\n"


class TextEncodeMOSSImageVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "Load with CLIPLoader -> type 'moss_image' (Qwen3.5-9B text encoder).",
                }),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True, "dynamicPrompts": True, "default": "",
                    "tooltip": "Negative conditioning. Leave empty when the workflow drives CFG with 1.0.",
                }),
                "image": ("IMAGE", {
                    "tooltip": "Reference image(s). A batch is treated as multiple references. "
                               "The image is embedded into the prompt through the vision tower; "
                               "no VAE is used and nothing is edited.",
                }),
                "image_weight": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Reference influence, applied to the image before the vision tower. "
                               "1.0 = standard, <1.0 fades the reference toward gray so the encoder "
                               "keys off it less, >1.0 pushes away from gray.",
                }),
                "vision_megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Maximum resolution (in megapixels) for each reference image before "
                               "the Qwen3.5 vision encoder. Oversized images are downscaled; smaller "
                               "images are kept at native resolution.",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional system instruction prepended to the prompt. Empty matches "
                               "MOSS-Image's normal text-only conditioning.",
                }),
                "print_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print the fully assembled prompt sent to the Qwen3.5 text encoder.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "model/conditioning/moss_image"
    DESCRIPTION = (
        "MOSS-Image vision conditioning. Takes a reference image (or batch), extracts visual "
        "embeddings through the Qwen3.5-9B vision tower and leaves them in the text prompt. "
        "No VAE / no reference latents."
    )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _build_prompt(system, text, num_images):
        refs = " ".join(VISION_BLOCK for _ in range(num_images))
        prompt = ""
        if system.strip():
            prompt += "<|im_start|>system\n" + system.strip() + "<|im_end|>\n"
        prompt += "<|im_start|>user\n" + refs + text + "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n" + THINK_OPEN
        return prompt

    @staticmethod
    def _prepare_references(image, vision_megapixels, image_weight):
        """Downscale each reference to the megapixel cap and build the vision inputs.

        Never upscales, only shrinks images that exceed the cap. The Qwen3.5 processor
        internally snaps to a multiple of 16. ``image_weight`` fades the reference toward
        mid-gray before the vision tower (the tower normalizes with mean/std 0.5, so gray
        reads as a zero vision signal); values above 1 push away from gray.
        """
        if image is None or image.numel() == 0:
            return []

        if image.dim() == 3:
            image = image.unsqueeze(0)

        images_vl = []
        max_pixels = int(vision_megapixels * 1024 * 1024)

        for i in range(image.shape[0]):
            sample = image[i:i + 1]                       # (1, H, W, C)
            samples = sample.movedim(-1, 1)               # (1, C, H, W)
            src_h, src_w = samples.shape[2], samples.shape[3]

            scale_by = min(1.0, math.sqrt(max_pixels / (src_h * src_w)))
            width = max(1, round(src_w * scale_by))
            height = max(1, round(src_h * scale_by))

            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")

            rgb = s.movedim(1, -1)[:, :, :, :3]           # (1, H, W, 3)
            if sample.shape[-1] > 3:                       # vision sees alpha over white
                alpha = sample[:, :, :, 3:]
                alpha = comfy.utils.common_upscale(alpha.movedim(-1, 1), width, height, "area", "disabled").movedim(1, -1)
                rgb = rgb * alpha + (1.0 - alpha)
            if image_weight != 1.0:
                rgb = (0.5 + image_weight * (rgb - 0.5)).clamp(0.0, 1.0)
            images_vl.append(rgb)

        return images_vl

    @staticmethod
    def _fp8_hint(exc, images_vl):
        if images_vl and isinstance(exc, NotImplementedError) and "Float8" in str(exc):
            return RuntimeError(
                "MOSS-Image: the Qwen3.5-9B text encoder is loaded in FP8, which cannot run the vision "
                "path ('add_stub not implemented for Float8_e4m3fn'). Load a bf16/fp16/int8 Qwen3.5-9B "
                "text encoder via CLIPLoader (type 'moss_image') when using image references."
            )
        return None

    # ------------------------------------------------------------------ main

    def encode(self, clip, prompt, negative_prompt="", image=None, image_weight=1.0,
               vision_megapixels=1.0, system_prompt="", print_prompt=False):

        use_images = image is not None and image_weight > 0.0 and image.numel() > 0
        images_vl = self._prepare_references(image if use_images else None, vision_megapixels, image_weight)

        positive_prompt = self._build_prompt(system_prompt, prompt, len(images_vl))
        negative_text = self._build_prompt(system_prompt, negative_prompt, len(images_vl))

        if print_prompt:
            print("\n========== Text Encode (MOSS-Image Vision) -> Qwen3.5 prompt ==========")
            print(positive_prompt)
            print("---- references: {} | image_weight: {} | vision_megapixels: {} ----".format(
                len(images_vl), image_weight, vision_megapixels))
            print("=======================================================================\n")

        def _encode(text):
            tokens = clip.tokenize(text, images=images_vl)
            try:
                return clip.encode_from_tokens_scheduled(tokens)
            except NotImplementedError as exc:
                hint = self._fp8_hint(exc, images_vl)
                if hint is not None:
                    raise hint from exc
                raise

        positive = _encode(positive_prompt)
        negative = _encode(negative_text)

        return (positive, negative)


NODE_CLASS_MAPPINGS = {
    "TextEncodeMOSSImageVision": TextEncodeMOSSImageVision,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeMOSSImageVision": "Text Encode (MOSS-Image Vision)",
}
