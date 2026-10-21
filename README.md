# ComfyUI-MOSSImageTextEncoder

Feeds a reference image through the Qwen3.5-9B **vision tower** and inserts its visual embeddings directly into the text prompt. The image conditions the generation natively through the text encoder.

<!-- SHOWCASE COLLAGE -->
![MOSS-Image Vision Workflow](assets/workflow.webp)

**Note:** This node is for vision/conditioning, not image editing.

## Workflow
A ready-to-use example workflow is included in the repository: `workflow.json`.

Simply drag and drop the `workflow.json` file directly into your ComfyUI workspace to load the entire setup.

## Requirements

* `qwen3.5_9b_*.safetensors` text encoder, `moss_image_*.safetensors` diffusion model,
  `flux2-vae*.safetensors` (for decoding only).
* Load the text encoder through **CLIPLoader** with type **`moss_image`**.
* Latents: use the core **Empty Flux 2 Latent** node (128 channels).

## Text Encode (MOSS-Image Vision)

| Input | Type | Notes |
|-------|------|-------|
| `clip` | CLIP | CLIPLoader → type `moss_image`. |
| `prompt` | STRING | Your text prompt. |
| `negative_prompt` | STRING | Optional negative conditioning. |
| `image` | IMAGE | Optional reference image. A batch is treated as multiple references. |
| `image_weight` | FLOAT | Reference influence applied before the vision tower: `0` ignores it, `1` standard, `<1` fades toward gray, `>1` pushes away from gray. |
| `vision_megapixels` | FLOAT | Max size per reference before the vision tower. Oversized images are downscaled, smaller ones kept native (never upscaled). Default `1.0`. |
| `system_prompt` | STRING | Optional system instruction prepended to the prompt. Empty matches MOSS-Image's normal conditioning. |
| `print_prompt` | BOOLEAN | Dump the assembled Qwen3.5 prompt to the console. |

**Outputs:** `positive`, `negative` (CONDITIONING).



## Typical wiring

```
UNETLoader   (moss_image_sft_*)
CLIPLoader   (type: moss_image, qwen3.5_9b_*)
VAELoader    (flux2-vae)                          # decode only
LoadImage ─┐
           ▼
  Text Encode (MOSS-Image Vision) ── positive/negative ─┐
Empty Flux 2 Latent ────────────────────────────────────┤
                                                        ▼
                                    KSampler ─► VAEDecode ─► PreviewImage
```

Start with `image_weight = 1.0`, `vision_megapixels = 1.0`, denoise `1.0`.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ThetaCursed/ComfyUI-MOSSImageTextEncoder.git
```

Restart ComfyUI — that's it, the node is installed. Then drag `workflow.json` in and go.

## License

MIT — see [LICENSE](LICENSE).
