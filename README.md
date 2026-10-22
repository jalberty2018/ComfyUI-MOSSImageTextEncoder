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
| `phrase_weighting` | BOOLEAN | Enable `(phrase:weight)` in positive and negative prompts. Default off preserves existing workflows. |

**Outputs:** `positive`, `negative` (CONDITIONING), `phrase_report` (STRING).
The first two outputs and the node ID are unchanged.

## Phrase weighting

Enable **phrase_weighting** in the existing **Text Encode (MOSS-Image Vision)** node.
Write weights directly in the prompt:

```text
A person wearing a (red jacket:2) beside a (blue wall:0.7).
```

The negative prompt supports independent weights. Use separate, non-nested whole
words or phrases. Only the marked occurrence is weighted; the system prompt is
not parsed for weights. With the switch off, text is passed through unchanged.

| Weight | Effect |
|---|---|
| `1` | Neutral; no attention hook is attached if all weights are neutral |
| Above `1` | More relative attention to the selected words |
| Between `0` and `1` | Less relative attention |
| `0` | Strong suppression using a -80 attention-score bias; not guaranteed removal |

This uses the MOSS attention implementation from
[ComfyUI-Phrase-Weight-Conditioning](https://github.com/jalberty2018/ComfyUI-Phrase-Weight-Conditioning),
whose standalone MOSS node has been user-tested. No separate node pack is needed.
Do not apply a second phrase-weight node to the same conditioning.

The integration verifies native Qwen3.5 token IDs and maps text positions after
visual-token expansion. It supports multiple image references and an optional
system prompt. Image embeddings are preserved. The weight changes generated-image
attention to the selected text keys; it does not rescale embeddings or change
`image_weight`, which still controls image preprocessing.

Connect `phrase_report` to a text display to inspect the cleaned positive/negative
prompts and final token rows. `active` reports an attached hook, not sampler execution.
Run prompt expansion before inserting annotations.

**Validation:** the inherited attention behavior and integration mapping/interface
have CPU tests. This vision integration still needs a GPU comparison with actual
reference images. Keep seed/settings fixed and compare plain `red` with
`(red:1)`, then `(red:2)` and `(red:5)`. Start without other attention patches;
the attention backend must support additive masks.

Developer checks: `python -m unittest discover -s tests` (NumPy required).


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
git clone https://github.com/jalberty2018/ComfyUI-MOSSImageTextEncoder.git
```

Restart ComfyUI — that's it, the node is installed. Then drag `workflow.json` in and go.

To update an existing checkout of this fork, run `git pull --ff-only` inside its
folder and restart ComfyUI. Existing workflows keep phrase weighting off until
you enable the new switch.

## License

MIT — see [LICENSE](LICENSE).
The phrase-weight parser and inherited attention code retain their upstream
MIT notice in [LICENSE-phrase-weights-MIT](LICENSE-phrase-weights-MIT).
