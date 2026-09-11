# vpipe-sensenova-u1.5

[SenseNova-U1.5-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)
as a [vpipe](https://github.com/tgo-app-dev/vpipe) plugin: image generation and
editing on
Apple Silicon, through vpipe's metal-compute backend.

```
  text-prompt ──▶ sensenova-u1.5-generate ──▶ save-image
   load-image ──▶  (reference, for editing)
                            ▲
  sensenova-u1.5-model-config
```

Text-to-image and image editing, both through one stage.

## Start here

**[docs/SENSENOVA-U1.5.md](docs/SENSENOVA-U1.5.md)** — how to fetch the
checkpoint, size a render, quantize it, and what every knob does. Five
ready-to-run graphs live in [docs/pipelines/](docs/pipelines/): text-to-image,
image edit, multi-reference edit, a live preview, and the 8-bit preparation.
Read that first; the rest of this file is what the port is made of and how it
was checked.

## What this model is

Not a diffusers pipeline. There is **no VAE**, no ViT tower, and no
`transformer/` + `text_encoder/` + `vae/` split — one flat 50.2 GB checkpoint
holding:

- a **Qwen3-shaped 42-layer Mixture-of-Transformers backbone**: every weight
  exists twice, once for *understanding* tokens and once (`_mot_gen`) for the
  image being generated, with joint attention across both;
- a **three-way split RoPE** — each 128-wide head carries a sequence position, a
  row and a column, on two different frequency ladders;
- a **pixel-space flow-matching head**, so generation emits RGB directly.

The conditioning is therefore a **KV cache** produced by the same weights that
denoise, not a tensor — which is why this ships one stage rather than plugging
into vpipe's `diffusion-conditioner → generate-image → vae-decode` chain.

`ARCHITECTURE.md` documents all of it, marked by how each claim is known
(checkpoint / reference source / derived), including the traps that cost
debugging time.

## Build

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/vpipe-install
cmake --build build -j
```

A plugin must be built against the vpipe it deploys with — the ABI handshake is
strict equality, so the requirement is **exactly the ABI of that vpipe**, never
a minimum. That is **3** today. The number moves whenever the host's plugin
surface does, and a plugin built against an older one is refused at load rather
than crashed.

## Run

```sh
vpipe --plugin build/vpipe-sensenova-u1.5.so \
      --launch docs/pipelines/sensenova-u1.5-text-to-image.vpipeline
```

Set `hf_dir` in the pipeline to the checkpoint directory, or wire a
`model-select` source to the generate stage's `model` iport.

## Stages

| stage | what |
|---|---|
| `sensenova-u1.5-generate` | prompt → image, and prompt + reference image(s) → edited image. Owns prefill, denoise and the pixel head; emits a planar U8 RGB `TensorBeat` tagged `rgb-frames`, the same payload `vae-decode` produces, so the stock `save-image` / `compare-image` consume it unchanged. Its own keys are the geometry and the run: `hf_dir`, `width`, `height`, `steps`, `seed`, `unload_when_idle`, and the two acceleration flags below. |
| `sensenova-u1.5-model-config` | the sampling knobs that are this family's own: `cfg_scale`, `img_cfg_scale`, `cfg_norm`, `timestep_shift`, `cfg_interval_lo`, `cfg_interval_hi`, `t_eps`, `init_noise`. |

`hf_dir` also joins the host's **shared-model channel**, so a `model-select`
source feeding a graph fills this field in the composer's picker the way it
fills every other family's.

### Editing

Wire `load-image` to the generate stage's `reference0` iport (and `reference1`
for a second image) — that is the whole switch from generation to editing. Any
input size: the stage applies the model's own `smart_resize` and ImageNet
normalisation, which are model facts and not a graph's business.

Reference images enter as *understanding* tokens, so the prefill runs under
**block-causal** attention (an image's tokens share one position, and the
mask-free causal kernel used for text-only prefixes would attend only the
earlier half of each image). `img_cfg_scale > 1` adds a third prefix and a third
forward per step.

## Accelerated modes

Two lossy tiers on the generate stage, **both off by default**, and
independent of each other: `i8_gemm` changes how a weight is multiplied,
`sage_attn` how a score is computed. `sage_dense_layers` leaves a prefix of
the backbone in bf16. `docs/SENSENOVA-U1.5.md` has what each one does and the
env overrides.

**Matrix cores are the whole condition for both.** The int8 fragment MMA has
no ALU fallback, so on a box without them the tier declines with a message and
the model runs bf16 — which is why nothing here is measured on the M4 Pro.

MEASURED on the M5 through `u15-perf-test` at 4096 tokens, interleaved
off/on/off/on: **148.0 / 147.8 ms per layer off against 100.9 / 102.1 on**,
a **1.46x** on the projections, with the attention arm unchanged at 28 ms
either way — which is what says the GEMMs moved and not the clock. **End to
end it does not show** on a 16 GB box (55–56 s either way at 1024²): the
33 GB checkpoint streams per pass at 1.31 GB resident, so the run is
read-bound and the compute win hides behind the I/O. The pipeline-level
number wants the 64 GB box with the model preloaded.

Sage reaches the **bidirectional pass and nothing else** here — the causal and
block-causal branches take their own kernels, and the int8 twin is a function
constant on the flash one.

## Cost

Measured on an M4 Pro (64 GB), bf16:

| | |
|---|---|
| load | ~55 s, **31.6 GB** resident (14.1 GB of it converted from F32 at bind) |
| 512², 20 steps | ~26 s (1.3 s/step) |
| 1024², 20 steps | ~110 s (5.5 s/step) |
| 384², 8 steps, edit | ~15 s (a reference adds a prefill and lengthens every step's attention) |

Those are the PRELOADED figures. The stage streams the 42 layers when the
checkpoint will not sit comfortably beside everything else — which on a 64 GB
box is the bf16 pack — and then grows the resident set back into whatever RAM
turns out to be there. Streaming costs one extra read of the checkpoint, paid
once: +5.5 s at 4 steps and +5.7 s at 20, measured at 256² on the w8 pack.
Streamed output is **bit-identical** to preloaded. See `ARCHITECTURE.md` §13.

Attention runs on `attn_steel_h_bd128_bf16`, the flash kernel the in-tree image
DiTs use — **24× the scalar SDPA** the port was first written against, and 3.2×
end to end at 1024². The model is GEMM-bound at 67–84% of this box's roofline;
`ARCHITECTURE.md` §11 has the full measurement, including two things measured
and *not* adopted and two harness faults that nearly produced false wins.

`256²` is the model's documented minimum and generates badly there — see
`ARCHITECTURE.md` §11. Use 384² or above.

The checkpoint is 50.2 GB on disk because the generation expert ships F32 while
the understanding expert ships BF16; both are converted to bf16 at load, which
is what the reference does too. That conversion is also why streaming a bf16
pack reads more than it holds — no raw-refill path can serve an F32 tensor
into a bf16 destination — and one more reason to quantize first.

## Quantization

`model-quantize` handles this checkpoint through its general path — no
`QuantizableFamily` is registered, because that registry is for multi-component
repacks and this model is one flat directory.

```sh
vpipe --plugin build/vpipe-sensenova-u1.5.so \
      --launch docs/pipelines/prepare-sensenova-u1.5-8bit.vpipeline
```

| | resident | load | vs bf16 |
|---|---:|---:|---|
| bf16 | 31.6 GB | 55 s | — |
| **w8g64** | **17.4 GB** | **6 s** | 1.8/255 at 512²/20 steps, correlation **0.9994** |

**Use 8-bit.** 4-bit produces structured garbage on this model at both group
sizes and through both kernel tiles — see `docs/SENSENOVA-U1.5.md` for the full
table and what that does and does not establish. The generate stage warns when
it loads a 4-bit pack.

## Verification

Every layer is checked against the reference implementation rather than against
itself. The goldens are produced by driving SenseNova's own classes at small
dimensions and dumping every weight, input and output; the generator
additionally **asserts its own reading of the operation order** against the
reference module, so a misreading fails at generation instead of becoming a
golden the port faithfully reproduces.

The tests read a golden directory and a checkpoint from the environment:

```sh
export VPIPE_U15_TEST_MODEL_PATH=/path/to/SenseNova-U1.5-8B-MoT
export VPIPE_U15_GOLDEN_DIR=$GOLDENS
ctest --test-dir build --output-on-failure
```

Without those set the golden-backed tests skip, and the rest of the suite
runs.

| test | checks | against |
|---|---|---|
| `u15-config` | geometry, noise scale, and that detection **cannot pass by default** | synthetic look-alikes + the real checkpoint |
| `u15-ref` | the CPU reference: split RoPE, MoT layer, patch embedder, pixel head, schedule | the reference implementation's goldens |
| `u15-prompt` | tokenisation and the chat template, **exactly** | the reference's own `AutoTokenizer` |
| `u15-weights` | tensor names, shapes, the dtype split, and that the two experts are distinct | the real checkpoint |
| `u15-kernels` | the plugin's kernels **and this code's use of libvpipe's** | the CPU reference |
| `u15-sage` | the int8 attention PLAN — which kernel a shape resolves to, with which tiles, remembering which strides | itself, deliberately: no checkpoint, so it runs everywhere. On a GPU without matrix cores what it proves is that asking for a tier that cannot run changed nothing |
| `u15-image` | patch embedder and pixel head at full width | the CPU reference, real weights |
| `u15-layer0` | one MoT layer, both experts, 4096 wide | the CPU reference, real weights |
| `u15-generate` | end to end, text-to-image | structural properties of the image |
| `u15-edit` | end to end, editing | two different references through one prompt and one seed must give different images |
| `u15-stream` | layer streaming and the stream/preload verdict | the preloaded render, **bit for bit** |
| `u15-perf` | where the time goes, at real shapes | reports rates; asserts only that a fast kernel is not slower than the one it replaced |
| `u15-quant` | a quantized pack | one render from each of the dense and quantized packs, same prompt and seed |

Tests gate on env vars and **skip vacuously** when unset — read the output, not
the exit code. Each says which it did.

Two more targets **build but are not registered with ctest**, because neither
asserts anything a CI run should gate on: `u15-io-bench` times mmap+memcpy
against pread per drive, and `u15-m5-test` reports what a matrix-core GPU
offers. Run them by hand from `build/`.

## Not implemented

**A 4-bit recipe.** Plain affine 4-bit does not work here; whether AWQ or a
mixed pack (the generation expert kept at 8-bit) rescues it is untested.

**Multimodal understanding (VQA).** The und expert is fully implemented and
verified — it is what runs the text prefix — so exposing understanding is a
`ModelExec` plus KV plumbing rather than new model work.

**Think mode** — the model can reason in a `<think>` block before generating.
That needs the `lm_head` (currently not even bound, saving ~1.2 GB) and a text
sampling loop.

**Bit-exact comparison against the reference.** torch's Philox RNG is not
reproduced here, so the same seed gives a different sample. The generate stage's
`init_noise` key exists as the injection point that would make an image-level
comparison meaningful.

**An end-to-end number for the accelerated tiers.** `i8_gemm` is measured per
layer on the M5 and `sage_attn` only as a plan; neither has a pipeline-level
figure, and neither has been checked against the golden render. That is why
both default to false.

**Prefetch overlap while streaming.** A streamed layer is read, used, and only
then is the next one read, so the read is serial with the compute. Overlapping
them needs a second in-flight slot and a fence per slot — see `ARCHITECTURE.md`
§13.6.

## Licence

This plugin's source is Apache-2.0 — see [`LICENSE`](LICENSE), and
[`NOTICE`](NOTICE) for the attribution a redistribution carries with it.
The SenseNova-U1.5 weights are Apache-2.0 as well, and are not
distributed here.
