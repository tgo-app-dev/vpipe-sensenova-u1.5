# SenseNova-U1.5 in vpipe

How to run the model. For *what the model is* and how the port was derived, see
`ARCHITECTURE.md` at the repository root.

## Fetching the checkpoint

The plugin contributes a catalogue entry, so once it is loaded the model shows
up in `vpipe --list-models` and can be fetched the way a built-in one is:

```sh
vpipe --plugin build/vpipe-sensenova-u1.5.so --list-models | grep -i sensenova
```

The repo is 50.2 GB and **all of it is needed** — there is no separate encoder
or VAE to skip, and the tokenizer files (`vocab.json`, `merges.txt`,
`added_tokens.json`) sit beside the shards.

## Graphs

`docs/pipelines/` has four:

- `sensenova-u1.5-text-to-image.vpipeline` — 1024², 20 steps.
- `sensenova-u1.5-preview.vpipeline` — 512², 8 steps, for iterating on a prompt.
- `sensenova-u1.5-image-edit.vpipeline` — one reference image, with a
  `compare-image` so the before/after sits side by side.
- `sensenova-u1.5-multi-reference-edit.vpipeline` — two references, with
  `<image>` markers in the prompt placing them.

Point `hf_dir` at the checkpoint directory, or wire a `model-select` source to
the generate stage's `model` iport (iport 1) and leave `hf_dir` unset.

## Editing

Wiring anything to `reference0` (iport 3) switches the run from text-to-image to
editing. `reference1` (iport 4) takes a second image.

The prompt may place them with `<image>` markers:

```
"<image>\n<image>\nplace the subject from the first into the scene of the second"
```

Without markers the model layer prepends them the way the reference
implementation does — and *names* them `Image-1:` / `Image-2:` when there is more
than one, which are tokens the model reads.

`img_cfg_scale` (on the model-config stage) is the second guidance axis: how hard
the result is pushed away from "these images, no instruction". Leaving it at 1
means only two prefixes are built; raising it costs a third prefill and a third
forward per step.

Reference images may be any size. The stage applies the model's own
`smart_resize` (a multiple of 32, within [512², min(2048², 4096²/n)] pixels),
PIL-compatible bicubic, and ImageNet normalisation — which is **not** the
normalisation the generated image uses, and is one of the two places this model
uses two conventions in one pipeline.

## Sizing

| resolution | image tokens | note |
|---|---|---|
| 256² | 64 | **generates badly** -- the model's documented minimum; see ARCHITECTURE.md §11 |
| 384² | 144 | the smallest size that works well |
| 512² | 256 | ~2.1 s/step on an M4 Pro |
| 1024² | 1024 | ~17 s/step |
| 2048² | 4096 | at the model's `max_image_seq_len` |

One LLM token covers a **32×32 pixel block**, so both dimensions must be
multiples of 32; anything else is rounded up. Cost grows with the token count
*and* quadratically inside attention, which is why 1024² is more than 4× the
cost of 512².

## Quantization

`model-quantize` handles this checkpoint through its general path -- it
auto-detects `arch='neo_chat'` and the layer prefix -- so no
`QuantizableFamily` is registered. That registry exists for MULTI-component
repacks with role subdirectories; this model is one flat directory.

```sh
vpipe --plugin build/vpipe-sensenova-u1.5.so \
      --launch docs/pipelines/prepare-sensenova-u1.5-8bit.vpipeline
```

`target` is a tensor-name PREFIX: `language_model.model.layers.` selects the
backbone and nothing else. On the released checkpoint that is **588 tensors
quantized, 528 passthrough** -- exactly the 42 layers x 14 matrices (7 per
expert x 2 experts), with every norm, both patch embedders, the pixel head, the
two scalar embedders and `embed_tokens` left dense. Quantizing the embedding or
the pixel head would be measurable in the image for almost no saving.

MEASURED on an M4 Pro, 384x384 / 8 steps, same prompt and seed as bf16:

| pack | on disk | resident | load | mean\|diff\| vs bf16 | correlation |
|---|---:|---:|---:|---:|---:|
| bf16 | 47 GB | 31.6 GB | 55 s | -- | -- |
| **w8g64** | 19 GB | **17.4 GB** | 6 s | **4.6** / 255 | **0.987** |
| w8g32 | 20 GB | 18.4 GB | 8 s | 5.0 | 0.985 |
| w4g64 | 11 GB | 9.9 GB | -- | 61.5 | 0.469 |
| w4g32 | 12 GB | 10.8 GB | -- | 78.9 | 0.545 |

At 512x512 / 20 steps w8g64 tightens to **1.8 / 255 and 0.99941** -- visually
indistinguishable. The metric loosens at fewer steps because a small per-step
perturbation has fewer steps to be corrected in.

**Use 8-bit.** 4-bit produces structured garbage on this model -- bars and
blocks, not a degraded picture -- at BOTH group sizes and through BOTH kernel
tiles, so it is neither a group-size nor a tile-selection problem, and the
dispatch is the same code that makes 8-bit work. Whether that is this model's
own sensitivity or a defect in the shared 4-bit path is NOT settled here. The
generate stage warns when it loads a 4-bit pack rather than silently rendering
noise.

Quantizing also makes loading **9x faster** (55 s -> 6 s), because most of the
bf16 load is converting the generation expert down from F32 and a quantized
pack has almost none of that left.

## Memory

The weights are ~31.6 GB resident in bf16 (**17.4 GB** at w8g64 -- see
Quantization above), plus a KV cache of roughly
`42 layers × 2 × 8 kv-heads × 128 × (prefix + image tokens) × 2 bytes` per
branch — about 350 MB at 1024² with guidance on — plus the backbone's shared
activation arena. All three are declared to the resource plan before anything
loads.

### Streaming

**Nothing needs to be configured.** The stage asks
`model_memory::plan_streaming` at load and streams the 42 layers when the
checkpoint will not sit comfortably beside everything else. On a 64 GB box the
bf16 pack streams and the w8 pack does not; on a 16 GB box both do. Set
`VPIPE_RAM_LIMIT_MB` to see what a smaller box would decide.

A streaming run keeps the trunk resident, pins a leading prefix of layers, and
reads the rest per pass — then keeps whatever the box turns out to hold. On a
roomy machine that converges on the whole stack within one image:

```
STREAMING the layers -- 47895 MB of weights would have to sit beside 658 MB
of activations and KV on a 65536 MB box. Pinning 6 of 42 layers
loaded ... 5.69 GB resident (6 of 42 layers held, 736 MB each)
streamed 36 layer reads (25.9 GB) this beat; 42 of 42 layers now resident
```

The cost is one extra read of the checkpoint, paid once. MEASURED at 256² on
the w8 pack: +5.5 s at 4 steps (3.15×) and +5.7 s at 20 steps (1.65×) — the
same absolute figure, amortised. Streamed output is **bit-identical** to
preloaded; `u15-stream-test` holds it to that, on both the bf16 and the w8
pack.

**Put the checkpoint on local SSD.** The same test on a bf16 pack sitting on an
external Thunderbolt drive (0.84 GB/s against the internal drive's 4.9 GB/s)
paid +55.8 s. Quantizing helps twice over here, because streaming reads what is
on DISK and a bf16 pack is 46.8 GB there against 31.6 GB resident.

On a slow drive the device is the whole cost; on a fast one the read primitive
matters too. The streaming path reads each layer with `pread(2)` into a
reusable slot rather than copying out of the shard's mmap -- MEASURED at
2.83 -> 3.67 GB/s (**1.30x**) on the internal SSD with the w8 pack, and 1.05x
on the external drive, where the device is the bottleneck and the primitive
cannot matter. `u15-io-bench` measures the primitives on their own; see
`ARCHITECTURE.md` §13.5.

So a **32 GB box now runs this model**, in bf16, which it could not before —
and a 16 GB box runs the w8 pack.

### Between images

`unload_when_idle` takes `keep` (default), `park`, `destroy` or `auto`.

* `park` hands the weights to the kernel as purgeable: they survive unless
  something else needs the RAM, and the next image takes them back without
  touching the disk. Never worse than `destroy`.
* `destroy` frees them; the next image pays a full reload. Right for a
  one-shot graph.
* `auto` resolves after the first image — where every peer has loaded and the
  numbers are real bytes — to `park` when the box is tight or a peer is
  already streaming, and `keep` otherwise.

Going idle **unwires** first -- wired and parked are opposites, and only
`unwire_from_pool()` decrements the pool's counter, so buffers freed while
wired leak their bytes for the rest of the run -- and then **releases** the
layers residency grew into. Those are the model's own buffers, read uncached,
so nothing else can reach them.

What happens next depends on the mode, because the host will not park a
checkpoint anything is still borrowing: a borrower holds aliases of the very
buffers a park makes purgeable, and the manager cannot tell an idle live model
from one mid-forward.

* **Streaming** keeps its binding. The release already gave back the larger
  term -- 26497 MB of 32322 held in the worked example, 82% -- against the
  ~3583 MB of trunk a park could add, and collecting that would mean letting
  go and re-binding on the next beat, F32 -> bf16 conversion included.
* **Preloaded** lets go, because nothing was promoted and the weight set is
  the only place bytes can come back from. Here `park` and `destroy` differ in
  what the *manager* does with the bytes, not in what the stage does with its
  models: both let go, and park keeps them purgeable and reactivated on the
  next read.

How much a preloaded park returns depends on the pack: a w8 pack parks 94% of
itself, a bf16 one 55% -- its generation expert is a dtype conversion, and
those are not parkable. The debug log says how much of each, every time:

```
idle -- unwired 3583 MB, released 26497 MB of streamed layers, kept the
trunk bound (a borrowed checkpoint cannot be parked), of 32322 MB held
```

### Wiring

Resident weights are mlocked through the host's shared wired pool, so the
compressor cannot take them mid-run (`--wired-pool-mb`, or
`VPIPE_WIRE_RESIDENT=0` to turn it off). This applies to preloaded models too,
not only streaming ones: a preloaded w8 run wires 16.8 GB.

## Matrix cores (M5 and newer)

The backbone runs three kinds of heavy math, and on a GPU with hardware
matrix units all three have a faster route than the simdgroup tiles the
plugin started on. It picks the route from
`MetalCompute::supports_matrix_cores()`, so a pre-M5 box is unaffected and
nothing here needs configuring.

What changes, and what it is worth. MEASURED on an M5, interleaved arms,
min of three rounds, at this model's own shapes (hidden 4096, 32 query
heads over 8 KV heads, head_dim 128, ffn 12288):

| path | shape | speed-up |
|---|---|---|
| dense projections -> `matmul2d` | qkv / o_proj / gate\|up, M=256 | **3.1-3.6x** |
| | ff_down (K=12288), M=256-1024 | **2.5-2.7x** |
| | qkv, M=1024 | **3.7x** (14.2 TFLOP/s against 3.8) |
| w8g64 projections -> dequant-once + `matmul2d` | M=256 | **2.4x** |
| | M=1024 | **3.5x** |
| flash attention -> `attn_steel_nax` | T=256 | **1.2-2.0x** |
| | T=1024 | **3.1x** |

All of it is the same arithmetic through different hardware, so the only
difference in the result is the order bf16 rounding happens in: rel-L2 is
5e-5 for the dense GEMMs, 5e-5 for the quantized ones and 1.3e-3 for the
attention, against a 6e-3 bar. `u15-m5-test` is the A/B and runs without
the checkpoint -- the shapes are the model's, the data is random, and
what it checks is the routing.

**The two row floors are different, and that is not an oversight.** The
dense path has effectively none: matmul2d beats the steel tile at every
row count down to M=1 (1.5-1.8x), because the 128-row tile being mostly
padding still costs less than the simdgroup path. The quantized path has
a real one at **M=96**: expanding the weight to bf16 costs one pass over
N*K whatever M is, so it is a 0.55x LOSS until the matmul it feeds is
big enough to amortize it -- 0.93x at M=64, 1.66x at M=96. Below the
floor the steel qmm, which unpacks inside its tile loop, is the right
kernel and is what runs.

Turn any of it off to measure it:

    VPIPE_U15_NO_MMA2=1        dense + quantized GEMMs back to steel
    VPIPE_U15_NO_NAX_ATTN=1    attention back to the steel flash kernel
    VPIPE_U15_MMA_MIN_M=N      dense row floor (default 1)
    VPIPE_U15_MMA_Q_MIN_M=N    quantized row floor (default 96)

### The two lossy tiers, and they compose

Everything above is the same arithmetic through different hardware. These
two are not: both are **opt-in and off by default**, because each trades
a little precision for speed, and which images that is acceptable on is a
judgement about the model rather than about the kernel.

| key | what it changes |
|---|---|
| `i8_gemm` | how a **weight is multiplied**: the backbone's big projections in dynamic int8 with per-512-group scales, instead of bf16. Self-gates on rows — the crossover is ~1k. |
| `sage_attn` | how a **score is computed**: the flash attention's QK^T product in int8, with one scale per attention block and the key side quantized as `K − mean(K)` over tokens. `P·V` stays in bf16. |

They act on different halves of a layer and neither reads the other's
state, so both can be on at once.

**Sage reaches the bidirectional pass and nothing else**, which is a fact
about this port rather than about the method: the causal and block-causal
branches take their own kernels, and the int8 twin is a function constant
on the flash kernel. Its smoothing is exact rather than approximate — a
per-channel shift moves every score in a row by the same amount, and
softmax does not see it — so nothing is added back afterwards.

`sage_dense_layers` (default 0) leaves a prefix of the backbone in bf16.
Zero rather than one: Sage computes every key and every query, so unlike
a method that *drops* keys there is no published reason to protect an
early layer's less redundant residual stream.

Both need matrix cores. Asked for on a box without them, each declines
with a message and the model runs bf16 — the graph still runs, and the
same graph runs on both boxes.

    VPIPE_SAGE_ATTN=0|1        override sage_attn either way
    VPIPE_I8_GEMM=0|1          override i8_gemm either way

## Knobs

On the generate stage: `width`, `height`, `steps`, `seed`, `hf_dir`,
`unload_when_idle`, and the two lossy accelerated tiers `i8_gemm` and
`sage_attn` (+ `sage_dense_layers`) — see
[Matrix cores](#matrix-cores-m5-and-newer).

On `sensenova-u1.5-model-config`: `cfg_scale` (4.0 in the reference's
examples; 1 disables guidance and halves the cost), `timestep_shift` (3.0),
`cfg_norm` (`none` / `global` / `channel` / `cfg_zero_star`), `cfg_interval_lo`
+ `cfg_interval_hi`, `t_eps`, and `init_noise`.

They are split that way on purpose: a stage serving several families
accumulates the union of their knobs, and each key is then silently inert on
whichever family is not resident.

## Reproducibility

`seed` selects a sample from **this port's** RNG stream. It does not reproduce
the reference implementation's image at the same seed — torch's Philox
generator is not reimplemented here. To compare against the reference, dump its
initial noise as raw f32 `[H][W][3]` channel-last and pass it as `init_noise`.
