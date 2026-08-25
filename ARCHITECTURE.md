# SenseNova-U1.5-8B-MoT — architecture, as pinned against the reference

Everything here was derived from the released checkpoint
(`sensenova/SenseNova-U1.5-8B-MoT`, 50.2 GB, 13 shards, 1116 tensors) and the
reference implementation (`github.com/OpenSenseNova/SenseNova-U1`, Apache-2.0).
Each claim is marked with how it is known:

- **[CKPT]** read off the checkpoint's tensor names/shapes/dtypes.
- **[REF]** read off the reference source, file and symbol named.
- **[DERIVED]** follows from the two above by arithmetic; the derivation is shown.
- **[OPEN]** not yet settled — do not build on it without checking.

The model is **not a diffusers-style pipeline**. There is no VAE, no ViT tower,
no adaLN DiT, and no `transformer/` + `text_encoder/` + `vae/` directory split.
It is one flat checkpoint holding a Qwen3-shaped LLM with **two complete
parameter sets** and a **pixel-space flow-matching head**.

---

## 1. What the checkpoint contains

Six top-level groups [CKPT]:

| tensors | prefix | what |
|---:|---|---|
| 1095 | `language_model.model.*` | the 42-layer MoT backbone (both experts) |
| 1 | `language_model.lm_head.weight` | `[151936, 4096]`, untied |
| 4 | `vision_model.embeddings.*` | the **understanding** patch embedder |
| 4 | `fm_modules.vision_model_mot_gen.embeddings.*` | the **generation** patch embedder |
| 4 | `fm_modules.fm_head.*` | the pixel decoder (`conv1`, `conv2`) |
| 8 | `fm_modules.{timestep,noise_scale}_embedder.mlp.*` | two sinusoid→MLP embedders |

`config.json` [CKPT]: `model_type: "neo_chat"`, `architectures: ["NEOChatModel"]`,
`llm_config` = Qwen3 with `hidden_size 4096`, `num_hidden_layers 42`,
`num_attention_heads 32`, `num_key_value_heads 8`, `head_dim 128`,
`intermediate_size 12288`, `vocab_size 151936`, `rms_norm_eps 1e-6`,
`rope_theta 5000000.0`, and — the two keys that are not stock Qwen3 —
`rope_theta_hw 10000.0`, `max_position_embeddings_hw 10000`.

Top level carries the generation config: `patch_size 16`, `downsample_ratio 0.5`,
`use_pixel_head true`, `use_adaLN false`, `timestep_shift 1.0`,
`time_schedule "standard"`, `base_image_seq_len 64`, `max_image_seq_len 4096`,
`noise_scale_mode "resolution"`, `noise_scale 1.0`, `noise_scale_max_value 16.0`,
`noise_scale_base_image_seq_len 64`, `add_noise_scale_embedding true`,
`t_eps 0.05`, `template "neo1_0"`, `concat_time_token_num 0`.

`fm_head_dim 1536` / `fm_head_layers 2` / `fm_head_mlp_ratio 1` are **vestigial**:
they configure `SimpleMLPAdaLN`, which `use_pixel_head: true` bypasses entirely.
No `res_blocks` tensors exist in the checkpoint [CKPT]. Do not implement them.

### 1.1 Dtype split — this is why 8B params is a 50 GB file

The base (understanding) weights are **BF16**; every `_mot_gen` twin is **F32**
[CKPT]. So the generation expert costs twice its parameter count in bytes. The
two patch embedders and both scalar embedders are F32 as well.

The reference declares `torch_dtype: bfloat16` and casts on load [REF
`config.json`], so **bf16 is the intended compute precision for both experts** and
converting the F32 twins down at load is faithful, not a shortcut. Doing so takes
the resident set from ~50 GB to ~36 GB.

> This is the mirror image of the LTX-2.5 trap where F32 scale/shift tables had to
> stay F32. Here the F32 is storage, not signal. But it is the same *class* of
> mistake to guess: the dtype is per-tensor in the safetensors header, so read it,
> do not assume it from the neighbours.

---

## 2. Mixture-of-Transformers: two experts, joint attention

Every transformer parameter appears twice — once bare, once with a `_mot_gen`
suffix [CKPT]:

```
layers.{N}.input_layernorm            / .input_layernorm_mot_gen
layers.{N}.post_attention_layernorm   / .post_attention_layernorm_mot_gen
layers.{N}.mlp.{gate,up,down}_proj    / .mlp_mot_gen.{gate,up,down}_proj
layers.{N}.self_attn.{q,k,v,o}_proj   / .self_attn.{q,k,v,o}_proj_mot_gen
layers.{N}.self_attn.{q,k}_norm       / .self_attn.{q,k}_norm_mot_gen
layers.{N}.self_attn.{q,k}_norm_hw    / .self_attn.{q,k}_norm_hw_mot_gen
model.norm                            / model.norm_mot_gen
```

`embed_tokens` and `lm_head` are **not** duplicated — text embedding is shared,
and the gen expert never produces logits.

**Which expert runs is a property of the TOKEN, not the layer.** Understanding
tokens (text, reference images) use the bare set; generation tokens (the noisy
image being denoised) use `_mot_gen` [REF `modeling_qwen3.py:Qwen3DecoderLayer.
forward_und` / `.forward_gen`].

### 2.1 The port never needs a per-token weight gather

`Qwen3DecoderLayer.forward` and `Qwen3Attention.forward` dispatch on two
booleans, and the **mixed** case `raise NotImplementedError` in the reference
itself [REF `modeling_qwen3.py:768` and `:1000`]:

> "The mixed und/gen forward path is not yet validated (issue #207) … Split the
> sequence at token-type boundaries and use forward_und / forward_gen."

So every forward pass is homogeneous. Two code paths, each with one weight set,
selected per call. There is no gather, no mask-select, no interleaving.

**Attention is still joint**: the gen tokens attend to the und tokens through the
KV cache. The experts are separate in the *projections*, not in the *attention*.

---

## 3. The three-way split RoPE

This is the most unusual part of the model and the easiest to get wrong.

`head_dim` is 128, but `q_norm`/`k_norm`/`q_norm_hw`/`k_norm_hw` are all `[64]`
[CKPT]. The reference constructs all four as `Qwen3RMSNorm(head_dim // 2)` [REF
`modeling_qwen3.py:Qwen3Attention.__init__`]. The head is split by position type:

```
             q_proj -> [.., 32 heads, 128]
                         |
              chunk(2, dim=-1)
              /                \
        t half (64)          hw half (64)
        q_norm(64)           q_norm_hw(64)      <- norms apply to the HALVES
        RoPE_t                    |
        theta 5e6            chunk(2, dim=-1)
        head_dim 64          /            \
                        h qtr (32)     w qtr (32)
                        RoPE_hw         RoPE_hw
                        theta 1e4       theta 1e4
                        head_dim 32     head_dim 32
                         \              /
        concat([ t(64), h(32), w(32) ], dim=-1)  ->  128
```

Order of operations that must be preserved [REF `forward_und` / `forward_gen`]:

1. project, view as `[.., H, 128]`;
2. `chunk(2, -1)` into t-half and hw-half;
3. RMSNorm **the halves** — `q_norm` over the t-half's 64, `q_norm_hw` over the
   hw-half's 64. The hw norm spans all 64, **before** the h/w split;
4. split the normed hw-half into h(32) and w(32);
5. apply three RoPEs with three different position vectors;
6. concat `[t, h, w]`.

The two rotary tables [REF]: `rotary_emb` is built from a config clone with
`head_dim = 128//2 = 64` and the model's `rope_theta` (5e6);
`rotary_emb_hw` from a clone with `head_dim = 128//4 = 32`,
`rope_theta = rope_theta_hw` (1e4), `max_position_embeddings =
max_position_embeddings_hw` (10000).

Both use `apply_rotary_pos_emb` with `rotate_half` — the **NeoX/split-half**
convention (`x[..., :d/2]` vs `x[..., d/2:]`), *not* interleaved pairs.
Note this differs from the vision embedder in §5, which *is* interleaved.

### 3.1 The position triple

Each token carries `(t, h, w)` [REF `_build_t2i_text_inputs`,
`_build_t2i_image_indexes`]:

| token kind | t | h | w |
|---|---|---|---|
| text | `arange(text_len)` | 0 | 0 |
| generated image | `text_len` (**constant**) | `i // token_w` | `i % token_w` |

Every image token shares one `t`. They are ordered only by `(h, w)`. That is what
makes the image block permutation-symmetric in time and fully bidirectional.

For the CFG pair the two prefixes have **different lengths**, so the image
tokens' constant `t` differs between the conditional and unconditional passes
[REF `t2i_generate`: `indexes_image_condition` vs `indexes_image_uncondition`].
Reusing one index array for both is a silent quality bug.

---

## 4. Geometry: 32 pixels per token

`patch_size 16`, `downsample_ratio 0.5` ⇒ `merge_size = 1/0.5 = 2` [REF
`t2i_generate`]. So [DERIVED]:

```
grid_h  = H / 16          grid_w  = W / 16     (patch grid, embedder input)
token_h = H / 32          token_w = W / 32     (LLM tokens, after 2x2 merge)
L       = token_h * token_w                    (image token count)
```

`image_size` is **(width, height)** throughout the reference — `image_size[0]` is
W, `image_size[1]` is H [REF `t2i_generate`, `_t2i_predict_v`]. Transposing this
gives a plausible-looking image at the wrong aspect ratio.

Constraint [DERIVED]: H and W must both be multiples of 32.

---

## 5. The patch embedder (`NEOVisionEmbeddings`)

Two instances with identical structure and separate weights [CKPT]:
`vision_model.embeddings` (understanding) and
`fm_modules.vision_model_mot_gen.embeddings` (generation).

Forward [REF `modeling_neo_vit.py:NEOVisionEmbeddings.forward`]:

1. input `[N, 768]` → view `[N, 3, 16, 16]`;
2. `patch_embedding` = `Conv2d(3→1024, k=16, s=16)`. Kernel == input, so this is a
   **per-patch linear**: `[1024, 768] @ x + bias` [DERIVED];
3. **GELU** — applied here, `gelu(patch_embedding(x))`. Easy to miss;
4. 2D RoPE, computed in **float32** then cast back:
   - `rope_dim_part = hidden/2 = 512`, `theta = rope_theta_vision = 1e4`;
   - first 512 dims rotated by **`abs_x` (the column)**, last 512 by
     **`abs_y` (the row)**;
   - **interleaved** pairs — `x[0::2]`/`x[1::2]` [REF `apply_rotary_emb_1d`].
     This is a *different convention from the backbone's rotate_half*. Both live
     in the same forward pass. Mixing them up is silent.
5. `dense_embedding` = `Conv2d(1024→4096, k=2, s=2)` over the `[h, w]` grid → the
   2× merge. **No activation after it.**

`abs_x = idx % W`, `abs_y = idx // W`, row-major within an image [REF
`build_abs_positions_from_grid_hw`].

---

## 6. The pixel head (`fm_head`)

The checkpoint's `fm_modules.fm_head` has exactly `conv1 [1024,1024,3,3]` and
`conv2 [192,256,3,3]` [CKPT]. Of the many decoder classes in
`modeling_fm_modules.py`, the one whose shapes match is `ConvDecoder`
(`input_dim=4096, hidden_dim=1024`) [REF, DERIVED]:

```
x: [B, 4096, H/32, W/32]          (LLM hidden states, as a 2-D feature map)
  PixelShuffle(2)   -> [B, 1024, H/16, W/16]
  conv1 3x3 pad 1   -> [B, 1024, H/16, W/16]
  GELU
  PixelShuffle(2)   -> [B,  256, H/8,  W/8 ]
  conv2 3x3 pad 1   -> [B,  192, H/8,  W/8 ]
  PixelShuffle(8)   -> [B,    3, H,    W   ]
```

`192 = 3 × 8²` [DERIVED]. There is **no activation after conv2**, and the GELU
sits between conv1 and the second shuffle. `PixelShuffle(r)` maps input channel
`c*r² + i*r + j` to output channel `c` at offset `(i, j)` — **channel-slowest**.

The GELU is torch's `nn.GELU()` default, i.e. the **erf** form, here and in the
patch embedder [REF]. The plugin implements erf exactly. Note though —
MEASURED, against an earlier claim here that said otherwise — the tanh
approximation libvpipe already ships differs by at most **4.7e-4 absolute** over
[-6, 6], which is *below* bf16's own resolution near 1.0 (2⁻⁸ = 3.9e-3). So the
erf form is exact and free, not required; a port that used the tanh GELU would
not be measurably wrong.

The hidden states enter as a feature map via
`[B, token_h, token_w, C] → einsum "b h w c -> b c h w"` [REF `_t2i_predict_v`].

---

## 7. The generation loop

### 7.1 Prompting

Template `neo1_0` = ChatML [REF `conversation.py:390`]:
`<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`,
`sep = "<|im_end|>\n"`, default system message empty, `SeparatorStyle.MPT`.

- **conditional** prefix: system = `SYSTEM_MESSAGE_FOR_GEN` [REF `utils.py:9`],
  user = the prompt, then `append_text`.
- **unconditional** prefix: system = the template **default (empty)**, user = `""`.
  Note it does *not* reuse the gen system message [REF `t2i_generate`].
- `append_text` in non-think mode is `"<think>\n\n</think>\n\n" + "<img>"` — the
  empty think block is *always* emitted, then the image-start token.

Two prefixes ⇒ two independent prefills ⇒ two KV caches.

### 7.2 Schedule

`timesteps = linspace(0, 1, num_steps+1)`, then shifted [REF
`_apply_time_schedule`]:

```
sigma = 1 - t
sigma = shift*sigma / (1 + (shift-1)*sigma)
t     = 1 - sigma
```

`t` **ascends 0 → 1**; `t=0` is pure noise, `t=1` is clean. (Inverted relative to
the usual sigma convention — the same inversion Boogu-Image has.)

`_apply_time_schedule` assigns `self.time_schedule = "standard"` on entry, so the
`"dynamic"` branch and `_calculate_dynamic_mu` are **dead code** [REF]. The
config's `time_shift_type`/`base_shift`/`max_shift`/`base_image_seq_len`/
`max_image_seq_len` therefore do not affect sampling. Do not port them.

### 7.3 Initial noise

[REF `t2i_generate`] with `noise_scale_mode == "resolution"`:

```
scale       = sqrt( (grid_h*grid_w) / merge_size^2 / 64 ) = sqrt(L / 64)
noise_scale = min(scale * 1.0, 16.0)
image_prediction = noise_scale * randn(B, 3, H, W)
```

The noise is drawn in **pixel space**, not patch space, and scaled by resolution.

### 7.4 Per step

```
z            = patchify(image_prediction, 32)                # [B,L,3072] channel-FASTEST
image_input  = patchify(image_prediction, 16, channel_first=True)  # [B,grid,768] channel-SLOWEST
image_embeds = vision_model_mot_gen(image_input, grid_hw)    # [B,L,4096]
t_emb        = timestep_embedder(t)                          # broadcast to all L
if add_noise_scale_embedding:
    t_emb   += noise_scale_embedder(noise_scale / 16.0)
image_embeds = image_embeds + t_emb

h      = backbone_gen(image_embeds, indexes_image, past_kv=prefix, update_cache=False)
x_pred = fm_head(h[-L:] as feature map)                      # PIXELS, then re-patchified
v_pred = (x_pred - z) / (1 - t).clamp_min(t_eps)
... CFG ...
z      = z + (t_next - t) * v_pred
image_prediction = unpatchify(z, 32)
```

Three things to underline:

**The head predicts x₀, not velocity.** `v_pred = (x_pred - z)/(1-t)`. Treating
the head output as a velocity produces structured noise, not an error.

**The two patchifications use different packings.** `z` (patch 32) is
`nchpwq->nhwpqc`, **channel-fastest**; `image_input` (patch 16) passes
`channel_first=True` giving `nchpwq->nhwcpq`, **channel-slowest** [REF
`patchify`]. They are computed from the same image in the same step.

**The prefix KV is never extended.** `update_cache=False`: image tokens attend to
`[prefix ‖ current image tokens]` bidirectionally (`causal=False`) and are then
discarded [REF `forward_gen`]. Every step re-runs the same L tokens against a
frozen prefix. This is a shared-prefix attention pattern, and it is why a 50-step
run costs 50 × L tokens of attention, not a growing cache.

### 7.5 CFG

`v = v_uncond + cfg_scale * (v_cond - v_uncond)`, then optionally renormalised
[REF `t2i_generate`]:

- `none` — as above.
- `global` — scale by `min(‖v_cond‖ / ‖v‖, 1)` over dims (1,2).
- `channel` — same but per-token (`dim=-1`).
- `cfg_zero_star` — `alpha = <v_cond, v_uncond>/‖v_uncond‖²` computed in **f32**
  [REF `optimized_scale`], then
  `v = v_uncond*alpha + cfg_scale*(v_cond - v_uncond*alpha)`;
  **step 0 outputs exactly zero** (`v_pred = v_pred_condition*0.`).

CFG is skipped outside `cfg_interval` and when `cfg_scale <= 1`.

Example defaults [REF `examples/t2i/inference.py`]: `cfg_scale 4.0`,
`timestep_shift 3.0`, `num_steps 50`, `cfg_norm "none"`, `cfg_interval (0,1)`.
Editing adds `img_cfg_scale 1.0` [REF `examples/editing/inference.py`].

---

## 8. Where this attaches in vpipe

The stock image path is
`diffusion-conditioner → generate-image → vae-decode → save-image`. **None of the
first three fit**:

- `diffusion-conditioner` emits a conditioning *tensor*; U1.5's conditioning is a
  **KV cache** tied to the same weights that denoise. It cannot cross a stage
  boundary.
- `generate-image` emits a *latent* for a VAE. U1.5 emits **pixels**.
- `vae-decode` has nothing to do.

So this plugin ships **one generation stage** that owns prefill + denoise + head,
emitting `TensorBeatPayload` planar U8 RGB `[3,H,W]` tagged `rgb-frames` — the
exact contract `vae-decode` produces [in-tree `vae-decode-stage.cc:106`] — so the
stock `save-image` consumes it unchanged.

This is the registry-vs-stage rule applied: a registry is right when the host
already has a stage doing that job for other families, and here it does not.

**No host change was needed.** Plugin ABI 2 as it stands was sufficient —
`register_stage` (with a spec), `register_metal_library` and
`register_catalog_entries`. Unlike the LTX-2.5 port, this family adds no
registry: `register_vae_family` has nothing to register (there is no VAE) and
`register_video_family` is the wrong modality. Whether a future *second* image
family of this shape would justify an `ImageModelFamily` registry is an open
question; one family is not evidence for a seam.

Two contract details the SDK does not enforce, both of which cost a launch:

- `allocate_oports()` must be called in **every** stage constructor, including a
  `ModelConfigSourceStage` subclass — the base class does not do it, and its
  header says so. Without it the stage reports zero oports and
  `pipeline_from_spec` refuses every downstream edge, so nothing launches.
- `declare_resources()` sizes from the checkpoint on DISK (47.9 GB), but the
  resident set after the F32→bf16 conversion is 31.6 GB. The stage calls
  `revise_declaration()` once loaded; leaving the disk figure would over-reserve
  16 GB against every peer that sizes after it. It revises again after every
  beat, because a streaming model's resident set GROWS — see §13.3.

---

## 9. Porting notes — things that cost a debug cycle

**libvpipe's SDPA kernels are HEAD-major on BOTH sides, and `kv_stride` is in
TOKENS.** Reading the doc comment above `sdpa_full_f16` is not enough; the body
is the contract:

```
qh  = q + (h * n_q + qi) * D          // head-major in
kkv = k + kv * kv_stride * D          // stride is TOKENS, kernel scales by D
out[(h * n_q + qi) * D + idx]         // head-major OUT
```

Two consequences, both of which produce finite, plausible, entirely wrong
numbers rather than an error:

- passing `kv_stride` in *elements* (`cap * head_dim`) indexes past the end of
  head 0 and reads another head's keys;
- feeding the attention output straight into `o_proj` transposes it — the matrix
  is exactly the right size, so nothing complains.

Measured: with both bugs the full-width layer-0 forward scored rel-L2 **1.72**
against the CPU reference (i.e. uncorrelated); with both fixed, **4.8e-3**, which
is bf16 round-off over ~15 dependent ops.

The lesson generalises: the plugin's OWN kernels were verified first and were
all correct. What was wrong was the *calling convention* for kernels borrowed
from the host — so a port needs a test for its use of libvpipe's kernels, not
only for the ones it writes. `u15-kernels-test` now covers both.

**The working image's channel layout is the other one that costs a day.**
The reference keeps `image_prediction` PLANAR `[B,3,H,W]`; this port keeps it
INTERLEAVED `[H][W][3]`, because that is what the pixel head emits and what the
u8 conversion consumes. Both are defensible — but `patchify(..., channel_first=
True)` for the embedder input indexes a *planar* image, so feeding it the
interleaved one silently reads the first third of the buffer as the red plane.

What makes this one nasty is the failure signature. At **step 0 the image is iid
noise**, and a channel permutation of iid noise is statistically identical — so
the first step produces a rich, prompt-sensitive prediction and everything looks
healthy. From step 1 the image carries structure, the permutation destroys it,
and the model's prediction collapses to a near-constant field. MEASURED: x_pred
absmax 3.23 at step 0, then 0.7–1.0 with mean stuck at −0.63 for every step
after; final image sd **13.0** (flat grey) against **71.9** once fixed.

Two lessons worth keeping:

- A statistical smoke test calibrated at "not constant, not saturated" certifies
  a flat grey field. The bar has to be set from a MEASURED good run.
- A single-step render is a cheap and very sharp diagnostic here: with
  `steps=1`, `dt=1` and `denom=1`, so the output *is* `x_pred`. Looking at it
  showed a 32-pixel periodic tiling — the token grid — which pointed straight at
  a spatial/layout fault rather than at the sampler.

## 10. Image editing

A reference image enters the prefix as **understanding** tokens, and changes
three things about the run.

### 10.1 The prefix carries the image

Each `<image>` marker in the prompt is replaced by
`<img>` + `<IMG_CONTEXT>` × N + `</img>`, where N is the image's **post-merge**
token count `(grid_h/2) × (grid_w/2)` [REF `it2i_generate`]. The text is
tokenised normally and the `<IMG_CONTEXT>` rows are then **overwritten** with the
patch embedder's output — through the **understanding** twin
(`vision_model.embeddings`), not the generation one [REF `_build_it2i_inputs`,
`extract_feature(gen_model=False)`].

When the prompt has fewer markers than images the reference prepends the rest,
and does it *differently* for one image than for several — several get named
`Image-1:`, `Image-2:`, … which are tokens the model reads.

### 10.2 Positions stop being an arange

`get_thw_indexes` [REF] advances `t` on every token **except** an
`<IMG_CONTEXT>`, and once more just after an `<img>`. So all of one image's
context tokens share a single `t`, and:

- the reference's block-causal mask **no longer reduces to plain causal**, so the
  prefill needs a masked kernel (`u15_sdpa_block_causal`; the mask-free
  `sdpa_causal_f16` used for text-only prefixes attends only the earlier half of
  each image);
- the generated image tokens sit at **`max(t) + 1`**, which is the token *count*
  only when every token has a distinct `t`. MEASURED: a 538-token edit prefix has
  `max(t) = 258`, so using the count would place the generated tokens 279
  positions too far along.

### 10.3 Reference images are normalised differently

`load_image_native` [REF] applies a Qwen-VL `smart_resize` to a multiple of 32
within `[512², min(2048², 4096²/n)]` pixels, PIL's default **BICUBIC** resample,
and **ImageNet** mean/std `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`.

That last one is the trap: the *generated* image uses mean = std = 0.5 (§7.3),
and the two live in one pipeline. A reference normalised the generated image's
way is a picture the model reads differently. Patch packing is channel-first,
the same as the generation path's embedder input.

### 10.4 A second guidance axis

`img_cfg_scale` guides away from "these images, no instruction", which needs a
**third** prefix. The reference selects branches by [REF `it2i_generate`]:

```
needs_cfg       = not (cfg == 1 and img_cfg == 1)
needs_img_cond  = needs_cfg and (img_cfg == 1 or cfg != img_cfg)
needs_uncond    = needs_cfg and img_cfg != 1
```

and combines them four different ways — the two-branch forms are *not* the
three-branch one with a term dropped, because which branch is the baseline
changes:

| case | v |
|---|---|
| `img_cfg == 1` | `v_img + cfg·(v_cond − v_img)` |
| `cfg == img_cfg` | `v_unc + cfg·(v_cond − v_unc)` |
| otherwise | `v_unc + cfg·(v_cond − v_img) + img_cfg·(v_img − v_unc)` |

Note also that the edit path's guidance-interval guard is spelled
`(t > lo and t < hi) or lo == 0` — **strict** inequalities plus an escape —
where t2i uses inclusive bounds and no escape. Two spellings of one idea in one
file; reproducing the wrong one silently changes which steps are guided.

## 11. Kernel choice, and how it was measured

The port was first assembled from whichever kernel was simplest to call
correctly. That is the right order to do things in and the wrong place to stop.
`u15-perf-test` measures each op at the shapes a 1024² render uses.

### 11.1 The one that mattered

**Attention was on the scalar kernel.** `sdpa_full_f16` materialises one query
at a time across 32 lanes; `attn_steel_h_bd128_bf16` — the flash kernel every
image DiT in the tree already uses, and whose head_dim 128 is exactly this
model's — tiles both axes and keeps the running softmax in registers.

MEASURED at 1024 queries × 1285 keys, GQA 32/8:

| | ms | GFLOP/s |
|---|---:|---:|
| `sdpa_full_f16` | 121.9 | 177 |
| `attn_steel_h_bd128_bf16` | 5.0 | 4290 |

**24.3×** on the kernel; **3.2× end to end** at 1024² (17.5 → 5.48 s/step).

Two fields are specific to this model and wrong by default:

- `gqa_factor` is the **group size** (32/8 = 4), not the kv head count. The
  kernel computes `kv_head = query_head / gqa_factor`, so leaving it at 1 — which
  a port with no GQA does — gives query head *h* the keys of kv head *h* and
  reads past an 8-head cache. The control in `u15-kernels-test` scores 1.4e17.
- the K/V stride is the cache **capacity**, not the key length.

Only the **bidirectional** path was switched over. The two masked paths run
*once* per generation (the prefill), so at 1024² they are ~0.1 s against a
20-step render's ~110 s; block-causal has no steel instantiation at all.

`rms_norm_fast_f16` (simd_sum) over `rms_norm_f16` (threadgroup tree) is a
further **1.58×** — 219 vs 139 GB/s, 80% of this box's bandwidth.

### 11.2 Measured and NOT adopted

- **`dense_gemm_t_bm64bn64_f16` (BN=64) is a TIE** with BN=32 on every shape —
  0.1–1%, and the verdict flips between runs. An earlier non-alternating
  measurement showed BN64 winning by 27% on `q_proj`; that was thermal drift.
- **Fused gate+up+SwiGLU** (`dense_gemm_swiglu_bm64_f16`) is a consistent but
  small **1.04×** — ~1.2 ms of a 62 ms layer. It needs the two projections
  **row-interleaved** (`gate0, up0, gate1, up1, …`, *not* concatenated) at load,
  and reading that backwards is silent. Not worth 2% of the forward.
- **MMA GEMM**: closed on M4 by earlier work in this tree (1.01×). Not re-tried.

### 11.3 Two harness faults worth remembering

**Non-alternating A/B measures thermal state.** Timing A nine times then B nine
times compares two thermal states as much as two kernels; an e2e run on this box
has ~4% spread. `ab_ms()` interleaves the arms. This is what turned "BN64 wins
27%" into "BN64 is a tie".

**A single dispatch cannot measure a small kernel.** One trivial dispatch costs
**2.98 ms** here against an empty command buffer's 0.06 ms. `split_qk_norm` and
`split_rope` first measured 1.9 ms → "8.4 GB/s, 3% of bandwidth", which sent a
rewrite after a problem that did not exist. Batched 64-per-command-buffer — which
is also how the backbone runs them, ~15 dispatches per layer into shared
encoders — they measure **128 and 75 GB/s** and cost 0.4 ms per layer, not 4.8.

The rewrite (a simdgroup per row with `simd_sum`, and a flat 1-D rope dispatch)
was kept: it is correct, better structured, and verified unchanged at 2.99e-3.
But it was not the win the first measurement implied, and the op is 0.6% of a
layer either way.

### 11.4 Where the time goes now

Per layer at 1024², after the changes:

| | ms | note |
|---|---:|---|
| GEMM (5 projections) | ~62 | 5300–6600 GFLOP/s = 67–84% of the ~7.9 TFLOP/s roofline |
| attention (steel) | 5.0 | was 122 |
| RMSNorm ×2 | 0.15 | 80% of bandwidth |
| split norm + rope | 0.4 | |

The model is **GEMM-bound**, at a decent fraction of roofline. The remaining
lever on this box is quantization (fewer bytes per MAC), not a better dense
kernel — matrix cores (M5) would be the other.

## 12. Quantization

`model-quantize` reaches this checkpoint through its GENERAL path, not the
`QuantizableFamily` registry. That registry exists for multi-component repacks
with role subdirectories — its code path refuses a component whose `role` is
empty — and this model is one flat directory with a `config.json`. The general
path already auto-detects `arch='neo_chat'`, the layer prefix and the layer
count, so registering a family here would add a path the host already has.

`target` is a tensor-name PREFIX. `language_model.model.layers.` selects
**588 tensors quantized, 528 passthrough** — exactly 42 layers × 14 matrices
(7 per expert × 2), leaving every norm, both patch embedders, the pixel head,
the two scalar embedders and `embed_tokens` dense. The 1-D norms fall out of
the wholesale rule's 2-D test on their own.

Reading a pack is per TENSOR, not per checkpoint: bits comes from the codes'
column count and group from the scales', against the known K. A pack whose
shapes do not close is refused loudly, because the tensor IS there — it just
does not mean what a quantized tensor means, and running it anyway produces a
checkpoint that loads and generates the wrong thing.

### 12.1 8-bit works; 4-bit does not

MEASURED at 384², 8 steps, one prompt and seed through each pack:

| pack | tile | resident | mean\|diff\| vs bf16 | correlation |
|---|---|---:|---:|---:|
| w8g64 | wide | 17.4 GB | 4.6 / 255 | 0.987 |
| w8g32 | narrow | 18.4 GB | 5.0 | 0.985 |
| w4g64 | wide | 9.9 GB | 61.5 | 0.469 |
| w4g32 | narrow | 10.8 GB | 78.9 | 0.545 |

At 512²/20 steps w8g64 tightens to **1.8 / 255 and 0.99941**. The metric
loosens at fewer steps because a per-step perturbation has fewer steps to be
corrected in — a bar taken from one configuration fails a healthy pack in
another, which it did when these were first written.

The 4-bit output is **structured garbage** — bars and blocks, not a degraded
picture. What that rules out: it is not the group size (both fail), not the
kernel tile (both fail, and 8-bit succeeds through both), and not the dispatch
(the same code selects all four, differing only in the function name). What it
does NOT establish is whether this is the model's own sensitivity to 4-bit on
all 588 matrices without AWQ, or a defect in the shared 4-bit path. Untested:
AWQ, and a mixed pack keeping the generation expert at 8-bit.

The generate stage WARNS on a 4-bit pack rather than silently rendering noise.

### 12.2 The other win is load time

Quantizing takes loading from **55 s to 6 s**. Most of the bf16 load is
converting the generation expert down from F32 (14.1 GB of it); a quantized
pack has almost none of that left — 0.98 GB.

## 13. Memory: streaming, residency, and the declarations

The bf16 checkpoint is **47.9 GB on disk and 31.6 GB resident**. Held whole it
does not fit a 32 GB box at all and leaves a 64 GB one with no room for
anything else, so the layers stream. Everything here is the host's own
machinery — `plan_streaming`, `BlockResidency`, `WiredPool`, the resource-plan
claims — with the model-specific parts noted where they differ.

### 13.1 What streams

The 42 MoT layers are the repeating unit and **97% of the bytes**. The trunk
(`embed_tokens` at 1.2 GB, both patch embedders, the pixel head, the two
scalar embedders, the two final norms) stays resident; each layer is read from
the checkpoint as the stack runs and either dropped or kept.

The decision is taken in the STAGE, before the loader runs, via
`model_memory::plan_streaming(session, dir, "", kStreamHeadroom)`. The wider
headroom is because this one cannot be walked back: streaming is an argument
to `U15Weights::load`, so changing the answer afterwards means rebuilding a
32 GB model. The rule is the host's — preload only when the checkpoint is at
most a third of RAM with headroom on top — which puts the bf16 pack on the
streaming path even on a 64 GB box, and the w8 pack on the preload path there.

`declare_resources()` reports the floor as well as the size, through
`weight_claim_streamable`, so a peer sizing after this stage sees that the
graph *can* run small:

```
resource-plan: 19053 MB preloaded / 3411 MB if every streamable component streams
```

**One thing this model does that the in-tree DiTs do not:** half the
checkpoint is F32 on disk and bf16 in the forward, so a streamed layer cannot
use the raw-refill fast path (`streamed-refill.h`) for the generation expert —
the destination is half the source and there is nowhere to put the bytes. Its
tensors are read uncached and converted on every pass. That is why a streamed
bf16 pass reads 47.9 GB where the resident set would be 31.6 GB, and it is
another reason to quantize first: a w8 pack is U32 codes and F16 scales
throughout, both of which a raw refill can place.

### 13.2 The loop inversion, and why streaming needed it

The denoise ran `for step { for branch { for layer } }`. Streaming that reads
the whole checkpoint **once per branch**, which for an edit is three times per
step.

`U15Backbone::forward_many` turns it inside out: `for step { for layer { for
branch } }`. The branches differ only in their prefix and their positions and
never read each other, so this is exactly a reordering — the same encoders in
the same order on one queue, the same number of command buffers. Each branch
gets its own hidden-state buffer (a few MB) and its own rope tables and steel
plan, both built once per pass rather than per layer.

VERIFIED by rendering the same prompt and seed through the code before and
after the change: **bit-identical**. That is the bar the streaming test holds
the streamed path to as well — not "close", not "correlated". A tolerance
would hide precisely the failure worth catching (a layer skipped, applied
twice, or read from the wrong index).

### 13.3 Growing the resident set back

Streaming keeps ~one layer live, which on a box with spare RAM is throughput
thrown away. `BlockResidency` keeps layers after they are used, as free memory
allows, and the important property is that it is **not a cache**: the access
pattern is a cyclic scan, for which LRU has a zero percent hit rate, so it
grows a fixed subset and then leaves it alone. `end_layer()` is where a
streamed layer is promoted, after the fence that retires its GPU work.

MEASURED on the M4 Pro 64 GB, the dense checkpoint at 256²:

```
STREAMING the layers -- 47895 MB of weights would have to sit beside 658 MB
of activations and KV on a 65536 MB box. Pinning 6 of 42 layers
loaded ... 5.69 GB resident (6 of 42 layers held, 736 MB each)
streamed 36 layer reads (25.9 GB) this beat; 42 of 42 layers now resident
```

So on a roomy box the streamed path converges on what preload would have held,
within one beat — which is the host's stated intent for the policy and is why
preload is kept only where it is obviously safe.

The cost is one read of the stack, paid once. MEASURED at 256², preloaded
against streamed-from-2-pinned-layers:

| pack | steps | preloaded | streamed | ratio | absolute |
|---|---:|---:|---:|---:|---:|
| w8 | 4 | 2.6 s | 8.1 s | 3.15× | +5.5 s |
| w8 | 20 | 8.8 s | 14.5 s | 1.65× | +5.7 s |
| bf16 | 4 | 2.6 s | 58.4 s | 22.5× | +55.8 s |

The overhead is the same in both w8 rows — it is the single pass that reads the
stack, and residency holds everything afterwards — so it amortises with the
step count and does not scale with it.

**The bf16 row is mostly a DISK measurement, not a dtype one**, and the
difference is worth being exact about because the obvious reading is wrong. The
two checkpoints were on different devices:

| | bytes on disk | device | `dd` rate | predicted | measured |
|---|---:|---|---:|---:|---:|
| w8 | 18.6 GB | internal SSD | 4.9 GB/s | 3.8 s | +5.5 s |
| bf16 | 46.8 GB | external Thunderbolt | 0.84 GB/s | 55.7 s | +55.8 s |

On the **external** drive the device is the whole story, to within the noise of
the measurement. The F32→bf16 conversion of the generation expert, which was
the natural suspect, disappears inside a read that is already 56 s — so
parallelising that loop would buy nothing, and the useful advice is the
ordinary one, for a different reason: keep the checkpoint on local SSD.

On the **internal** drive it is NOT the whole story, and an earlier draft of
this section said it was. 3.8 s predicted against 5.5 s measured is a 45% gap,
and it is not per-tensor overhead — it is the read PRIMITIVE. See §13.5, which
measures it: the mmap+memcpy this streaming path uses runs at 3.2 GB/s on that
drive where a pread runs at 4.5. Predicting from `dd` assumes the code reads the
way `dd` does, and it does not.

The other half of the same fact: the bf16 checkpoint is 46.8 GB on disk against
31.6 GB resident, because half of it is F32. Streaming reads what is ON DISK,
so a bf16 pack pays a 48% surcharge on every byte it streams that a quantized
one does not.

**A streamed model's declaration is re-revised every beat.** A figure frozen at
load says 5.69 GB for a model that is holding 32 GB by the end of the first
pass, and that reads to every peer as 26 GB of room which is not there.

### 13.4 Wiring, and why the page walk survives it

Resident layers are wired through the host's shared `WiredPool` (mlock +
`NonVolatile`), in the order MiniMax-H3 and Krea-2 establish and for the same
reason: **the activation arena first, then the trunk, then the layers.** The
arena is what a pass cannot proceed without and the trunk is read on every
layer of every pass; a resident layer is an optimisation the model can always
shed and stream instead. Protecting the optional half first is how a run ends
up with 30 GB of wired layers beside an activation buffer the compressor is
free to take.

Three things follow that are easy to get wrong, and two of them were:

**The arena must be unwired before it is replaced.** `ensure_scratch_` swaps
all eleven buffers wholesale when the token count grows, and only
`unwire_from_pool()` decrements the pool's counter — so a buffer dropped while
wired leaks its bytes for the rest of the run, and the leak compounds every
time the arena grows.

**Wiring is not a streaming feature.** A preloaded model has no layers to admit,
but `for_each_weight()` is then the whole checkpoint, and that is the case the
pool was measured on (the host's `wired-pool.h`: 1.21× on a preloaded 35 GB
DiT, with compression *falling* across the run). Gating the pool on `streaming`
forfeited that on the w8 path, which is the common one on a 64 GB box.
MEASURED after fixing it: **16832 MB wired** on a preloaded w8 run that wired
nothing before.

**Wired and parked are opposites.** `mark_inactive()` refuses a wired buffer
outright, so an idle model that is still in the pool parks *nothing*. The idle
path therefore unwires first and parks second; the next pass re-wires, since
`wire_trunk_` runs every pass and is a no-op per buffer already in the state
asked for.

#### Owned, never mapped

Every weight read here is `Residency::Copied` — six call sites, none falling
through to the accessor's `Mapped` default, and `load_mapped()` is never
called. That follows from the same rule: a mapped tensor is a subview of a
whole-shard mmap, so it can be **neither wired** (mlock on file-backed pages is
refused well short of the pool's ceiling — measured at ~4 GB on MiniMax-H3) nor
**parked** (`mark_inactive` refuses a handle that does not own its allocation),
and both are the whole point of keeping a resident set.

H3's rule is conditional — `weights_may_be_mapped(stream_blocks,
wire_resident)` — so "H3 always owns" is really "H3 streams and wires, and that
forbids mapping". The same is true here. But a third reason applies to this
checkpoint that holds even when the first two do not, and it is why there is no
residency switch in the loader at all: **mapping is per shard, not per tensor.**
Half this checkpoint is F32 that becomes bf16 at bind, interleaved in the same
shards as the BF16 half, so a mapped read would hold ~46.8 GB of shard beside
the ~15.8 GB of converted generation-expert buffers it still had to allocate —
~62.6 GB against 31.6 GB for Copied. That is arithmetic over the checkpoint,
not a measurement; the measurement vpipe already has is the same shape
(`docs/MODEL-MEMORY.md`: the LM reads went 3.62 → 5.00 GB, +38%, on switching
to Mapped).

The one configuration where mapping might genuinely have paid is a **quantized**
pack with the wired pool off, whose codes are U32 pass-through and therefore
conversion-light. Not taken: the pool is on by default, so the question is moot
in every configuration anyone runs, and a second residency path would exist only
to be wrong in.

#### Then is the mincore walk still needed?

For a wired buffer, no — and the code says so per buffer:

```cpp
if (b.empty() || b.is_wired()) { return; }   // mlock already guarantees it
```

A wired page cannot have left RAM, so asking is spending the walk to be told
what mlock guarantees. The skip is per BUFFER rather than per layer, because
`wire_layer_` stops at the first refusal and leaves the rest of that layer
unwired — the remainder is exactly what still needs measuring. With everything
wired `examined` stays 0, and `begin_pass` tests `examined > 0` before reading
a shortfall, so that is correctly interpreted as *no evidence* rather than as
*everything is gone*. The walk costs ~57 ms per 4.3 GB, so a fully wired
31.6 GB stack would otherwise pay ~420 ms per look for a guaranteed answer.

**But "everything is wired" is not a state this model reaches**, which is why
the walk stays rather than being deleted. Two things are never in the pool:

* The pool can be off — no `wired_pool_mb`, or `VPIPE_WIRE_RESIDENT=0` — and
  then nothing at all is wired.
* `WeightSet::for_each_weight()` yields only owned tensors whose source name is
  still known, so **`derived()` entries are outside it**. On a bf16 pack that is
  the entire generation expert — half the checkpoint — because it ships F32 and
  is converted at bind. So on the pack that most needs streaming, roughly half
  of every pinned layer is unwirable by construction.

The walk is also gated twice, both cheaply, and both gates are load-bearing:

| gate | why |
|---|---|
| `_resid.count() > 0` | `evict_tail_layer_` only sheds PROMOTED layers, never the pinned prefix. With nothing promoted there is no response available, so the measurement cannot be acted on. |
| `self_compression_grew()` | when none of *our* pages are being compressed there is nothing for the walk to find, and a healthy run would otherwise pay ~57 ms per 4.3 GB every pass to be told nothing. |

The first of those was `count() > 0 || _pinned > 0` here before matching H3,
which bought a walk this model has no way to answer.

### 13.5 mmap against pread, measured on both drives

`WeightSet::read` / `tensor` / `stream_tensor` all end at
`MetalLlamaWeights::load()`, which allocates a buffer and **memcpys into it
from the shard's mmap**. The alternative is `pread_into()` — `pread(2)`
straight into a buffer the caller already owns — which is what
`shared/streamed-refill.h` is built on. The claim worth testing is that mmap
loses structurally: the kernel tracks page residency at 4 KB granularity over
files of tens of gigabytes, and that bookkeeping costs more than a
kernel→user copy saves.

`tests/u15-io-bench.cc` measures it. Four arms, interleaved, over four
BYTE-BALANCED groups of the checkpoint's 32 largest tensors, with the
arm→group assignment rotated so neither thermal drift nor one arm warming the
cache for the next can be read as a result. The **same 19 GB w8 pack** on both
drives, so the device is the only thing that differs. Cold runs evict by
reading 46.8 GB of unrelated data first (`purge` needs root).

**COLD** — GB/s, range over four rotations:

| arm | internal SSD | external Thunderbolt |
|---|---:|---:|
| mmap + memcpy (what streaming does today) | 3.14 – 3.26 | 0.77 – 0.78 |
| pread, `F_NOCACHE` (the `pread_into` default) | 3.25 – 3.72 | 0.69 – 0.81 |
| **pread, cached** | **4.44 – 4.54** | 0.78 – 0.82 |
| mmap wrap + fault, NO copy (`load_mapped`) | 0.72 – 0.75 | 0.24 – 0.29 |

**WARM** (page cache hot), internal: mmap+memcpy 10.09, pread `F_NOCACHE`
14.39, pread cached 18.01, mmap wrap 14.21 GB/s.

Four things fall out, and the last two were not what I expected:

**1. pread wins, but only where the device is not the bottleneck.**
1.38–1.44× cold on the internal drive and 1.78× warm; **1.03–1.06× on the
external — a tie.** At 0.84 GB/s the drive is so far the slowest term that the
bookkeeping hides behind it entirely. So "mmap always loses" is right about the
direction and wrong about the magnitude: the win is a property of the drive,
and it is the fast drive that exposes it.

**2. The zero-copy wrap is a catastrophe, not a saving.** `load_mapped` +
touching the pages is **4.3× slower than mmap+memcpy** on the internal drive
(0.73 against 3.2) and 3× slower on the external. It copies nothing, so what
that time buys is page faults alone — the 4 KB bookkeeping in its purest form.
This is an independent, throughput-side confirmation of §13.4's footprint
argument for never using `Residency::Mapped`, and a much blunter one.

**3. `F_NOCACHE` costs about 25%** (3.4 against 4.5 cold), because it gives up
readahead. `pread_into` defaults it to true, which is right for a streaming
model — caching a checkpoint it re-reads every pass would compete for the RAM
the resident set wants — but it is a real price, not a free win, and a one-shot
preload would want it false.

**4. This corrects §13.3.** The w8 streaming read ran at 15.27 GB ÷ 5.5 s =
2.78 GB/s, which I attributed to per-tensor overhead against a 4.9 GB/s `dd`
rate. It is the mmap arm, measured here at 3.2 GB/s cold; the remainder is the
per-tensor and allocation overhead. Predicting a streaming cost from `dd`
assumes the code reads the way `dd` does.

#### What was done about it

**MiniMax-H3 is already on the pread route** — `refill_block_` for the block
stream and `adaln_into_` for the bake, with `Block _slot[2]`, an async prefetch
issued under the previous block's GPU work, per-tensor repair when a refill
will not serve, and `VPIPE_H3_NO_SLOTS` to turn it off. Its measurement on the
bake is the same shape as this one, larger on its box: 1.32–1.36 GB/s mapped
against 5.97–5.98 GB/s pread.

It is the ONLY caller of `refill_streamed_tensor` / `stream_into` in the host
tree. Krea-2, FLUX.2, Qwen-Image-Edit and Boogu all stream through
`stream_tensor` / `stream_derived`, i.e. mmap+memcpy with a block's worth of
buffers allocated and freed per block per forward.

What the slot costs is a decision at promotion time: MOVE out of it (leaving it
empty, so the next read rebuilds) or CLONE into the resident set (keeping it,
at a full copy per promotion). H3 clones; this moves. See below for why moving
is right here.

**U1.5 has the property that made the bake eligible**, and now uses it. All 42
MoT layers are identical in shape, so `U15Weights::_slot` is built once by the
ordinary allocate-and-convert route and then OVERWRITTEN IN PLACE for every
subsequent layer. The only allocation the streaming path makes after the first
layer is a rebuild when residency promotes the slot — promotion is a move out
of it — and that is bounded by the layer count over a run, against reads
bounded by layers × passes.

Three details decide whether it is correct:

* **`kUnservable` is per TENSOR, not a veto.** A bf16 pack's F32 generation
  expert has no raw-refill path, because the destination is half the source.
  Those tensors are `pread` into an F32 scratch and converted out of it, so
  even that half stays off the mmap arm. A quantized pack is U32 and F16
  throughout and refills everywhere.
* **`kFailed` means REBUILD, not retry.** The destination may be partly
  written, so the slot is dropped and rebuilt wholesale — which also re-derives
  the quantization metadata from the shapes, and that is the likeliest meaning
  of a size disagreement.
* **The two conversions were factored out** (`f32_to_bf16_`, `f16_to_bf16_`) so
  the build path and the refill path cannot drift. A second copy of
  round-to-nearest-even that differed by a half-ULP would fail the bit-identity
  test without pointing at itself.

MEASURED, driving the layer protocol directly over two passes with no residency
growth (`u15-stream-test`, "read path" section — growth stays off until a
reserve is declared, so nothing is promoted and the slots survive every layer).
Both arms produce **identical bytes**, and the refill arm rebuilds exactly one
layer (the first) against the rebuild arm's 84.

**The speed result here is INCONCLUSIVE, and an earlier draft of this section
reported the top of its spread as if it were the answer.** Five runs of the w8
pack on the internal SSD:

| | run 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| rebuild (mmap+memcpy) | 11.2 s | 11.3 | 11.3 | 11.3 | 11.4 |
| refill (pread) | 8.9 s | 10.8 | 10.6 | 10.2 | 11.7 |
| ratio | 1.30× | 1.04 | 1.06 | 1.11 | 0.97 |

The rebuild arm is stable to ±1%; the refill arm swings 30%. The likely reason
is the page cache: this pack is 18.6 GB on a 64 GB box, so the mmap path is
reading mostly warm pages and gets a stable fast answer, while `pread_into`
defaults to `F_NOCACHE` and goes to the device every time. That is the same
trade the `F_NOCACHE` row in §13.5 showed as a 25% cost — here it appears to
eat the whole win on a file the cache can hold.

So: the mechanism is correct and the byte-for-byte equality is solid, but on
THIS pack and THIS box the read is not reliably faster. What it does buy
unconditionally is the prefetch, which the synthetic section cannot show
because it has no GPU work to hide a read behind — the streamed RENDER arm
above issues 39 prefetches and lands all 39.

The host's 60-block image DiT, whose checkpoint is 38 GB and therefore does NOT
fit the page cache, shows the effect cleanly through the same shared
mechanism: 12.9 s streamed forward without the slots against 9.6-10.0 s with,
over three rounds each way, byte-for-byte identical. The lesson for the table
above is that a read-only A/B on a checkpoint smaller than the box measures the
page cache more than it measures the read.

**Residency and the fast path divide the work, and the division is the right
one.** On a roomy box every layer is admitted on pass one, promotion empties
the slot each time, and there are no streamed reads afterwards — 42 rebuilds,
then nothing, and the refill never runs. On a box that cannot admit, nothing is
ever moved out and every read from the second layer onward is a refill. So the
fast path is available exactly where streaming is not a formality, which is
also why promotion MOVES rather than clones: cloning would keep the slot alive
at the cost of a full copy per promotion (16 GB of memcpy on the w8 pack) to
accelerate reads that will never happen.

`Options::refill_streamed` turns it off, which is how the table above was
measured and is an escape hatch for a pack the fast path mishandles.

### 13.6 What the plan did not previously see

Two large allocations were declared by nothing: the backbone's shared
activation arena, and the KV cache — which is the only large allocation that
**grows during a run**, so a weights-only accounting reads as healthy right up
until a long prompt exhausts the box. Both are now `scratch_claims`, estimated
from the config during the planning phase and revised from the real figures
after the first beat.

At 256² with three branches that is 658 MB; the arena is ~129 MB of it and the
KV the rest, and the KV grows linearly with the prefix and with the image
token count.

### 13.7 Parking, and a host bug it uncovered

`unload_when_idle` now takes the host's full vocabulary — `keep`, `park`,
`destroy`, `auto` — where before it took three and silently treated `auto` as
`keep`. `auto` resolves after the first beat, where every peer has loaded and
real bytes are authoritative, and picks `park` when the box is tight or a peer
is already streaming.

`park` reported **0 MB**, which is the documented behaviour for models that
read uncached — but this loader caches everything, so the explanation did not
fit. The cause was in the host:
`GenerativeModelManager::park_weights()` looked a bare canonical directory up
in `_weight_sets`, which is keyed `"<canonical dir>|<variant>"`. That lookup
cannot match — not even in the ordinary empty-variant case, whose key still
carries the separator — so **the by-directory park entry point returned 0 for
every checkpoint, everywhere**, and the zero was indistinguishable from the
legitimate zero an uncached-reading model gives back. The memory-cap path was
never affected: `enforce_memory_cap()` walks `_weight_sets` itself and never
asks by name.

With the lookup widened to a prefix match, MEASURED:

| model | resident | parked | what limits it |
|---|---:|---:|---|
| w8 pack, preloaded | 17.4 GB | **16.8 GB (94%)** | nothing much — the codes are plain cached tensors for both experts |
| bf16, preloaded | 31.6 GB | **17.5 GB (55%)** | the generation expert is a dtype CONVERSION, and `derived()` entries are not parkable |
| bf16, streaming (6 pinned, grown to 42) | 31.6 GB after growth | **3.5 GB** | only the trunk — and see the borrow rule below, which is why this row is no longer collected |

The three rows are the three different answers, and each is a property of how
the loader READ the tensors rather than of the policy:

* A quantized pack parks almost completely because its codes go through
  `WeightSet::tensor()` for both experts — cached, `Copied`, reloadable.
* A bf16 pack parks about half, because the generation expert ships F32 and is
  bound through `derived()`. Derived entries cannot be parked: the registry's
  contract is that a reclaimed buffer can be re-read, and the transform that
  produced these is not retained.
* A streaming model parks only what the weight SET holds. The layers residency
  promoted are the model's own buffers, not cache entries, so parking cannot
  see them — giving those back is `BlockResidency::release`, which is what a
  peer needing room actually reaches for.

None of this is a reason to prefer `keep`: park is never worse (the pages
survive unless something else wants them, and come back without a reload).

**`park` also releases the grown layer set**, via
`U15Weights::release_at_idle()`. Parking cannot reach it — those layers were
read UNCACHED precisely so the weight set would not hold them — so an idle
streaming model would otherwise sit on whatever residency grew to, which on a
roomy box is the whole checkpoint. That release resets the growth ratchet,
which is not optional: `BlockResidency::release` ratchets the ceiling down to
what is left, so giving everything back would cap the next beat at one layer
forever. The ratchet exists to react to memory PRESSURE, and a deliberate
release is not pressure. And it unwires first (§13.4): only
`unwire_from_pool()` decrements the pool's counter, so a buffer freed while
wired leaks its bytes for the rest of the run.

MEASURED on the dense streaming run: **unwired 3583 MB, released 26497 MB of
streamed layers, of 32322 MB held.**

#### The borrow rule, and why the streaming row above is now the whole story

The host has since made `_weight_sets` a STRONG map: the manager owns every
checkpoint and a stage holds a BORROW. It will not park a set anything is
still borrowing, and the reasoning is not conservatism — a borrower holds
aliases of the very buffers a park makes purgeable and reads them in a forward
pass that never asks the set for anything. A park underneath it hands its
reader pages the kernel may discard, with nothing on the read path to notice.
The manager cannot tell an idle live model from one mid-forward, so it does
not try.

So `park_weights()` around a live model returns 0 — a THIRD zero, and the only
one that is a refusal rather than an absence. What this stage does about it
depends on the mode, and the two answers are different because the numbers
are:

* **Streaming keeps its binding.** The release already returned 26497 of the
  32322 MB held (82%), against ~3583 MB of trunk a park could add. Collecting
  that would mean letting go and re-binding on the next beat — trunk, layer
  scaffolding, and the F32 → bf16 conversion of the generation expert. Not
  worth 11%.
* **Preloaded lets go**, because nothing was promoted, the release returns 0,
  and the weight set is the only place bytes can come back from. There `park`
  and `destroy` differ in what the MANAGER does with the bytes rather than in
  what this stage does with its models: both let go, and park keeps them
  purgeable and reactivated on the next read, so the next beat re-binds
  without re-reading the disk.

The two preloaded rows of the table above are what that path still returns.
The streaming row is now the trunk this stage deliberately does not chase.

### 13.8 Not done

**The prefills are not inverted.** The denoise branches share one stack pass;
the prefills do not, because each branch's prefix is a different length and the
whole pass shape — scratch size, rope tables, attention rule — follows from it.
So a streamed edit pays 2–3 extra stack reads at the start. On a box where
residency can grow that is absorbed within the first beat; on one where it
cannot, it is a real cost.

**No prefetch overlap.** A streamed layer is read, used, and only then is the
next one read — the fence that lets a layer go is also what serialises the read
against the compute. Reading layer *i+1* while the GPU runs layer *i* would
hide most of the read behind work already happening, and the read touches no
GPU state, so issuing it between `commit()` and `fence.wait()` is enough.

Now much cheaper than it was: §13.5 built the reusable slot, and this needs a
SECOND one plus the bookkeeping for which slot holds which layer (promotion
empties one of them). Not done — it is a separate optimisation with its own
failure mode, and unlike the refill it has no measurement behind it yet.

## 14. Open questions

- **[OPEN]** `256²` generates badly — a tiny subject on a mostly-empty canvas,
  reproducibly, at 8 and at 30 steps — while 384², 512² and 1024² are all clean
  through the same code path. 256² is exactly the config's `min_pixels` (65536)
  and exactly `base_image_seq_len` (64 tokens), so this is *probably* the model's
  own lower boundary rather than a port defect: a resolution bug would not switch
  off between 12×12 and 16×16 tokens while being pixel-perfect at 32×32. NOT
  proven — that needs a reference run.
- **[OPEN]** Whether `lm_head` is needed at all outside think-mode. Non-think
  generation never samples a text token, so it may be droppable — worth ~1.2 GB.
- **[OPEN]** Think mode (`<think>…</think>` reasoning before generating), which
  needs the lm_head and a sampling loop.
