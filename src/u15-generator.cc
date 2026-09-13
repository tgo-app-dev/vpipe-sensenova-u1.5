#include "u15-generator.h"

#include "u15-ref.h"

#include "common/flex-data.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>

using vpipe::FlexData;
using vpipe::metal_compute::SharedBuffer;

namespace u15 {

namespace {

constexpr std::size_t kElt = 2;   // bf16

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
  return (std::uint16_t)((u + r) >> 16);
}

float
from_bf16_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

void
upload_(const std::vector<float>& v, const SharedBuffer& b)
{
  auto* p = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { p[i] = to_bf16_(v[i]); }
}

void
download_(const SharedBuffer& b, std::vector<float>* v)
{
  const auto* p = static_cast<const std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v->size(); ++i) { (*v)[i] = from_bf16_(p[i]); }
}

// The image tokens' position triple: ALL share t = prefix_len, and vary
// only in (h, w). The prefix length differs between the conditional and
// unconditional passes, so this is called once per branch -- reusing one
// for both is a silent quality bug.
void
image_positions_(int prefix_len, int th, int tw, std::vector<int>* pt,
                 std::vector<int>* ph, std::vector<int>* pw)
{
  const int n = th * tw;
  pt->assign((std::size_t)n, prefix_len);
  ph->resize((std::size_t)n);
  pw->resize((std::size_t)n);
  for (int i = 0; i < n; ++i) {
    (*ph)[(std::size_t)i] = i / tw;
    (*pw)[(std::size_t)i] = i % tw;
  }
}

}  // namespace

GenParams
GenParams::from_flex(const FlexData& fd, const GenParams& def)
{
  GenParams p = def;
  if (!fd.is_object()) { return p; }
  const auto o = fd.as_object();
  const auto gi = [&](const char* k, int& v) {
    if (o.contains(k)) { v = (int)o.at(k).as_int(v); }
  };
  const auto gd = [&](const char* k, double& v) {
    if (o.contains(k)) { v = o.at(k).as_real(v); }
  };
  gi("width", p.width);
  gi("height", p.height);
  gi("steps", p.steps);
  gd("cfg_scale", p.cfg_scale);
  gd("img_cfg_scale", p.img_cfg_scale);
  gd("timestep_shift", p.timestep_shift);
  gd("t_eps", p.t_eps);
  if (o.contains("seed")) {
    p.seed = (std::uint64_t)o.at("seed").as_int((std::int64_t)p.seed);
  }
  if (o.contains("cfg_norm")) {
    const std::string s(o.at("cfg_norm").as_string("none"));
    if (s == "global") { p.cfg_norm = CfgNorm::Global; }
    else if (s == "channel") { p.cfg_norm = CfgNorm::Channel; }
    else if (s == "cfg_zero_star") { p.cfg_norm = CfgNorm::CfgZeroStar; }
    else { p.cfg_norm = CfgNorm::None; }
  }
  if (o.contains("cfg_interval")) {
    const FlexData a = o.at("cfg_interval");
    if (a.is_array()) {
      const auto arr = a.as_array();
      if (arr.size() >= 2) {
        p.cfg_interval_lo = arr.at(0).as_real(p.cfg_interval_lo);
        p.cfg_interval_hi = arr.at(1).as_real(p.cfg_interval_hi);
      }
    }
  }
  if (o.contains("init_noise")) {
    p.init_noise_path = std::string(o.at("init_noise").as_string(""));
  }
  return p;
}

std::unique_ptr<Generator>
Generator::create(const Deps& d, const U15Config& cfg, std::string* err)
{
  const auto fail = [err](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<Generator>{};
  };
  if (d.ops == nullptr || d.weights == nullptr || d.backbone == nullptr ||
      d.image == nullptr || d.prompt == nullptr) {
    return fail("the generator is missing a dependency");
  }
  auto g = std::unique_ptr<Generator>(new Generator());
  g->_d = d;
  g->_cfg = cfg;
  return g;
}

std::size_t
Generator::kv_bytes(const GenParams& p) const
{
  const int th = align_dimension(p.height) / kPixelsPerToken;
  const int tw = align_dimension(p.width) / kPixelsPerToken;
  // A generous prefix allowance; the real one is known only after
  // tokenising, and this is used for a DECLARATION, which must not
  // under-report.
  const int cap = 1024 + th * tw;
  return _d.backbone->kv_bytes(cap) * 2;   // conditional + unconditional
}

bool
Generator::generate(const std::string& prompt,
                    const std::vector<RefImage>& refs, const GenParams& p,
                    std::vector<std::uint8_t>* out_u8,
                    const std::function<void(int, int)>& progress,
                    const std::function<bool()>& stop,
                    std::string* err)
{
  const auto fail = [err](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (out_u8 == nullptr) { return fail("no output"); }

  const int W = align_dimension(p.width);
  const int H = align_dimension(p.height);
  const int th = H / kPixelsPerToken;
  const int tw = W / kPixelsPerToken;
  const int L  = th * tw;
  const int gh = H / _cfg.gen.patch_size;
  const int gw = W / _cfg.gen.patch_size;
  const int hidden = _cfg.llm.hidden_size;

  const bool editing = !refs.empty();

  // Which branches this run needs, exactly as the reference decides it.
  // The two axes are not symmetric: `img_cfg_scale` guides away from
  // "these images, no instruction", which only exists when there ARE
  // images.
  const double cfg = p.cfg_scale;
  const double icfg = editing ? p.img_cfg_scale : 1.0;
  const bool needs_cfg = !(cfg == 1.0 && icfg == 1.0);
  const bool needs_img_cond =
      editing && needs_cfg && (icfg == 1.0 || cfg != icfg);
  const bool needs_uncond =
      needs_cfg && (!editing || icfg != 1.0);

  auto* mc = _d.ops->mc();
  auto stream = mc->make_command_stream();

  // ---- reference images --------------------------------------------
  //
  // Preprocessed here rather than by the graph: smart_resize's factor,
  // pixel bounds and the ImageNet normalisation are MODEL facts, and a
  // reference normalised like the generated image (mean = std = 0.5) is
  // a picture the model reads differently.
  std::vector<Prompt::RefImage> ref_grids;
  std::vector<SharedBuffer> ref_embeds;   // [tokens][hidden] bf16, per image
  if (editing) {
    // The reference caps total reference pixels across all images.
    const long max_px =
        std::min(2048L * 2048, (4096L * 4096) / (long)refs.size());
    for (const RefImage& r : refs) {
      if (r.width <= 0 || r.height <= 0 ||
          r.rgb.size() != (std::size_t)r.width * r.height * 3) {
        return fail("a reference image is empty or not [h][w][3] u8");
      }
      int rh = 0, rw = 0;
      if (!ref::smart_resize(r.height, r.width, kPixelsPerToken,
                             512L * 512, max_px, &rh, &rw)) {
        return fail("a reference image's aspect ratio exceeds 200:1");
      }
      std::vector<std::uint8_t> resized((std::size_t)rw * rh * 3);
      ref::resize_bicubic_u8(r.rgb.data(), r.width, r.height,
                             resized.data(), rw, rh, 3);

      const int gh = rh / _cfg.gen.patch_size;
      const int gw = rw / _cfg.gen.patch_size;
      std::vector<float> patches((std::size_t)gh * gw * 3 *
                                 _cfg.gen.patch_size * _cfg.gen.patch_size);
      ref::reference_patches(resized.data(), rw, rh, _cfg.gen.patch_size,
                             patches.data());

      SharedBuffer px = mc->make_shared_buffer(patches.size() * kElt);
      SharedBuffer emb = mc->make_shared_buffer(
          (std::size_t)(gh / 2) * (gw / 2) * hidden * kElt);
      if (px.empty() || emb.empty()) {
        return fail("out of memory preprocessing a reference image");
      }
      upload_(patches, px);
      // The UNDERSTANDING twin of the patch embedder -- gen=false. Using
      // the generation twin here would embed the reference with the
      // weights trained to read NOISE.
      if (!_d.image->embed(stream, px, gh, gw, /*gen=*/false, emb, err)) {
        return false;
      }
      ref_grids.push_back({gh, gw});
      ref_embeds.push_back(std::move(emb));
    }
    stream.commit().wait();
  }

  // ---- the branches -------------------------------------------------
  struct Branch {
    std::vector<int> ids;
    std::vector<int> t, h, w;      // prefix positions
    bool with_images = false;
    KvCache kv;
    SharedBuffer t_index;          // block-causal only
    std::vector<int> pt, ph, pw;   // the IMAGE tokens' positions
    std::vector<float> v;          // this branch's velocity
  };
  std::vector<Branch> branches;
  {
    Branch b;
    b.ids = _d.prompt->encode(
        editing ? _d.prompt->edit_conditional_text(prompt, ref_grids)
                : _d.prompt->conditional_text(prompt));
    b.with_images = editing;
    branches.push_back(std::move(b));
  }
  if (needs_img_cond) {
    Branch b;
    b.ids = _d.prompt->encode(_d.prompt->edit_image_only_text(ref_grids));
    b.with_images = true;
    branches.push_back(std::move(b));
  }
  if (needs_uncond) {
    Branch b;
    b.ids = _d.prompt->encode(_d.prompt->unconditional_text());
    b.with_images = false;      // no system message, no prompt, no image
    branches.push_back(std::move(b));
  }
  const int i_cond = 0;
  const int i_img = needs_img_cond ? 1 : -1;
  const int i_unc = needs_uncond ? (needs_img_cond ? 2 : 1) : -1;

  if (branches[0].ids.empty()) {
    return fail("the prompt tokenised to nothing");
  }

  // ---- pace the residency growth ------------------------------------
  //
  // Only a streaming model listens to any of this, and it is here
  // rather than in the stage because this is where both terms are
  // known: how many passes the run will make, and what has to stay free
  // through each of them.
  //
  // The reserve is the KV caches plus the activation arena. Both are
  // this run's, both are large, and neither is a weight -- so a policy
  // that only counted weights would admit layers into room the next
  // allocation needs. Each is reported by whoever allocates it, as it
  // happens.
  {
    std::size_t kv_total = 0;
    int longest = L;
    for (const Branch& b : branches) {
      kv_total += _d.backbone->kv_bytes((int)b.ids.size() + L);
      longest = std::max(longest, (int)b.ids.size());
    }
    _d.weights->set_residency_reserve(
        kv_total + _d.backbone->scratch_bytes(longest));
    // One pass per branch to prefill, then one per step for the whole
    // set of them -- the branch loop is inside the layer loop now.
    _d.weights->set_residency_schedule((int)branches.size() + p.steps);
  }

  std::size_t kv_done = 0;
  for (Branch& b : branches) {
    const int n = (int)b.ids.size();
    if (b.with_images) {
      _d.prompt->thw_indexes(b.ids, ref_grids, &b.t, &b.h, &b.w);
    } else {
      b.t.resize((std::size_t)n);
      for (int i = 0; i < n; ++i) { b.t[(std::size_t)i] = i; }
      b.h.assign((std::size_t)n, 0);
      b.w.assign((std::size_t)n, 0);
    }

    b.kv = _d.backbone->make_cache(n + L, err);
    if (!b.kv.valid()) { return false; }
    kv_done += _d.backbone->kv_bytes(n + L);
    _d.weights->note_kv_allocated(kv_done);

    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * hidden * kElt);
    if (x.empty()) { return fail("out of memory embedding the prefix"); }
    {
      const auto* tab = static_cast<const std::uint16_t*>(
          _d.weights->trunk().embed_tokens.contents());
      auto* dst = static_cast<std::uint16_t*>(x.contents());
      for (int i = 0; i < n; ++i) {
        const int id = b.ids[(std::size_t)i];
        if (id < 0 || id >= _cfg.llm.vocab_size) {
          return fail("token id out of range");
        }
        std::memcpy(dst + (std::size_t)i * hidden,
                    tab + (std::size_t)id * hidden,
                    (std::size_t)hidden * kElt);
      }
      // Splice the reference-image embeddings over the <IMG_CONTEXT>
      // rows. Those rows currently hold the embedding of the CONTEXT
      // TOKEN itself, which is a placeholder the model never sees.
      if (b.with_images) {
        std::size_t img = 0, seen = 0;
        for (int i = 0; i < n; ++i) {
          if (b.ids[(std::size_t)i] != _d.prompt->specials().img_ctx) {
            continue;
          }
          while (img < ref_grids.size() &&
                 seen >= (std::size_t)ref_grids[img].tokens()) {
            ++img;
            seen = 0;
          }
          if (img >= ref_grids.size()) {
            return fail("more <IMG_CONTEXT> tokens than reference-image "
                        "tokens -- the prefix and the images disagree");
          }
          const auto* src = static_cast<const std::uint16_t*>(
              ref_embeds[img].contents());
          std::memcpy(dst + (std::size_t)i * hidden,
                      src + seen * (std::size_t)hidden,
                      (std::size_t)hidden * kElt);
          ++seen;
        }
      }
    }

    Attn attn = Attn::Causal;
    if (b.with_images) {
      // A reference image's tokens share one t, so the block-causal rule
      // no longer reduces to causal and the pass needs the t vector.
      b.t_index = mc->make_shared_buffer((std::size_t)(n + L) * sizeof(int));
      if (b.t_index.empty()) { return fail("out of memory"); }
      auto* ti = static_cast<int*>(b.t_index.contents());
      for (int i = 0; i < n; ++i) { ti[i] = b.t[(std::size_t)i]; }
      for (int i = 0; i < L; ++i) { ti[n + i] = 0; }   // unused by prefill
      attn = Attn::BlockCausal;
    }
    if (!_d.backbone->forward(stream, x, n, Expert::Und, b.kv, 0, n, b.t,
                              b.h, b.w, attn, err, &b.t_index)) {
      return false;
    }

    // The generated image's t is max(prefix t) + 1 -- which is the token
    // COUNT only when every token has a distinct t. With an image in the
    // prefix it is not, and using the count would place the generated
    // tokens hundreds of positions too far along.
    int max_t = 0;
    for (int v : b.t) { max_t = std::max(max_t, v); }
    image_positions_(max_t + 1, th, tw, &b.pt, &b.ph, &b.pw);
    b.v.assign((std::size_t)H * W * 3, 0.0f);
  }

  // ---- initial noise -----------------------------------------------
  //
  // Pixel space, channel-LAST [H][W][3], scaled by resolution:
  // sqrt(L / base), capped. The reference draws in pixel space too.
  const double noise_scale = _cfg.gen.init_noise_scale(L);
  std::vector<float> img((std::size_t)H * W * 3);
  if (!p.init_noise_path.empty()) {
    // Injected noise, so an end-to-end run can be compared against the
    // reference: torch's Philox RNG cannot be reproduced here, so a
    // matching seed does NOT give a matching image and pretending
    // otherwise would make any comparison meaningless.
    std::ifstream f(p.init_noise_path, std::ios::binary);
    if (!f) { return fail("cannot read init_noise " + p.init_noise_path); }
    f.read(reinterpret_cast<char*>(img.data()),
           (std::streamsize)(img.size() * sizeof(float)));
    if ((std::size_t)f.gcount() != img.size() * sizeof(float)) {
      return fail("init_noise is the wrong size: want " +
                  std::to_string(img.size()) + " f32");
    }
  } else {
    std::mt19937_64 rng(p.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : img) { v = (float)(noise_scale * nd(rng)); }
  }

  // ---- the schedule -------------------------------------------------
  const std::vector<float> ts =
      ref::time_schedule(p.steps, p.timestep_shift);

  // ---- scratch ------------------------------------------------------
  SharedBuffer px_in =
      mc->make_shared_buffer((std::size_t)gh * gw * 3 *
                             _cfg.gen.patch_size * _cfg.gen.patch_size *
                             kElt);
  SharedBuffer emb =
      mc->make_shared_buffer((std::size_t)L * hidden * kElt);
  // ONE HIDDEN STATE PER BRANCH, because the branches now run through
  // the stack together rather than one after another (see
  // U15Backbone::forward_many). At [tokens][hidden] bf16 this is a few
  // megabytes each -- next to nothing beside the layer it saves
  // re-reading.
  std::vector<SharedBuffer> hstate(branches.size());
  for (SharedBuffer& h : hstate) {
    h = mc->make_shared_buffer((std::size_t)L * hidden * kElt);
  }
  SharedBuffer px_out =
      mc->make_shared_buffer((std::size_t)H * W * 3 * kElt);
  SharedBuffer u8 = mc->make_shared_buffer((std::size_t)H * W * 3);
  bool hs_ok = true;
  for (const SharedBuffer& h : hstate) { hs_ok &= !h.empty(); }
  if (px_in.empty() || emb.empty() || !hs_ok || px_out.empty() ||
      u8.empty()) {
    return fail("out of memory allocating the denoise scratch");
  }

  std::vector<float> patch((std::size_t)gh * gw * 3 *
                           _cfg.gen.patch_size * _cfg.gen.patch_size);
  std::vector<float> x_pred((std::size_t)H * W * 3);

  // THE PREDICATE REACHES THE LAYER LOOP, not just this one. A step here
  // is one read of the 42-layer checkpoint on a streamed model, so a
  // check only between steps leaves the Stop button doing nothing for
  // the length of one -- minutes at a large geometry. Cleared on every
  // exit: the backbone outlives this call, and a dangling predicate
  // capturing a stage's context would be read on the next run.
  struct StopGuard {
    U15Backbone* b;
    ~StopGuard() { b->set_stop({}); }
  } stop_guard{_d.backbone};
  _d.backbone->set_stop(stop);

  const double ns_ratio =
      noise_scale / std::max(1e-9, _cfg.gen.noise_scale_max_value);

  for (int step = 0; step < p.steps; ++step) {
    if (stop && stop()) {
      if (err != nullptr) { *err = kStopped; }
      return false;
    }
    if (progress) { progress(step, p.steps); }
    const double t = (double)ts[(std::size_t)step];
    const double t_next = (double)ts[(std::size_t)step + 1];

    // The embedder's input is patch 16, packed CHANNEL-FIRST. The
    // denoised image itself is what z is (patch 32, channel-last) --
    // both come from this same array in this same step, with different
    // packings.
    ImagePath::patchify_for_embed(img.data(), H, W, _cfg.gen.patch_size,
                                  patch.data());
    upload_(patch, px_in);

    if (!_d.image->embed(stream, px_in, gh, gw, /*gen=*/true, emb, err)) {
      return false;
    }
    // One timestep row, broadcast over every image token.
    const std::vector<float> trow = _d.image->timestep_row(t, ns_ratio);
    {
      SharedBuffer row =
          mc->make_shared_buffer((std::size_t)hidden * kElt);
      if (row.empty()) { return fail("out of memory"); }
      upload_(trow, row);
      auto enc = stream.begin_compute();
      _d.ops->add_row(enc, emb, row, L, hidden);
      enc.end();
      stream.commit().wait();
    }

    const double denom = std::max(1.0 - t, p.t_eps);

    // The reference's own guard, quirks included: STRICT inequalities,
    // plus an `lo == 0` escape that turns the interval off entirely.
    // t2i uses inclusive bounds and no escape. Two spellings of one idea
    // in one file, and reproducing the wrong one silently changes which
    // steps are guided.
    const bool use_cfg = (t > p.cfg_interval_lo && t < p.cfg_interval_hi) ||
                         p.cfg_interval_lo == 0.0;
    const bool guided = needs_cfg && use_cfg;
    const int n_branch = guided ? (int)branches.size() : 1;

    // THE BRANCHES RUN THROUGH THE STACK TOGETHER, layer by layer,
    // rather than one whole stack each. It is the same arithmetic --
    // the branches never read each other -- but it is one traversal of
    // the 42 layers per step instead of `n_branch`, which is what makes
    // streaming affordable: a three-branch edit would otherwise read
    // the whole checkpoint three times per step.
    std::vector<U15Backbone::PassSlot> slots;
    slots.reserve((std::size_t)n_branch);
    for (int bi = 0; bi < n_branch; ++bi) {
      Branch& b = branches[(std::size_t)bi];

      // The image tokens are re-run every step against the FROZEN
      // prefix and then discarded: they are written into the cache at
      // `prefix` and the cache length never grows.
      //
      // Copy the embedding into the working buffer -- the pass updates
      // its input in place, and every branch starts from the SAME
      // embedding (only the prefix and the positions differ).
      std::memcpy(hstate[(std::size_t)bi].contents(), emb.contents(),
                  (std::size_t)L * hidden * kElt);

      U15Backbone::PassSlot sl;
      sl.x = &hstate[(std::size_t)bi];
      sl.kv = &b.kv;
      sl.kv_off = (int)b.ids.size();
      sl.kv_len = (int)b.ids.size() + L;
      sl.pos_t = &b.pt;
      sl.pos_h = &b.ph;
      sl.pos_w = &b.pw;
      slots.push_back(sl);
    }

    // Bidirectional regardless of how the PREFILL attended: the
    // reference passes attention_mask=None for every denoise pass,
    // including the edit ones.
    if (!_d.backbone->forward_many(stream, slots, L, Expert::Gen,
                                   Attn::Bidirectional, err)) {
      return false;
    }

    // The pixel head is per branch and runs AFTER the stack: it is two
    // convolutions on one branch's hidden state, so there is nothing to
    // share and no reason to interleave it.
    for (int bi = 0; bi < n_branch; ++bi) {
      Branch& b = branches[(std::size_t)bi];
      const SharedBuffer& h = hstate[(std::size_t)bi];
      {
        auto enc = stream.begin_compute();
        _d.backbone->final_norm(enc, h, L, Expert::Gen);
        enc.end();
      }
      if (!_d.image->decode(stream, h, th, tw, px_out, err)) {
        return false;
      }
      stream.commit().wait();
      download_(px_out, &x_pred);

      if (std::getenv("VPIPE_U15_DEBUG") != nullptr) {
        double xm = 0, xa = 0;
        for (std::size_t i = 0; i < x_pred.size(); ++i) {
          xm += x_pred[i];
          xa = std::max(xa, (double)std::fabs(x_pred[i]));
        }
        std::fprintf(stderr,
                     "[u15] step %d branch %d t=%.4f x_pred mean %.4f "
                     "absmax %.4f\n", step, bi, t,
                     xm / (double)x_pred.size(), xa);
      }

      for (std::size_t i = 0; i < b.v.size(); ++i) {
        b.v[i] = (float)(((double)x_pred[i] - (double)img[i]) / denom);
      }
    }

    // ---- CFG ------------------------------------------------------
    //
    // The reference's five-way selection. The two-branch forms are NOT
    // the same expression with a term dropped: which branch plays the
    // baseline differs, and so does what each scale multiplies.
    const std::vector<float>& v_cond = branches[(std::size_t)i_cond].v;
    std::vector<float> combined;
    const std::vector<float>* vp = &v_cond;

    if (guided) {
      combined.resize(v_cond.size());
      if (i_img >= 0 && i_unc < 0) {
        // img_cfg == 1: guide the instruction away from "these images,
        // no instruction". The IMAGE-conditioned branch is the baseline.
        const std::vector<float>& v_img = branches[(std::size_t)i_img].v;
        for (std::size_t i = 0; i < combined.size(); ++i) {
          combined[i] = (float)((double)v_img[i] +
                                cfg * ((double)v_cond[i] - (double)v_img[i]));
        }
      } else if (i_img < 0 && i_unc >= 0) {
        // cfg == img_cfg (or plain t2i): one axis, unconditional
        // baseline.
        const std::vector<float>& v_unc = branches[(std::size_t)i_unc].v;
        if (p.cfg_norm == CfgNorm::CfgZeroStar) {
          const double alpha = ref::optimized_scale(
              v_cond.data(), v_unc.data(), v_cond.size());
          if (step <= 0) {
            // The reference outputs EXACTLY zero on the first step in
            // this mode. Not an approximation of small -- zero.
            std::fill(combined.begin(), combined.end(), 0.0f);
          } else {
            for (std::size_t i = 0; i < combined.size(); ++i) {
              const double a = alpha * (double)v_unc[i];
              combined[i] =
                  (float)(a + cfg * ((double)v_cond[i] - a));
            }
          }
        } else {
          for (std::size_t i = 0; i < combined.size(); ++i) {
            combined[i] = (float)((double)v_unc[i] +
                                  cfg * ((double)v_cond[i] -
                                         (double)v_unc[i]));
          }
        }
      } else if (i_img >= 0 && i_unc >= 0) {
        // Both axes: the text instruction is guided against the
        // image-only branch, and the images against the empty one.
        const std::vector<float>& v_img = branches[(std::size_t)i_img].v;
        const std::vector<float>& v_unc = branches[(std::size_t)i_unc].v;
        for (std::size_t i = 0; i < combined.size(); ++i) {
          combined[i] = (float)(
              (double)v_unc[i] +
              cfg * ((double)v_cond[i] - (double)v_img[i]) +
              icfg * ((double)v_img[i] - (double)v_unc[i]));
        }
      } else {
        combined = v_cond;
      }

      // The renormalisations apply to whichever combination was made.
      // cfg_zero_star is handled above and the edit path refuses it (the
      // reference asserts cfg_norm in [none, global, channel] there).
      if (p.cfg_norm == CfgNorm::Global) {
        double nc = 0.0, nv = 0.0;
        for (std::size_t i = 0; i < combined.size(); ++i) {
          nc += (double)v_cond[i] * v_cond[i];
          nv += (double)combined[i] * combined[i];
        }
        const double sc = std::min(1.0, std::sqrt(nc) /
                                            (std::sqrt(nv) + 1e-8));
        for (auto& x : combined) { x = (float)(x * sc); }
      } else if (p.cfg_norm == CfgNorm::Channel) {
        // Per-TOKEN in the reference, i.e. per 32x32x3 patch here. The
        // image is channel-last, so a token's elements are not
        // contiguous and have to be gathered by position.
        const int ps = kPixelsPerToken;
        for (int ty = 0; ty < th; ++ty) {
          for (int tx = 0; tx < tw; ++tx) {
            double nc = 0.0, nv = 0.0;
            for (int y = 0; y < ps; ++y) {
              for (int x = 0; x < ps; ++x) {
                for (int c = 0; c < 3; ++c) {
                  const std::size_t i =
                      (((std::size_t)(ty * ps + y) * W) + (tx * ps + x)) *
                          3 + c;
                  nc += (double)v_cond[i] * v_cond[i];
                  nv += (double)combined[i] * combined[i];
                }
              }
            }
            const double sc = std::min(1.0, std::sqrt(nc) /
                                                (std::sqrt(nv) + 1e-8));
            for (int y = 0; y < ps; ++y) {
              for (int x = 0; x < ps; ++x) {
                for (int c = 0; c < 3; ++c) {
                  const std::size_t i =
                      (((std::size_t)(ty * ps + y) * W) + (tx * ps + x)) *
                          3 + c;
                  combined[i] = (float)(combined[i] * sc);
                }
              }
            }
          }
        }
      }
      vp = &combined;
    }


    // ---- Euler ----------------------------------------------------
    //
    // Done in IMAGE space rather than patch space. patchify is a pure
    // permutation and every operation here is elementwise, so the two
    // are identical -- and the CFG norms above already account for the
    // groupings that are NOT permutation-invariant.
    const double dt = t_next - t;
    for (std::size_t i = 0; i < img.size(); ++i) {
      img[i] = (float)((double)img[i] + dt * (double)(*vp)[i]);
    }
  }
  if (progress) { progress(p.steps, p.steps); }

  // ---- out ----------------------------------------------------------
  upload_(img, px_out);
  _d.image->to_u8_planar(stream, px_out, H, W, u8);
  stream.commit().wait();

  out_u8->resize((std::size_t)H * W * 3);
  std::memcpy(out_u8->data(), u8.contents(), out_u8->size());
  return true;
}

}  // namespace u15
