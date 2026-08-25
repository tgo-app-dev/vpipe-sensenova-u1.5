// The CPU reference against the goldens `tools/gen_goldens.py` dumps
// from the reference implementation.
//
// This is the test that settles semantics. Everything downstream -- the
// Metal kernels, the weight binding, the sampler -- is checked against
// the reference verified HERE, so that a kernel bug and a transcription
// bug are never in flight together.
//
// Gated on VPIPE_U15_GOLDEN_DIR. Unset => SKIPS, and says so: a vacuous
// skip looks exactly like a pass.

#include "u15-ref.h"
#include "npy.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
int g_ran  = 0;
std::string g_dir;

npy::Array
G(const std::string& name)
{
  npy::Array a = npy::load(g_dir + "/" + name + ".npy");
  if (!a.ok) {
    std::printf("  [FAIL] cannot load %s: %s\n", name.c_str(),
                a.err.c_str());
    ++g_fail;
  }
  return a;
}

// Compare against a golden and report the relative L2. `tol` is stated
// per call rather than shared, so loosening one comparison cannot
// quietly loosen the rest.
void
cmp(const std::string& what, const std::vector<float>& got,
    const npy::Array& want, double tol)
{
  ++g_ran;
  if (!want.ok) { return; }
  if (got.size() != want.data.size()) {
    std::printf("  [FAIL] %s: size %zu vs golden %zu\n", what.c_str(),
                got.size(), want.data.size());
    ++g_fail;
    return;
  }
  const double r = npy::rel_l2(got, want.data);
  const bool ok = (r <= tol) && std::isfinite(r);
  std::printf("  [%s] %-42s rel-L2 %.3e (tol %.0e)\n",
              ok ? " OK " : "FAIL", what.c_str(), r, tol);
  if (!ok) { ++g_fail; }
}

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  ++g_ran;
  if (!ok) { ++g_fail; }
}

std::vector<int>
row_as_int(const npy::Array& a, int row, int n)
{
  std::vector<int> v((std::size_t)n);
  for (int i = 0; i < n; ++i) {
    v[(std::size_t)i] = (int)std::lround(a.data[(std::size_t)row * n + i]);
  }
  return v;
}

// dims, mirroring gen_goldens.py
constexpr int HID = 64, HEADS = 4, KV = 2, HD = 32, INTER = 128;
constexpr double EPS = 1e-6, TH_T = 5000000.0, TH_HW = 10000.0;

// -------------------------------------------------------------------
void
test_rope_ladder()
{
  std::printf("rope frequency ladders\n");
  const auto t  = G("rope.table.inv_freq_t");
  const auto hw = G("rope.table.inv_freq_hw");
  cmp("inv_freq t  (dim 16, theta 5e6)",
      u15::ref::rope_inv_freq(HD / 2, TH_T), t, 1e-6);
  cmp("inv_freq hw (dim 8,  theta 1e4)",
      u15::ref::rope_inv_freq(HD / 4, TH_HW), hw, 1e-6);
}

void
test_split_rope()
{
  std::printf("three-way split rope (t | h | w)\n");
  const auto in  = G("rope.input");
  const auto idx = G("rope.indexes");
  if (!in.ok || !idx.ok) { return; }
  const int S = in.shape[1];

  const auto pt = row_as_int(idx, 0, S);
  const auto ph = row_as_int(idx, 1, S);
  const auto pw = row_as_int(idx, 2, S);

  for (const char* tag : {"und", "gen"}) {
    const std::string sfx =
        (std::string(tag) == "gen") ? "_mot_gen" : "";
    const auto qw  = G("rope.attn.q_proj" + sfx + ".weight");
    const auto kw  = G("rope.attn.k_proj" + sfx + ".weight");
    const auto qn  = G("rope.attn.q_norm" + sfx + ".weight");
    const auto qnh = G("rope.attn.q_norm_hw" + sfx + ".weight");
    const auto kn  = G("rope.attn.k_norm" + sfx + ".weight");
    const auto knh = G("rope.attn.k_norm_hw" + sfx + ".weight");
    if (!qw.ok || !qn.ok) { return; }

    u15::ref::SplitRopeParams p;
    p.head_dim = HD;
    p.theta_t = TH_T;
    p.theta_hw = TH_HW;
    p.rms_eps = EPS;

    p.heads = HEADS;
    std::vector<float> q((std::size_t)HEADS * S * HD);
    u15::ref::split_rope_project(in.data.data(), qw.data.data(),
                                 qn.data.data(), qnh.data.data(), pt, ph,
                                 pw, (std::size_t)S, HID, p, q.data());
    cmp(std::string(tag) + ".q", q, G("rope." + std::string(tag) + ".q"),
        2e-6);

    p.heads = KV;
    std::vector<float> k((std::size_t)KV * S * HD);
    u15::ref::split_rope_project(in.data.data(), kw.data.data(),
                                 kn.data.data(), knh.data.data(), pt, ph,
                                 pw, (std::size_t)S, HID, p, k.data());
    cmp(std::string(tag) + ".k", k, G("rope." + std::string(tag) + ".k"),
        2e-6);
  }
}

// The negative control that matters most: if the h and w positions are
// swapped, the golden must REJECT the result. Without this, a port that
// transposes the two axes passes every other check in this file --
// because the ONLY thing distinguishing them is which position feeds
// which quarter of the head.
void
test_rope_axis_control()
{
  std::printf("negative control: h/w axes are not interchangeable\n");
  const auto in  = G("rope.input");
  const auto idx = G("rope.indexes");
  const auto qw  = G("rope.attn.q_proj_mot_gen.weight");
  const auto qn  = G("rope.attn.q_norm_mot_gen.weight");
  const auto qnh = G("rope.attn.q_norm_hw_mot_gen.weight");
  const auto want = G("rope.gen.q");
  if (!in.ok || !idx.ok || !qw.ok || !want.ok) { return; }
  const int S = in.shape[1];

  const auto pt = row_as_int(idx, 0, S);
  const auto ph = row_as_int(idx, 1, S);
  const auto pw = row_as_int(idx, 2, S);

  u15::ref::SplitRopeParams p;
  p.head_dim = HD; p.heads = HEADS;
  p.theta_t = TH_T; p.theta_hw = TH_HW; p.rms_eps = EPS;

  std::vector<float> swapped((std::size_t)HEADS * S * HD);
  u15::ref::split_rope_project(in.data.data(), qw.data.data(),
                               qn.data.data(), qnh.data.data(), pt,
                               pw, ph,   // <- h and w exchanged
                               (std::size_t)S, HID, p, swapped.data());
  const double r = npy::rel_l2(swapped, want.data);
  check(r > 1e-3,
        "swapping h/w gives a DIFFERENT answer (rel-L2 " +
            std::to_string(r) + ")");

  // And the same for the two thetas: if the hw rope used the t theta,
  // the result must move. A shared ladder is an easy simplification to
  // make by accident.
  p.theta_hw = TH_T;
  std::vector<float> wrong_theta((std::size_t)HEADS * S * HD);
  u15::ref::split_rope_project(in.data.data(), qw.data.data(),
                               qn.data.data(), qnh.data.data(), pt, ph, pw,
                               (std::size_t)S, HID, p, wrong_theta.data());
  const double r2 = npy::rel_l2(wrong_theta, want.data);
  check(r2 > 1e-3,
        "theta_hw != theta_t matters (rel-L2 " + std::to_string(r2) + ")");
}

void
test_attention()
{
  std::printf("attention + o_proj (the gen expert, unmasked)\n");
  const auto in  = G("rope.input");
  const auto idx = G("rope.indexes");
  const auto qg  = G("rope.gen.q");
  const auto kg  = G("rope.gen.k");
  const auto vw  = G("rope.attn.v_proj_mot_gen.weight");
  const auto ow  = G("rope.attn.o_proj_mot_gen.weight");
  if (!in.ok || !qg.ok || !vw.ok) { return; }
  const int S = in.shape[1];

  // v: plain projection, then the head-major transpose.
  std::vector<float> vt((std::size_t)S * KV * HD);
  u15::ref::linear(in.data.data(), vw.data.data(), nullptr, (std::size_t)S,
                   HID, (std::size_t)KV * HD, vt.data());
  std::vector<float> v((std::size_t)KV * S * HD);
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < KV; ++h) {
      for (int c = 0; c < HD; ++c) {
        v[((std::size_t)h * S + s) * HD + c] =
            vt[((std::size_t)s * KV + h) * HD + c];
      }
    }
  }

  std::vector<float> att((std::size_t)S * HEADS * HD);
  u15::ref::attention(qg.data.data(), kg.data.data(), v.data(), nullptr,
                      HEADS, KV, (std::size_t)S, (std::size_t)S, HD,
                      att.data());
  std::vector<float> out((std::size_t)S * HID);
  u15::ref::linear(att.data(), ow.data.data(), nullptr, (std::size_t)S,
                   (std::size_t)HEADS * HD, HID, out.data());
  cmp("gen attention output", out, G("rope.gen.attn_out"), 2e-6);
  (void)idx;
}

void
test_block_causal_mask()
{
  std::printf("block-causal mask\n");
  const auto idx  = G("layer.indexes");
  const auto want = G("layer.und.mask");
  if (!idx.ok || !want.ok) { return; }
  const int S = idx.shape[1];
  const auto t = row_as_int(idx, 0, S);
  const auto m = u15::ref::block_causal_mask(t);

  // -inf compares badly under rel-L2, so compare the PATTERN.
  ++g_ran;
  bool same = m.size() == want.data.size();
  if (same) {
    for (std::size_t i = 0; i < m.size(); ++i) {
      const bool a = std::isfinite(m[i]);
      const bool b = std::isfinite(want.data[i]);
      if (a != b) { same = false; break; }
      if (a && std::fabs(m[i] - want.data[i]) > 1e-6) {
        same = false; break;
      }
    }
  }
  std::printf("  [%s] mask pattern matches (causal across t-blocks, "
              "bidirectional within)\n", same ? " OK " : "FAIL");
  if (!same) { ++g_fail; }
}

void
test_decoder_layer()
{
  std::printf("MoT decoder layer, both experts\n");
  const auto in  = G("layer.input");
  const auto idx = G("layer.indexes");
  if (!in.ok || !idx.ok) { return; }
  const int S = in.shape[1];
  const auto pt = row_as_int(idx, 0, S);
  const auto ph = row_as_int(idx, 1, S);
  const auto pw = row_as_int(idx, 2, S);

  u15::ref::LayerDims d;
  d.hidden = HID; d.heads = HEADS; d.kv_heads = KV; d.head_dim = HD;
  d.intermediate = INTER; d.rms_eps = EPS;
  d.theta_t = TH_T; d.theta_hw = TH_HW;

  struct Held { npy::Array a; };
  const auto bind = [&](const std::string& sfx, const std::string& lsfx,
                        std::vector<npy::Array>& keep) {
    u15::ref::LayerWeights w;
    const auto take = [&](const std::string& n) -> const float* {
      keep.push_back(G(n));
      return keep.back().ok ? keep.back().data.data() : nullptr;
    };
    w.input_layernorm = take("layer.input_layernorm" + lsfx + ".weight");
    w.post_attention_layernorm =
        take("layer.post_attention_layernorm" + lsfx + ".weight");
    w.q_proj = take("layer.self_attn.q_proj" + sfx + ".weight");
    w.k_proj = take("layer.self_attn.k_proj" + sfx + ".weight");
    w.v_proj = take("layer.self_attn.v_proj" + sfx + ".weight");
    w.o_proj = take("layer.self_attn.o_proj" + sfx + ".weight");
    w.q_norm = take("layer.self_attn.q_norm" + sfx + ".weight");
    w.k_norm = take("layer.self_attn.k_norm" + sfx + ".weight");
    w.q_norm_hw = take("layer.self_attn.q_norm_hw" + sfx + ".weight");
    w.k_norm_hw = take("layer.self_attn.k_norm_hw" + sfx + ".weight");
    w.gate_proj = take("layer.mlp" + lsfx + ".gate_proj.weight");
    w.up_proj   = take("layer.mlp" + lsfx + ".up_proj.weight");
    w.down_proj = take("layer.mlp" + lsfx + ".down_proj.weight");
    return w;
  };

  // gen: the _mot_gen weights, and NO mask (bidirectional).
  std::vector<npy::Array> keep_g;
  const auto wg = bind("_mot_gen", "_mot_gen", keep_g);
  std::vector<float> og((std::size_t)S * HID);
  u15::ref::decoder_layer(in.data.data(), wg, d, pt, ph, pw, nullptr,
                          (std::size_t)S, og.data());
  cmp("gen expert output", og, G("layer.gen.output"), 3e-6);

  // und: the bare weights, WITH the block-causal mask.
  std::vector<npy::Array> keep_u;
  const auto wu = bind("", "", keep_u);
  const auto mask = u15::ref::block_causal_mask(pt);
  std::vector<float> ou((std::size_t)S * HID);
  u15::ref::decoder_layer(in.data.data(), wu, d, pt, ph, pw, mask.data(),
                          (std::size_t)S, ou.data());
  cmp("und expert output", ou, G("layer.und.output"), 3e-6);

  // The experts must NOT agree -- a port that bound one weight set for
  // both would still pass one of the two comparisons above.
  const auto gap = G("layer.expert_gap");
  if (gap.ok) {
    double mx = 0.0;
    for (std::size_t i = 0; i < og.size(); ++i) {
      mx = std::max(mx, (double)std::fabs(og[i] - ou[i]));
    }
    check(mx > 0.1 * (double)gap.data[0],
          "the two experts genuinely differ (max abs " +
              std::to_string(mx) + ")");
  }
}

void
test_vision()
{
  std::printf("patch embedder\n");
  const auto px = G("vision.input");
  const auto gw = G("vision.grid_hw");
  const auto pw = G("vision.patch_embedding.weight");
  const auto pb = G("vision.patch_embedding.bias");
  const auto dw = G("vision.dense_embedding.weight");
  const auto db = G("vision.dense_embedding.bias");
  if (!px.ok || !gw.ok || !pw.ok) { return; }

  const int gh = (int)std::lround(gw.data[0]);
  const int gwid = (int)std::lround(gw.data[1]);
  const int hidden = pw.shape[0];       // 16
  const int patch  = pw.shape[2];       // 4
  const int llm    = dw.shape[0];       // 64
  const int merge  = dw.shape[2];       // 2

  u15::ref::VisionWeights w;
  w.patch_w = pw.data.data(); w.patch_b = pb.data.data();
  w.dense_w = dw.data.data(); w.dense_b = db.data.data();

  std::vector<float> out(
      (std::size_t)(gh / merge) * (gwid / merge) * llm);
  u15::ref::vision_embed(px.data.data(), w, gh, gwid, patch, hidden, llm,
                         merge, 10000.0, out.data());
  cmp("vision output (gelu + 2d interleaved rope + merge)", out,
      G("vision.output"), 5e-6);
}

void
test_fm_head()
{
  std::printf("pixel head (ConvDecoder)\n");
  const auto x  = G("fm_head.input");
  const auto c1w = G("fm_head.conv1.weight");
  const auto c1b = G("fm_head.conv1.bias");
  const auto c2w = G("fm_head.conv2.weight");
  const auto c2b = G("fm_head.conv2.bias");
  if (!x.ok || !c1w.ok) { return; }

  const int in_ch = x.shape[1], th = x.shape[2], tw = x.shape[3];
  const int hidden_ch = c1w.shape[0];

  u15::ref::HeadWeights w;
  w.conv1_w = c1w.data.data(); w.conv1_b = c1b.data.data();
  w.conv2_w = c2w.data.data(); w.conv2_b = c2b.data.data();

  std::vector<float> out((std::size_t)3 * (th * 32) * (tw * 32));
  u15::ref::conv_decoder(x.data.data(), w, in_ch, th, tw, hidden_ch,
                         out.data());
  cmp("fm_head output", out, G("fm_head.output"), 5e-6);
}

void
test_patchify()
{
  std::printf("patchify -- the TWO packings\n");
  const auto img = G("patch.image");
  if (!img.ok) { return; }
  const int c = img.shape[1], h = img.shape[2], w = img.shape[3];

  std::vector<float> z((std::size_t)c * h * w);
  u15::ref::patchify_channel_last(img.data.data(), c, h, w, 4, z.data());
  cmp("patch 4, channel-LAST (z)", z, G("patch.p4_channel_last"), 0.0);

  std::vector<float> s((std::size_t)c * h * w);
  u15::ref::patchify_channel_first(img.data.data(), c, h, w, 2, s.data());
  cmp("patch 2, channel-FIRST (embedder input)", s,
      G("patch.p2_channel_first"), 0.0);

  // The two must differ -- they are computed from ONE image in ONE step,
  // and using either packing for both is the trap this pins.
  const auto a = G("patch.p4_channel_last");
  check(z.size() == s.size() && z != s,
        "the two packings are genuinely different");

  std::vector<float> back((std::size_t)c * h * w);
  u15::ref::unpatchify_channel_last(z.data(), c, h, w, 4, back.data());
  cmp("unpatchify round-trip", back, img, 0.0);
  (void)a;
}

void
test_schedule()
{
  std::printf("schedule, sinusoid, cfg\n");
  cmp("timesteps shift=1", u15::ref::time_schedule(10, 1.0),
      G("sched.timesteps_shift1"), 1e-6);
  cmp("timesteps shift=3", u15::ref::time_schedule(10, 3.0),
      G("sched.timesteps_shift3"), 1e-6);

  const auto ti = G("tstep.input");
  const auto ts = G("tstep.sinusoid");
  if (ti.ok && ts.ok) {
    const int dim = ts.shape[1];
    std::vector<float> all;
    for (std::size_t i = 0; i < ti.data.size(); ++i) {
      const auto e = u15::ref::timestep_sinusoid((double)ti.data[i], dim);
      all.insert(all.end(), e.begin(), e.end());
    }
    cmp("timestep sinusoid (cos first)", all, ts, 1e-6);
  }

  const auto cp = G("cfg.pos");
  const auto cn = G("cfg.neg");
  const auto ca = G("cfg.alpha");
  if (cp.ok && cn.ok && ca.ok) {
    const int rows = cp.shape[0], n = cp.shape[1];
    std::vector<float> got((std::size_t)rows);
    for (int r = 0; r < rows; ++r) {
      got[(std::size_t)r] = (float)u15::ref::optimized_scale(
          cp.data.data() + (std::size_t)r * n,
          cn.data.data() + (std::size_t)r * n, (std::size_t)n);
    }
    cmp("cfg_zero_star alpha", got, ca, 1e-6);
  }
}

// -------------------------------------------------------------------
// reference-image preprocessing (the EDIT path)
// -------------------------------------------------------------------
void
test_reference_image()
{
  std::printf("reference-image preprocessing\n");
  // The golden's metadata lives in image.json beside the arrays.
  std::string txt;
  {
    std::ifstream f(g_dir + "/image.json", std::ios::binary);
    if (!f) {
      std::printf("  image.json absent -- run tools/gen_image_golden.py. "
                  "This section did NOT run.\n");
      return;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    txt = ss.str();
  }

  // A deliberately small hand-parse: the test already links no JSON
  // reader, and the fields wanted are four integers per case.
  const auto find_int = [&txt](std::size_t from, const char* key,
                               int* out) -> std::size_t {
    const auto at = txt.find(key, from);
    if (at == std::string::npos) { return std::string::npos; }
    const auto colon = txt.find(':', at);
    *out = std::atoi(txt.c_str() + colon + 1);
    return colon;
  };

  std::size_t pos = 0;
  for (const char* tag : {"a", "b"}) {
    int sw = 0, sh = 0, rw = 0, rh = 0, gh = 0, gw = 0;
    pos = find_int(pos, "\"src_w\"", &sw);
    if (pos == std::string::npos) { break; }
    pos = find_int(pos, "\"src_h\"", &sh);
    pos = find_int(pos, "\"resized_w\"", &rw);
    pos = find_int(pos, "\"resized_h\"", &rh);
    pos = find_int(pos, "\"grid_h\"", &gh);
    pos = find_int(pos, "\"grid_w\"", &gw);

    // ---- smart_resize picks the same size --------------------------
    int oh = 0, ow = 0;
    const bool ok = u15::ref::smart_resize(sh, sw, 32, 512L * 512,
                                           2048L * 2048, &oh, &ow);
    check(ok && oh == rh && ow == rw,
          std::string(tag) + ": smart_resize " + std::to_string(sw) + "x" +
              std::to_string(sh) + " -> " + std::to_string(ow) + "x" +
              std::to_string(oh) + " (want " + std::to_string(rw) + "x" +
              std::to_string(rh) + ")");

    const auto src = G(std::string("img.") + tag + ".src");
    const auto want_r = G(std::string("img.") + tag + ".resized");
    const auto want_p = G(std::string("img.") + tag + ".patches");
    if (!src.ok || !want_r.ok || !want_p.ok) { continue; }

    std::vector<unsigned char> s8(src.data.size());
    for (std::size_t i = 0; i < src.data.size(); ++i) {
      s8[i] = (unsigned char)src.data[i];
    }
    std::vector<unsigned char> r8((std::size_t)ow * oh * 3);
    u15::ref::resize_bicubic_u8(s8.data(), sw, sh, r8.data(), ow, oh, 3);

    // Compare the RESIZE on its own, so a filter mismatch is localised
    // rather than blamed on the normalisation downstream. PIL runs its
    // passes in fixed point; this runs in double and rounds, so a
    // handful of pixels may differ by one level. What must NOT happen is
    // a systematic difference, which is what the mean catches.
    double max_d = 0.0, sum_d = 0.0;
    for (std::size_t i = 0; i < r8.size(); ++i) {
      const double d = std::fabs((double)r8[i] - (double)want_r.data[i]);
      max_d = std::max(max_d, d);
      sum_d += d;
    }
    const double mean_d = sum_d / (double)r8.size();
    std::printf("       %s resize: max %.0f levels, mean %.4f\n", tag,
                max_d, mean_d);
    check(max_d <= 1.0 && mean_d < 0.05,
          std::string(tag) + ": bicubic matches PIL to within rounding");

    // ---- the full patch tensor -------------------------------------
    std::vector<float> got((std::size_t)gh * gw * 3 * 16 * 16);
    u15::ref::reference_patches(r8.data(), ow, oh, 16, got.data());
    check(got.size() == want_p.data.size(),
          std::string(tag) + ": patch tensor is [" + std::to_string(gh * gw) +
              ", 768]");
    if (got.size() == want_p.data.size()) {
      const double r = npy::rel_l2(got, want_p.data);
      std::printf("       %s patches: rel-L2 %.3e\n", tag, r);
      check(r < 5e-3,
            std::string(tag) + ": ImageNet-normalised patches match");
    }
  }

  // The normalisation is NOT the generated image's. Stated as a check
  // because the two live in one file and the wrong one runs cleanly.
  {
    std::vector<unsigned char> mid(3 * 16 * 16, 128);
    std::vector<float> p(3 * 16 * 16);
    u15::ref::reference_patches(mid.data(), 16, 16, 16, p.data());
    // 128/255 = 0.502; ImageNet would give (0.502-0.485)/0.229 = 0.074
    // for red, while a 0.5/0.5 normalisation would give ~0.004.
    check(std::fabs(p[0] - 0.0743f) < 0.01f,
          "reference images use IMAGENET normalisation, not the "
          "generated image's 0.5/0.5");
  }
}

}  // namespace

int
main()
{
  const char* d = std::getenv("VPIPE_U15_GOLDEN_DIR");
  if (d == nullptr || *d == '\0') {
    std::printf("SKIPPED: VPIPE_U15_GOLDEN_DIR unset -- this test did "
                "NOT run.\nGenerate with tools/gen_goldens.py.\n");
    return 0;
  }
  g_dir = d;
  std::printf("goldens: %s\n\n", g_dir.c_str());

  test_rope_ladder();
  test_split_rope();
  test_rope_axis_control();
  test_attention();
  test_block_causal_mask();
  test_decoder_layer();
  test_vision();
  test_fm_head();
  test_patchify();
  test_schedule();
  test_reference_image();

  std::printf("\n%d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
