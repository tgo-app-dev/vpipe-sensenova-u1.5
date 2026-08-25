#include "u15-image.h"

#include "u15-ref.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

using vpipe::metal_compute::CommandStream;
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

}  // namespace

std::unique_ptr<ImagePath>
ImagePath::create(MetalOps* ops, const U15Config& cfg,
                  const U15Weights* w, std::string* err)
{
  const auto fail = [err](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<ImagePath>{};
  };
  if (ops == nullptr || w == nullptr) { return fail("no ops or weights"); }

  auto p = std::unique_ptr<ImagePath>(new ImagePath());
  p->_ops = ops;
  p->_w = w;
  p->_cfg = cfg;

  // The vision rope ladder spans HALF the embedding: the first half is
  // rotated by the column, the second by the row, and both use the same
  // frequencies.
  p->_vis_inv = ops->make_inv_freq(cfg.vision.hidden_size / 2,
                                   cfg.vision.rope_theta_vision);
  if (p->_vis_inv.empty()) { return fail("could not build the vision rope"); }

  if (!p->build_conv_weights_(err)) { return nullptr; }
  return p;
}

bool
ImagePath::build_conv_weights_(std::string* err)
{
  // The checkpoint stores a conv weight [cout][cin][ky][kx]. im2col
  // emits columns ordered (ky, kx, cin) -- kernel position major,
  // channel minor -- so the GEMM needs [cout][ky][kx][cin]. Same bytes,
  // different order; feeding the raw layout would multiply a correctly
  // shaped matrix by the wrong elements.
  // The pixel head ships BF16 in this checkpoint while the two scalar
  // embedders and both patch embedders ship F32 -- the dtype is
  // per-tensor in the safetensors header, so it is DERIVED from the
  // buffer size here rather than assumed from which module it belongs
  // to. (Assuming cost a debug cycle: the head was first bound on the
  // f32 path and its shape check refused it.)
  const auto elem_of = [](const SharedBuffer& b, std::size_t n) {
    if (b.byte_size() == n * 4) { return 4; }
    if (b.byte_size() == n * 2) { return 2; }
    return 0;
  };
  const auto at = [](const SharedBuffer& b, int elt, std::size_t i) {
    if (elt == 4) { return static_cast<const float*>(b.contents())[i]; }
    const std::uint32_t u =
        (std::uint32_t)static_cast<const std::uint16_t*>(b.contents())[i]
        << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };

  const auto repack = [&](const SharedBuffer& src, int cout, int cin,
                          SharedBuffer* dst) {
    if (src.empty()) { return false; }
    const std::size_t n = (std::size_t)cout * cin * 9;
    const int elt = elem_of(src, n);
    if (elt == 0) { return false; }
    *dst = _ops->mc()->make_shared_buffer(n * kElt);
    if (dst->empty()) { return false; }
    auto* d = static_cast<std::uint16_t*>(dst->contents());
    for (int o = 0; o < cout; ++o) {
      for (int c = 0; c < cin; ++c) {
        for (int k = 0; k < 9; ++k) {
          d[((std::size_t)o * 9 + k) * cin + c] =
              to_bf16_(at(src, elt, ((std::size_t)o * cin + c) * 9 + k));
        }
      }
    }
    return true;
  };
  const auto to_bf16_vec = [&](const SharedBuffer& src, int n,
                               SharedBuffer* dst) {
    if (src.empty()) { return false; }
    const int elt = elem_of(src, (std::size_t)n);
    if (elt == 0) { return false; }
    *dst = _ops->mc()->make_shared_buffer((std::size_t)n * kElt);
    if (dst->empty()) { return false; }
    auto* d = static_cast<std::uint16_t*>(dst->contents());
    for (int i = 0; i < n; ++i) { d[i] = to_bf16_(at(src, elt, (std::size_t)i)); }
    return true;
  };

  const Trunk& t = _w->trunk();
  // conv1: [1024][1024][3][3]; conv2: [192][256][3][3].
  const int c1_out = _cfg.vision.hidden_size;            // 1024
  const int c1_in  = _cfg.llm.hidden_size / 4;           // 4096/4 = 1024
  const int c2_in  = c1_out / 4;                         // 256
  const int c2_out = 3 * 8 * 8;                          // 192
  if (!repack(t.conv1_w, c1_out, c1_in, &_conv1_w) ||
      !repack(t.conv2_w, c2_out, c2_in, &_conv2_w) ||
      !to_bf16_vec(t.conv1_b, c1_out, &_conv1_b) ||
      !to_bf16_vec(t.conv2_b, c2_out, &_conv2_b)) {
    if (err != nullptr) {
      *err = "the pixel head's conv weights are not the expected shape";
    }
    return false;
  }
  return true;
}

void
ImagePath::patchify_for_embed(const float* img, int h, int w, int patch_px,
                              float* out)
{
  // `img` is CHANNEL-LAST [h][w][3] -- the layout the pixel head emits
  // and the u8 conversion consumes -- while the packing the embedder
  // needs is channel-FIRST within each patch. Both facts are true at
  // once, which is what makes this the easiest call in the file to get
  // wrong; see the note on patchify_channel_first_hwc.
  ref::patchify_channel_first_hwc(img, 3, h, w, patch_px, out);
}

bool
ImagePath::embed(CommandStream& stream, const SharedBuffer& px, int grid_h,
                 int grid_w, bool gen, const SharedBuffer& out,
                 std::string* err)
{
  const int C = _cfg.vision.hidden_size;
  const int LLM = _cfg.vision.llm_hidden_size;
  const int m = _cfg.vision.merge_size();
  const int p = _cfg.vision.patch_size;
  const int n = grid_h * grid_w;
  const int pin = 3 * p * p;

  if (grid_h % m != 0 || grid_w % m != 0) {
    if (err != nullptr) {
      *err = "the patch grid is not a whole number of " +
             std::to_string(m) + "x" + std::to_string(m) + " tiles";
    }
    return false;
  }

  if (n > _tokens) {
    auto* mc = _ops->mc();
    _emb = mc->make_shared_buffer((std::size_t)n * C * kElt);
    _merged = mc->make_shared_buffer(
        (std::size_t)(n / (m * m)) * C * m * m * kElt);
    if (_emb.empty() || _merged.empty()) {
      if (err != nullptr) { *err = "out of memory in the patch embedder"; }
      return false;
    }
    _tokens = n;
  }

  const Trunk& t = _w->trunk();
  const SharedBuffer& pw = gen ? t.gen_patch_w : t.und_patch_w;
  const SharedBuffer& pb = gen ? t.gen_patch_b : t.und_patch_b;
  const SharedBuffer& dw = gen ? t.gen_dense_w : t.und_dense_w;
  const SharedBuffer& db = gen ? t.gen_dense_b : t.und_dense_b;

  {
    auto enc = stream.begin_compute();
    // patch_embedding is Conv2d(3 -> C, k=p, s=p) whose kernel EQUALS
    // its input, so it is a per-patch linear and the weight's memory
    // order [C][3][p][p] is already [C][3*p*p]. That only lines up
    // because the patch was packed channel-FIRST.
    _ops->linear(enc, px, pw, &pb, _emb, n, pin, C);
    _ops->gelu_erf(enc, _emb, n * C);
    _ops->vision_rope2d(enc, _emb, _vis_inv, n, C, grid_w);
    _ops->merge_tiles(enc, _emb, _merged, grid_h, grid_w, C, m);
    // dense_embedding: no activation after it.
    _ops->linear(enc, _merged, dw, &db, out, n / (m * m), C * m * m, LLM);
  }
  return true;
}

std::vector<float>
ImagePath::timestep_row(double t, double noise_scale_ratio) const
{
  const int H = _cfg.llm.hidden_size;
  const Trunk& tr = _w->trunk();

  // sinusoid(256) -> Linear(256 -> H) -> SiLU -> Linear(H -> H)
  // BF16 IN, float math. The loader hands these over as bf16 whatever
  // the checkpoint stored (see bind_bf16_scalar_), so both the width
  // and the frequency count come from a 2-byte element. Reading them as
  // float -- which this did while the first release happened to store
  // F32 -- halves `freq` AND reinterprets the bits, and the result is a
  // rendered image that looks like a picture of nothing.
  const auto f32 = [](const SharedBuffer& b) {
    const std::size_t n = b.byte_size() / 2;
    std::vector<float> v(n);
    const auto* p = static_cast<const std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t u = (std::uint32_t)p[i] << 16;
      std::memcpy(&v[i], &u, 4);
    }
    return v;
  };
  const auto run = [&](double v, const SharedBuffer& w0,
                       const SharedBuffer& b0, const SharedBuffer& w2,
                       const SharedBuffer& b2) {
    std::vector<float> out((std::size_t)H, 0.0f);
    if (w0.empty() || w2.empty()) { return out; }
    const int freq = (int)(w0.byte_size() / 2 / (std::size_t)H);
    const std::vector<float> s = ref::timestep_sinusoid(v, freq);
    const std::vector<float> fw0 = f32(w0), fb0 = f32(b0);
    const std::vector<float> fw2 = f32(w2), fb2 = f32(b2);
    std::vector<float> h((std::size_t)H);
    ref::linear(s.data(), fw0.data(), fb0.data(), 1,
                (std::size_t)freq, (std::size_t)H, h.data());
    ref::silu(h.data(), h.size());
    ref::linear(h.data(), fw2.data(), fb2.data(), 1,
                (std::size_t)H, (std::size_t)H, out.data());
    return out;
  };

  std::vector<float> row = run(t, tr.tstep_w0, tr.tstep_b0, tr.tstep_w2,
                               tr.tstep_b2);
  if (_cfg.gen.add_noise_scale_embedding && !tr.nscale_w0.empty()) {
    // The reference feeds the noise scale as a RATIO of its maximum, not
    // the raw scale.
    const std::vector<float> ns =
        run(noise_scale_ratio, tr.nscale_w0, tr.nscale_b0, tr.nscale_w2,
            tr.nscale_b2);
    for (std::size_t i = 0; i < row.size(); ++i) { row[i] += ns[i]; }
  }
  return row;
}

bool
ImagePath::decode(CommandStream& stream, const SharedBuffer& hidden, int th,
                  int tw, const SharedBuffer& out_px, std::string* err)
{
  const int H = _cfg.llm.hidden_size;      // 4096
  const int c1_in = H / 4;                 // 1024
  const int c1_out = _cfg.vision.hidden_size;   // 1024
  const int c2_in = c1_out / 4;            // 256
  const int c2_out = 192;                  // 3 * 8 * 8

  const std::size_t need_a = (std::size_t)c1_in * (2 * th) * (2 * tw);
  const std::size_t need_b = (std::size_t)c1_out * (2 * th) * (2 * tw);
  const std::size_t need_c = (std::size_t)c2_in * (4 * th) * (4 * tw);
  const std::size_t need_e = (std::size_t)c2_out * (4 * th) * (4 * tw);
  const std::size_t need_col =
      std::max((std::size_t)(2 * th) * (2 * tw) * 9 * c1_in,
               (std::size_t)(4 * th) * (4 * tw) * 9 * c2_in);

  const int pix = th * tw;
  if (pix > _pix) {
    auto* mc = _ops->mc();
    _a = mc->make_shared_buffer(need_a * kElt);
    _b = mc->make_shared_buffer(std::max(need_b, need_e) * kElt);
    _c = mc->make_shared_buffer(need_c * kElt);
    _col = mc->make_shared_buffer(need_col * kElt);
    if (_a.empty() || _b.empty() || _c.empty() || _col.empty()) {
      if (err != nullptr) { *err = "out of memory in the pixel head"; }
      return false;
    }
    _pix = pix;
  }

  {
    auto enc = stream.begin_compute();
    // The hidden states ARE a [th][tw][4096] channel-last map: tokens
    // are in row-major (h, w) order, so no reshape is needed.
    _ops->pixel_shuffle_hwc(enc, hidden, _a, H, th, tw, 2);
    _ops->conv3x3_hwc(enc, _a, _col, _conv1_w, &_conv1_b, _b, 2 * th,
                      2 * tw, c1_in, c1_out);
    _ops->gelu_erf(enc, _b, (int)need_b);
    _ops->pixel_shuffle_hwc(enc, _b, _c, c1_out, 2 * th, 2 * tw, 2);
  }
  {
    auto enc = stream.begin_compute();
    // conv2 has NO activation after it.
    _ops->conv3x3_hwc(enc, _c, _col, _conv2_w, &_conv2_b, _b, 4 * th,
                      4 * tw, c2_in, c2_out);
    _ops->pixel_shuffle_hwc(enc, _b, out_px, c2_out, 4 * th, 4 * tw, 8);
  }
  return true;
}

void
ImagePath::to_u8_planar(CommandStream& stream, const SharedBuffer& px,
                        int h, int w, const SharedBuffer& u8)
{
  auto enc = stream.begin_compute();
  _ops->to_u8_planar(enc, px, u8, h, w);
}

}  // namespace u15
