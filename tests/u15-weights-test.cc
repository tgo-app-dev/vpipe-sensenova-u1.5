// Binding the real checkpoint: names, dtypes, shapes, and -- the one a
// structural bug would slip past -- that the two MoT experts are bound
// to DIFFERENT tensors.
//
// Binds the trunk and ONE layer. Binding all 42 is ~35 GB and answers
// nothing this does not: every layer has the same thirteen tensors per
// expert, so a naming or dtype mistake shows up in layer 0.
//
// Gated on VPIPE_U15_TEST_MODEL_PATH; SKIPS when unset and says so.

#include "u15-config.h"
#include "u15-weights.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/weight-set.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;

namespace {

int g_fail = 0;
int g_ran  = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  ++g_ran;
  if (!ok) { ++g_fail; }
}

std::string
mb(std::size_t b)
{
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.1f MB", (double)b / (1024.0 * 1024.0));
  return buf;
}

}  // namespace

int
main()
{
  const char* p = std::getenv("VPIPE_U15_TEST_MODEL_PATH");
  if (p == nullptr || *p == '\0') {
    std::printf("SKIPPED: VPIPE_U15_TEST_MODEL_PATH unset -- this test "
                "did NOT run.\n");
    return 0;
  }
  std::printf("checkpoint: %s\n", p);

  u15::U15Config cfg;
  std::string why;
  if (!u15::parse_config(p, &cfg, &why)) {
    std::printf("  [FAIL] config: %s\n", why.c_str());
    return 1;
  }

  MetalCompute mc(nullptr);
  std::shared_ptr<WeightSet> ws = WeightSet::open(p, nullptr);
  if (ws == nullptr) {
    std::printf("  [FAIL] cannot open the weight set\n");
    return 1;
  }

  u15::U15Weights::Options opt;
  opt.max_layers = 1;
  opt.with_lm_head = false;
  std::string err;
  auto w = u15::U15Weights::load(ws, &mc, cfg, opt, &err);
  check(w != nullptr, "trunk + layer 0 bind" +
                          (err.empty() ? "" : " (" + err + ")"));
  if (w == nullptr) { return 1; }

  const auto& t = w->trunk();
  const auto& L = w->layers()[0];

  // ---- shapes, from the config rather than from constants ----------
  const std::size_t H = (std::size_t)cfg.llm.hidden_size;
  const std::size_t HD = (std::size_t)cfg.llm.head_dim;
  const std::size_t NH = (std::size_t)cfg.llm.num_attention_heads;
  const std::size_t KV = (std::size_t)cfg.llm.num_key_value_heads;
  const std::size_t I = (std::size_t)cfg.llm.intermediate_size;
  const std::size_t V = (std::size_t)cfg.llm.vocab_size;

  check(t.embed_tokens.byte_size() == V * H * 2,
        "embed_tokens is [vocab, hidden] bf16 (" +
            mb(t.embed_tokens.byte_size()) + ")");
  check(t.norm.byte_size() == H * 2 && t.norm_gen.byte_size() == H * 2,
        "both final norms are [hidden] bf16");
  check(t.lm_head.empty(),
        "lm_head NOT bound (with_lm_head=false saves ~1.2 GB)");

  const bool quantized = w->quant_bits() > 0;
  if (quantized) {
    std::printf("       pack is QUANTIZED: w%d g%d\n", w->quant_bits(),
                w->quant_group());
  } else {
    std::printf("       pack is DENSE\n");
  }

  for (const auto* e : {&L.und, &L.gen}) {
    const char* tag = (e == &L.und) ? "und" : "gen";
    check(e->complete(), std::string(tag) + ": all 13 tensors bound");
    // The matrix shape checks only mean anything on a dense pack -- a
    // quantized one stores u32 codes at bits/32 of the width, plus two
    // per-group tables, so the byte counts are a different arithmetic.
    if (!quantized) {
      check(e->q.byte_size() == NH * HD * H * 2,
            std::string(tag) + ": q_proj [heads*hd, hidden] bf16");
      check(e->k.byte_size() == KV * HD * H * 2 &&
                e->v.byte_size() == KV * HD * H * 2,
            std::string(tag) + ": k/v_proj are GQA-narrow [kv*hd, hidden]");
      check(e->o.byte_size() == H * NH * HD * 2,
            std::string(tag) + ": o_proj [hidden, heads*hd]");
      check(e->gate.byte_size() == I * H * 2 &&
                e->up.byte_size() == I * H * 2 &&
                e->down.byte_size() == H * I * 2,
            std::string(tag) + ": SwiGLU trio at intermediate " +
                std::to_string(I));
    } else {
      // codes at bits/32 of the dense element count, plus scales and
      // biases at one pair per group.
      const std::size_t codes =
          (std::size_t)NH * HD * H * w->quant_bits() / 8;
      const std::size_t tabs =
          2 * (std::size_t)NH * HD * (H / w->quant_group()) * 2;
      check(e->q.byte_size() == codes + tabs,
            std::string(tag) + ": q_proj codes+scales+biases close at w" +
                std::to_string(w->quant_bits()) + "g" +
                std::to_string(w->quant_group()));
      check(e->q.quantized && e->gate.quantized,
            std::string(tag) + ": the matrices are quantized");
    }
    // All four head norms are head_dim/2 -- NOT head_dim, and not
    // head_dim/4 for the hw pair. This is the shape that says the head
    // splits 64 / 32 / 32 with the hw norm spanning both quarters.
    check(e->q_norm.byte_size() == (HD / 2) * 2 &&
              e->k_norm.byte_size() == (HD / 2) * 2 &&
              e->q_norm_hw.byte_size() == (HD / 2) * 2 &&
              e->k_norm_hw.byte_size() == (HD / 2) * 2,
          std::string(tag) + ": all four head norms are head_dim/2 = " +
              std::to_string(HD / 2));
  }

  // ---- the structural check ----------------------------------------
  // If a port bound the same tensor for both experts -- an easy mistake,
  // since the names differ only by a suffix -- every shape check above
  // still passes. These do not.
  // Whichever form the pack is in, the two experts must be distinct.
  const auto* uq = quantized ? L.und.q.codes.contents() : L.und.q.w.contents();
  const auto* gq = quantized ? L.gen.q.codes.contents() : L.gen.q.w.contents();
  const auto* ug =
      quantized ? L.und.gate.codes.contents() : L.und.gate.w.contents();
  const auto* gg =
      quantized ? L.gen.gate.codes.contents() : L.gen.gate.w.contents();
  check(uq != gq, "the two experts' q_proj are DIFFERENT buffers");
  check(ug != gg, "the two experts' gate_proj are DIFFERENT buffers");
  check(L.und.input_ln.contents() != L.gen.input_ln.contents(),
        "the two experts' input_layernorm are DIFFERENT buffers");
  check(t.und_patch_w.contents() != t.gen_patch_w.contents(),
        "the two patch embedders are DIFFERENT buffers");

  // And the values must actually differ, not merely the addresses -- a
  // conversion that silently produced zeros would pass the check above.
  {
    const auto* a = static_cast<const std::uint16_t*>(uq);
    const auto* b = static_cast<const std::uint16_t*>(gq);
    std::size_t same = 0, nz_a = 0, nz_b = 0;
    const std::size_t n = 4096;
    for (std::size_t i = 0; i < n; ++i) {
      if (a[i] == b[i]) { ++same; }
      if (a[i] != 0) { ++nz_a; }
      if (b[i] != 0) { ++nz_b; }
    }
    check(nz_a > n / 2 && nz_b > n / 2,
          "both experts' q_proj are non-zero (the F32->BF16 conversion "
          "produced values)");
    check(same < n / 2,
          "the two experts' q_proj CONTENTS differ (" +
              std::to_string(same) + "/" + std::to_string(n) + " equal)");
  }

  // ---- the dtype split ---------------------------------------------
  {
    if (quantized) {
      std::printf("       (dtype split not checked: a quantized pack's "
                  "codes are u32 for both experts)\n");
      std::printf("\n  resident: %s\n", mb(w->resident_bytes()).c_str());
      std::printf("\n%d checks, %d failed\n", g_ran, g_fail);
      return g_fail == 0 ? 0 : 1;
    }
    const auto* ig = ws->src().info(
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight");
    const auto* iu = ws->src().info(
        "language_model.model.layers.0.self_attn.q_proj.weight");
    check(ig != nullptr && iu != nullptr, "both q_proj tensors found");
    if (ig != nullptr && iu != nullptr) {
      std::printf("       checkpoint dtypes: und=%s gen=%s\n",
                  iu->dtype.c_str(), ig->dtype.c_str());
      check(iu->dtype == "BF16",
            "the und expert ships BF16 (bound with no copy)");
      check(ig->dtype == "F32",
            "the gen expert ships F32 (converted at bind)");
    }
  }

  std::printf("\n  resident: %s   of which converted: %s\n",
              mb(w->resident_bytes()).c_str(),
              mb(w->converted_bytes()).c_str());

  std::printf("\n%d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
