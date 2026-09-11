// SageAttention wired into this backbone: the QK^T product in int8.
//
// WHAT THIS FILE CAN CHECK DEPENDS ON THE GPU, and it says which on
// every run. The int8 fragment MMA has no ALU fallback, so on a box
// without matrix cores the tier declines and the model runs bf16 --
// which is not a failure, and IS the property most worth pinning there:
// asking for a mode that cannot run must change nothing at all.
//
// It needs no checkpoint. Everything asserted below is about the PLAN --
// which kernel a shape resolves to, and with which tiles -- and a plan
// is built from the config and the GPU, not from weights. That matters
// because the alternative gate is a 32 GB model, and a wiring test that
// only runs where the model is present is a wiring test that does not
// run.
//
// THE THREE THINGS THAT WOULD BE SILENT IF WRONG, and are therefore what
// is checked:
//
//   * the int8 twin conditioned on the SETTING instead of the hardware.
//     Plans are cached per pass shape and outlive a beat where the
//     setting does not, so a beat that turned the tier on after one that
//     had it off would find no twin and render bf16 with the log saying
//     otherwise.
//   * the tiles. The scales are one per the ATTENTION kernel's own
//     tiles, so a prologue sized against 32/16 while the kernel indexes
//     with 64/32 reads past its scale arrays.
//   * the KEY head stride. K lives in a cache whose per-head stride is
//     the CAPACITY, not the key length in use, and the quantizer has to
//     walk the bytes the kernel walks.

#include "u15-config.h"
#include "u15-metal-ops.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/sage-attention.h"

#include <cstdio>
#include <cstdlib>
#include <string>

extern "C" const unsigned char u15_kernels_bf16_metallib[];
extern "C" const unsigned long u15_kernels_bf16_metallib_len;

using vpipe::metal_compute::MetalCompute;

namespace {

int g_fail = 0;
int g_ran  = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
  ++g_ran;
}

// The 8B MoT's own attention shape: 32 query heads over 8 key heads --
// GQA group 4 -- at head_dim 128. The numbers matter here because the
// plan is what is being asserted.
constexpr int kHQ = 32, kHKV = 8, kHD = 128;
constexpr int kTQ = 1024, kTKV = 1024, kCAP = 4096;

vpipe::genai::sage::Config
cfg_of(bool on, int dense)
{
  vpipe::genai::sage::Config c;
  c.enabled = on;
  c.dense_layers = dense;
  return c;
}

}  // namespace

int
main()
{
  ::unsetenv("VPIPE_SAGE_ATTN");
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no usable Metal device. NOTHING was checked.\n");
    return 0;
  }
  if (!mc.register_metal_library(u15::kMetalLibBf16,
                                 u15_kernels_bf16_metallib,
                                 u15_kernels_bf16_metallib_len)) {
    std::printf("FAILED to register the metallib\n");
    return 1;
  }
  u15::MetalOps ops;
  std::string err;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED to init MetalOps: %s\n", err.c_str());
    return 1;
  }
  // The tier needs the MATRIX-CORE flash kernel, so it needs both the
  // hardware and this model's own NAX gate to have said yes. Reporting
  // them apart is what tells a reader which one declined.
  const bool cores = vpipe::genai::MetalSageAttention::available(&mc);
  std::printf("  matrix cores: %s, this model's attention on nax: %s\n",
              cores ? "yes" : "no", ops.attn_is_nax() ? "yes" : "no");
  const bool live = cores && ops.attn_is_nax();

  // ---- asking is never a failure ------------------------------------
  check(ops.set_sage(cfg_of(true, 0), &err),
        "asking for sage_attn is accepted -- a box with no matrix cores "
        "declines, and declining is not a failure");
  check(ops.sage_takes(kHD) == live,
        "and it is live exactly when this GPU and this model's own "
        "attention gate both allow it");
  check(!ops.sage_takes(96),
        "a head width steel has no entry point for has no int8 twin "
        "either");
  check(ops.sage_config().enabled == live,
        "a tier with no driver behind it reports itself OFF, so nothing "
        "downstream can be told yes by a config alone");

  // ---- the plan, which is what a beat actually dispatches -----------
  //
  // Built with the tier OFF on purpose. A plan is cached per shape and
  // outlives a beat; the setting does not. So the twin has to be there
  // even when nothing asked for it, or the second beat of a session that
  // turned Sage on would quietly run bf16.
  check(ops.set_sage(cfg_of(false, 0), &err), "the tier turns back off");
  u15::MetalOps::SteelAttn p;
  const bool planned =
      ops.steel_attn_plan(&p, kHQ, kHKV, kTQ, kTKV, kHD, kCAP);
  check(planned && p.valid(), "the bidirectional pass plans on steel");
  if (planned) {
    check(p.fn_i8.valid() == live,
          "...and it carries the int8 twin whenever the GPU could run "
          "it -- NOT only when the config asked, because this plan "
          "outlives the setting");
    // The three numbers the quantizer must agree with the kernel about.
    check(p.heads_kv == kHKV,
          "the plan remembers the KEY head count: K is quantized once "
          "per kv head, not once per query head");
    check(p.kv_stride == kCAP,
          "...and the cache CAPACITY, which is the K head stride and is "
          "not the key length in use");
    if (live) {
      check(p.bq == vpipe::genai::MetalSageAttention::nax_query_block() &&
                p.bk == vpipe::genai::MetalSageAttention::nax_key_block(),
            "and the plan's tiles are the ones the scales are sized by");
    }
  }

  if (!live) {
    std::printf("  SKIPPED on this GPU: whether the int8 twin computes "
                "the right thing needs matrix cores. What is checked "
                "above is that asking for it changed nothing.\n");
  }

  // ---- the override, both ways --------------------------------------
  {
    ::setenv("VPIPE_SAGE_ATTN", "0", 1);
    check(ops.set_sage(cfg_of(true, 0), &err) && !ops.sage_config().enabled,
          "VPIPE_SAGE_ATTN=0 takes the tier away from a graph that asked");
    ::setenv("VPIPE_SAGE_ATTN", "1", 1);
    check(ops.set_sage(cfg_of(false, 0), &err) &&
              ops.sage_config().enabled == live,
          "VPIPE_SAGE_ATTN=1 gives it to one that did not, where the GPU "
          "allows");
    ::unsetenv("VPIPE_SAGE_ATTN");
  }

  // ---- it composes with i8_gemm -------------------------------------
  //
  // Different halves of a layer: i8_gemm changes how a weight is
  // multiplied, Sage how a score is computed. Neither reads the other's
  // state, and the assertion is that turning one on does not disturb
  // what the other reports.
  {
    check(ops.set_sage(cfg_of(true, 0), &err), "sage_attn set");
    const bool sage_before = ops.sage_takes(kHD);
    ops.enable_i8_gemm(true);
    check(ops.sage_takes(kHD) == sage_before,
          "enabling i8_gemm leaves sage_attn exactly where it was");
    ops.enable_i8_gemm(false);
  }

  std::printf("u15-sage: %d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
