// A QUANTIZED pack, against the dense one it was made from.
//
// The question a quantization test has to answer is not "does it load"
// -- an affine pack whose bits or group were read wrong loads fine and
// generates something plausible. It is whether the SAME prompt and seed
// through both packs produce the same picture.
//
// So this renders once from each and compares. The bar is set from
// measurement: w8g64 lands at ~1.8 u8 mean difference and 0.999
// correlation against bf16, which is visually indistinguishable; a
// mis-read packing is not close to that.
//
// Needs BOTH paths:
//   VPIPE_U15_TEST_MODEL_PATH   the dense checkpoint
//   VPIPE_U15_QUANT_MODEL_PATH  a pack from model-quantize
// Either unset => SKIPS, and says so.

#include "u15-backbone.h"
#include "u15-config.h"
#include "u15-generator.h"
#include "u15-image.h"
#include "u15-metal-ops.h"
#include "u15-prompt.h"
#include "u15-weights.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/weight-set.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;

extern "C" const unsigned char u15_kernels_bf16_metallib[];
extern "C" const unsigned long u15_kernels_bf16_metallib_len;

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

// Everything one checkpoint needs, held together so a render can be run
// from it and then the whole 17-35 GB dropped before the next one loads.
// Both resident at once would be ~50 GB, which this box can just about
// do and a 32 GB one cannot.
struct Loaded {
  MetalCompute*                   mc = nullptr;
  std::unique_ptr<u15::MetalOps>  ops;
  std::shared_ptr<WeightSet>      ws;
  std::unique_ptr<u15::U15Weights> weights;
  std::unique_ptr<u15::U15Backbone> backbone;
  std::unique_ptr<u15::ImagePath>  image;
  std::unique_ptr<u15::Prompt>     prompt;
  std::unique_ptr<u15::Generator>  gen;
  u15::U15Config                   cfg;
};

bool
load(const std::string& path, MetalCompute& mc, Loaded* out, std::string* err)
{
  out->mc = &mc;
  if (!u15::parse_config(path, &out->cfg, err)) { return false; }
  out->ops = std::make_unique<u15::MetalOps>();
  if (!out->ops->init(&mc, err)) { return false; }
  out->ws = WeightSet::open(path, nullptr);
  if (out->ws == nullptr) { *err = "cannot open"; return false; }
  u15::U15Weights::Options opt;
  out->weights = u15::U15Weights::load(out->ws, &mc, out->cfg, opt, err);
  if (out->weights == nullptr) { return false; }
  out->backbone =
      u15::U15Backbone::create(out->ops.get(), out->cfg,
                               out->weights.get(), err);
  out->image = u15::ImagePath::create(out->ops.get(), out->cfg,
                                      out->weights.get(), err);
  out->prompt = u15::Prompt::load(path, nullptr, err);
  if (out->backbone == nullptr || out->image == nullptr ||
      out->prompt == nullptr) {
    return false;
  }
  u15::Generator::Deps d;
  d.ops = out->ops.get();
  d.weights = out->weights.get();
  d.backbone = out->backbone.get();
  d.image = out->image.get();
  d.prompt = out->prompt.get();
  out->gen = u15::Generator::create(d, out->cfg, err);
  return out->gen != nullptr;
}

double
mean_abs(const std::vector<std::uint8_t>& a,
         const std::vector<std::uint8_t>& b)
{
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    d += std::fabs((double)a[i] - (double)b[i]);
  }
  return d / (double)a.size();
}

double
correlation(const std::vector<std::uint8_t>& a,
            const std::vector<std::uint8_t>& b)
{
  double ma = 0.0, mb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) { ma += a[i]; mb += b[i]; }
  ma /= (double)a.size();
  mb /= (double)b.size();
  double num = 0.0, va = 0.0, vb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double da = (double)a[i] - ma, db = (double)b[i] - mb;
    num += da * db;
    va += da * da;
    vb += db * db;
  }
  const double den = std::sqrt(va) * std::sqrt(vb);
  return den > 0.0 ? num / den : 0.0;
}

}  // namespace

int
main()
{
  const char* dense_p = std::getenv("VPIPE_U15_TEST_MODEL_PATH");
  const char* quant_p = std::getenv("VPIPE_U15_QUANT_MODEL_PATH");
  if (dense_p == nullptr || *dense_p == '\0' || quant_p == nullptr ||
      *quant_p == '\0') {
    std::printf("SKIPPED: VPIPE_U15_TEST_MODEL_PATH / "
                "VPIPE_U15_QUANT_MODEL_PATH unset -- this test did NOT "
                "run.\n");
    return 0;
  }
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

  int size = 384, steps = 8;
  if (const char* s = std::getenv("VPIPE_U15_TEST_SIZE")) {
    size = std::atoi(s);
  }
  if (const char* s = std::getenv("VPIPE_U15_TEST_STEPS")) {
    steps = std::atoi(s);
  }
  u15::GenParams gp;
  gp.width = gp.height = size;
  gp.steps = steps;
  gp.cfg_scale = 4.0;
  gp.timestep_shift = 3.0;
  gp.seed = 42;
  const std::vector<u15::RefImage> no_refs;
  const char* kPrompt = "a red fox sitting in snow, photorealistic";
  std::printf("%dx%d, %d steps, seed %llu\n\n", size, size, steps,
              (unsigned long long)gp.seed);

  std::string err;
  std::vector<std::uint8_t> dense_img, quant_img;
  std::size_t dense_bytes = 0, quant_bytes = 0;
  double dense_s = 0.0, quant_s = 0.0;
  int bits = 0, group = 0;

  // ---- the DENSE render, then dropped -------------------------------
  {
    Loaded L;
    const auto t0 = std::chrono::steady_clock::now();
    check(load(dense_p, mc, &L, &err),
          std::string("dense pack loads") +
              (err.empty() ? "" : " (" + err + ")"));
    if (L.gen == nullptr) { return 1; }
    dense_bytes = L.weights->resident_bytes();
    check(L.weights->quant_bits() == 0,
          "the dense pack reports no quantization");
    const auto t1 = std::chrono::steady_clock::now();
    check(L.gen->generate(kPrompt, no_refs, gp, &dense_img, nullptr, {}, &err),
          "dense render");
    dense_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t1).count();
    std::printf("       dense: %.2f GB, load %.1f s, render %.1f s\n",
                (double)dense_bytes / (1024.0 * 1024 * 1024),
                std::chrono::duration<double>(t1 - t0).count(), dense_s);
  }

  // ---- the QUANTIZED render -----------------------------------------
  {
    Loaded L;
    const auto t0 = std::chrono::steady_clock::now();
    check(load(quant_p, mc, &L, &err),
          std::string("quantized pack loads") +
              (err.empty() ? "" : " (" + err + ")"));
    if (L.gen == nullptr) { return 1; }
    quant_bytes = L.weights->resident_bytes();
    bits = L.weights->quant_bits();
    group = L.weights->quant_group();
    check(bits == 4 || bits == 8,
          "the pack reports a packing: w" + std::to_string(bits) + "g" +
              std::to_string(group));
    check(L.ops->quant_available(),
          "the host ships the affine qmm kernels");
    const auto t1 = std::chrono::steady_clock::now();
    check(L.gen->generate(kPrompt, no_refs, gp, &quant_img, nullptr, {}, &err),
          "quantized render");
    quant_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t1).count();
    std::printf("       w%dg%d: %.2f GB, load %.1f s, render %.1f s\n",
                bits, group,
                (double)quant_bytes / (1024.0 * 1024 * 1024),
                std::chrono::duration<double>(t1 - t0).count(), quant_s);
  }

  // ---- the comparison ------------------------------------------------
  const double d = mean_abs(dense_img, quant_img);
  const double c = correlation(dense_img, quant_img);
  std::printf("\n       mean|dense - w%dg%d| = %.2f u8, correlation "
              "%.5f\n", bits, group, d, c);
  std::printf("       footprint %.2f -> %.2f GB (%.0f%%), render %.1f -> "
              "%.1f s\n\n",
              (double)dense_bytes / (1024.0 * 1024 * 1024),
              (double)quant_bytes / (1024.0 * 1024 * 1024),
              100.0 * (double)quant_bytes / (double)dense_bytes,
              dense_s, quant_s);

  check(quant_bytes < dense_bytes, "the quantized pack is smaller");

  // The bars, from measurement AT THIS TEST'S OWN DEFAULT (384^2, 8
  // steps). The metric tightens sharply with step count, because a small
  // per-step perturbation has fewer steps to be corrected in:
  //
  //   w8g64 @ 512^2/20 steps   1.78 u8   0.99941
  //   w8g64 @ 384^2/8  steps   4.60 u8   0.98695
  //
  // A bar taken from the first and applied to the second FAILS a healthy
  // pack -- which is exactly what happened when these were first
  // written. What the bar has to separate is a good pack from a pack
  // whose bits or group were read wrong, and that one produces noise or
  // a collapse, i.e. correlation near zero. So there is a lot of room
  // between "healthy" and "broken", and the bar sits in it rather than
  // hugging one measurement.
  const double d_bar = bits == 8 ? 10.0 : 25.0;
  const double c_bar = bits == 8 ? 0.95 : 0.85;
  check(d < d_bar,
        "the quantized render is close to the dense one (mean|diff| " +
            std::to_string(d) + " < " + std::to_string(d_bar) + ")");
  check(c > c_bar,
        "and correlates with it (" + std::to_string(c) + " > " +
            std::to_string(c_bar) + ")");

  // A pack that produced the SAME bytes would mean the quantized weights
  // were never used -- e.g. a loader that silently fell back to dense.
  check(d > 0.0, "the two renders are not byte-identical (the quantized "
                 "weights are actually being used)");

  std::printf("\n%d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
