// End-to-end: prompt in, image out, on the full 42-layer checkpoint.
//
// What this can and cannot check. It CANNOT compare against the
// reference pixel-for-pixel: the initial noise comes from torch's
// Philox RNG, which is not reproduced here, so the same seed gives a
// different sample. (`init_noise` exists as an injection point for
// exactly that comparison; feeding it the reference's noise is the way
// to make an image-level check meaningful, and that is a separate,
// bigger piece of work.)
//
// What it DOES check is everything a broken assembly breaks: that the
// run completes, that every pixel is finite, that the image is not
// degenerate (a constant field, or pure noise), and that more steps
// give a different image -- i.e. that the loop is actually iterating.
//
// Gated on VPIPE_U15_TEST_MODEL_PATH. Loads ~35 GB, so it is slow.

#include "u15-backbone.h"
#include "u15-config.h"
#include "u15-generator.h"
#include "u15-image.h"
#include "u15-metal-ops.h"
#include "u15-prompt.h"
#include "u15-weights.h"
#include "u15-image-golden.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// A binary PPM, so the result can actually be looked at without adding
// an image-library dependency to a test.
void
write_ppm(const std::string& path, const std::vector<std::uint8_t>& planar,
          int h, int w)
{
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) { return; }
  std::fprintf(f, "P6\n%d %d\n255\n", w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < 3; ++c) {
        std::fputc(planar[((std::size_t)c * h + y) * w + x], f);
      }
    }
  }
  std::fclose(f);
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

  u15::U15Config cfg;
  std::string err;
  if (!u15::parse_config(p, &cfg, &err)) {
    std::printf("FAILED config: %s\n", err.c_str());
    return 1;
  }

  // 384x384 (12x12 = 144 image tokens), NOT 256.
  //
  // 256^2 is exactly the model's documented min_pixels (65536) and
  // exactly base_image_seq_len (64 tokens), and it generates BADLY
  // there -- a tiny subject on a mostly-empty canvas, reproducibly, at
  // 8 and at 30 steps. 384, 512 and 1024 all produce clean images
  // through the same code path, which is the argument that this is the
  // model's own lower boundary rather than a port defect: a resolution
  // bug does not switch off between 12x12 and 16x16 tokens while being
  // pixel-perfect at 32x32. Not proven against the reference
  // implementation, which would need a 36 GB torch run.
  u15::GenParams gp;
  gp.width = 384;
  gp.height = 384;
  gp.steps = 8;
  gp.cfg_scale = 4.0;
  gp.timestep_shift = 3.0;
  gp.seed = 42;
  if (const char* s = std::getenv("VPIPE_U15_TEST_STEPS")) {
    gp.steps = std::atoi(s);
  }
  if (const char* s = std::getenv("VPIPE_U15_TEST_SIZE")) {
    gp.width = gp.height = std::atoi(s);
  }

  std::printf("checkpoint: %s\n", p);
  std::printf("%dx%d, %d steps, cfg %.1f, shift %.1f\n\n", gp.width,
              gp.height, gp.steps, gp.cfg_scale, gp.timestep_shift);

  auto ws = WeightSet::open(p, nullptr);
  if (ws == nullptr) { std::printf("FAILED to open\n"); return 1; }

  const auto t0 = std::chrono::steady_clock::now();
  // STREAMED, because binding all 42 layers is 35 GB and this test has
  // to run on a box that does not have it. On a 16 GB machine the
  // preloading form swapped ~20 GB and took 86 s/step; streamed it is
  // the same arithmetic (u15-stream-test holds the streamed render
  // bit-identical to preloaded) at a ~4 GB floor.
  //
  // VPIPE_U15_TEST_PRELOAD=1 restores the old behaviour for a box with
  // the RAM to spare.
  u15::U15Weights::Options opt;          // all 42 layers, no lm_head
  opt.stream_layers = std::getenv("VPIPE_U15_TEST_PRELOAD") == nullptr;
  auto w = u15::U15Weights::load(ws, &mc, cfg, opt, &err);
  const double load_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  check(w != nullptr, "all 42 layers bound" +
                          (err.empty() ? "" : " (" + err + ")"));
  if (w == nullptr) { return 1; }
  std::printf("       %.1f s, %.2f GB resident (%.2f GB converted from "
              "F32)\n", load_s,
              (double)w->resident_bytes() / (1024.0 * 1024 * 1024),
              (double)w->converted_bytes() / (1024.0 * 1024 * 1024));

  u15::MetalOps ops;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED MetalOps: %s\n", err.c_str());
    return 1;
  }
  auto bb = u15::U15Backbone::create(&ops, cfg, w.get(), &err);
  check(bb != nullptr, "backbone");
  auto ip = u15::ImagePath::create(&ops, cfg, w.get(), &err);
  check(ip != nullptr, "image path" + (err.empty() ? "" : " (" + err + ")"));
  auto pr = u15::Prompt::load(p, nullptr, &err);
  check(pr != nullptr, "tokenizer" + (err.empty() ? "" : " (" + err + ")"));
  if (bb == nullptr || ip == nullptr || pr == nullptr) { return 1; }

  u15::Generator::Deps d;
  d.ops = &ops; d.weights = w.get(); d.backbone = bb.get();
  d.image = ip.get(); d.prompt = pr.get();
  auto gen = u15::Generator::create(d, cfg, &err);
  check(gen != nullptr, "generator");
  if (gen == nullptr) { return 1; }

  std::string prompt_text = "a red fox sitting in snow, photorealistic";
  if (const char* s = std::getenv("VPIPE_U15_TEST_PROMPT")) {
    prompt_text = s;
  }
  std::vector<std::uint8_t> img;
  const auto g0 = std::chrono::steady_clock::now();
  const std::vector<u15::RefImage> no_refs;   // text-to-image
  const bool ok = gen->generate(
      prompt_text, no_refs, gp, &img,
      [](int s, int n) {
        std::printf("\r       step %d/%d ", s, n);
        std::fflush(stdout);
      },
      &err);
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - g0).count();
  std::printf("\r                    \r");
  check(ok, "generate" + (err.empty() ? "" : " (" + err + ")"));
  if (!ok) { return 1; }
  std::printf("       %.1f s (%.2f s/step)\n", gen_s,
              gen_s / std::max(1, gp.steps));

  check(img.size() == (std::size_t)gp.width * gp.height * 3,
        "output is [3][H][W] u8");

  // ---- the image is not degenerate ---------------------------------
  double mean = 0.0, mn = 255.0, mx = 0.0;
  for (std::uint8_t v : img) {
    mean += v;
    mn = std::min(mn, (double)v);
    mx = std::max(mx, (double)v);
  }
  mean /= (double)img.size();
  double var = 0.0;
  for (std::uint8_t v : img) {
    var += ((double)v - mean) * ((double)v - mean);
  }
  var /= (double)img.size();
  const double sd = std::sqrt(var);
  std::printf("       pixels: mean %.1f, sd %.1f, range [%.0f, %.0f]\n",
              mean, sd, mn, mx);

  check(mx > mn, "the image is not a constant field");
  // sd > 25 is the bar that MATTERS, and it is set from measurement.
  // When the working image was fed to the patch embedder with the wrong
  // channel layout, every check here still passed: the result was a
  // flat grey field at sd 13.0, which is neither constant nor
  // saturated. A correct 20-step render measures sd ~72. A bar of 2 --
  // which is what this said first -- certifies garbage.
  check(sd > 25.0,
        "the image has real contrast (sd > 25; a collapsed denoise "
        "gives ~13)");
  check(sd < 100.0, "the image is not saturated noise");

  // ---- the denoise actually moved it -------------------------------
  //
  // One step from the same seed, compared against the full run. If the
  // loop were not feeding the head's output back -- or were iterating
  // on a stale buffer -- these would be identical.
  {
    u15::GenParams one = gp;
    one.steps = 1;
    std::vector<std::uint8_t> img1;
    if (gen->generate(prompt_text, no_refs, one, &img1, nullptr,
                      &err)) {
      std::size_t same = 0;
      for (std::size_t i = 0; i < img.size(); ++i) {
        if (img[i] == img1[i]) { ++same; }
      }
      const double frac = (double)same / (double)img.size();
      std::printf("       1-step vs %d-step: %.1f%% of pixels identical\n",
                  gp.steps, frac * 100.0);
      check(frac < 0.9,
            "more steps give a DIFFERENT image (the loop is iterating)");
    }
  }

  // Does the PROMPT reach the image tokens at all? If the prefix KV
  // were not being attended, every prompt would give the same picture --
  // and a flat, plausible-looking field passes every statistical check
  // above. This is the one that catches it.
  {
    std::vector<std::uint8_t> other;
    if (gen->generate("a solid black square on a white background",
                      no_refs, gp, &other, nullptr, &err)) {
      double d = 0.0;
      for (std::size_t i = 0; i < img.size(); ++i) {
        d += std::fabs((double)img[i] - (double)other[i]);
      }
      d /= (double)img.size();
      std::printf("       two different prompts differ by %.2f mean abs "
                  "(u8)\n", d);
      // This metric is NOT scale-free -- it grows with resolution and
      // step count, because both give the conditioning more chance to
      // act. MEASURED: ~104 at 512^2/20 steps, ~38 at 256^2/8, and ~49
      // in the broken case where both prompts collapsed to a similar
      // flat field. So there is no single threshold that separates
      // working from broken across configurations, and a bar set from
      // one config (60, from the 512^2 run) FAILS a healthy 256^2 one.
      //
      // The structural guard is the sd check above -- it read 72 when
      // correct and 13 when collapsed, at BOTH sizes. This check stays
      // as an independent signal that the prefix is reaching the image
      // tokens at all, with a bar low enough to hold at the smallest
      // configuration the test runs.
      check(d > 15.0, "the PROMPT changes the image (conditioning "
                      "reaches the image tokens)");
    }
  }

  // ---- against a known-good render -------------------------------
  //
  // The check with teeth. Everything above is satisfied by a textured
  // field; this one is not. See u15-image-golden.h for why it is a
  // 24x24 luminance thumbnail and a correlation rather than pixels.
  if (gp.width == u15_golden::kWidth && gp.height == u15_golden::kHeight &&
      gp.steps == u15_golden::kSteps && gp.seed == u15_golden::kSeed &&
      prompt_text == "a red fox sitting in snow, photorealistic") {
    constexpr int N = u15_golden::kSize;
    double mine[N * N] = {0.0};
    for (int ty = 0; ty < N; ++ty) {
      for (int tx = 0; tx < N; ++tx) {
        const int x0 = tx * gp.width / N, x1 = (tx + 1) * gp.width / N;
        const int y0 = ty * gp.height / N, y1 = (ty + 1) * gp.height / N;
        double sum = 0.0;
        int n = 0;
        for (int y = y0; y < y1; ++y) {
          for (int x = x0; x < x1; ++x) {
            // The image is [3][H][W] planar, which is what makes this
            // three strided reads rather than one.
            const std::size_t px = (std::size_t)y * gp.width + x;
            const std::size_t plane = (std::size_t)gp.height * gp.width;
            sum += 0.299 * img[px] + 0.587 * img[plane + px] +
                   0.114 * img[2 * plane + px];
            ++n;
          }
        }
        mine[ty * N + tx] = n > 0 ? sum / n : 0.0;
      }
    }
    double ma = 0.0, mb = 0.0;
    for (int i = 0; i < N * N; ++i) {
      ma += mine[i];
      mb += u15_golden::kThumb[i];
    }
    ma /= N * N;
    mb /= N * N;
    double num = 0.0, va = 0.0, vb = 0.0;
    for (int i = 0; i < N * N; ++i) {
      const double a = mine[i] - ma;
      const double b = (double)u15_golden::kThumb[i] - mb;
      num += a * b;
      va += a * a;
      vb += b * b;
    }
    const double r = (va > 0.0 && vb > 0.0)
                         ? num / std::sqrt(va * vb) : 0.0;
    std::printf("       vs the golden render: correlation %.4f "
                "(bar %.2f)\n", r, u15_golden::kMinCorrelation);
    check(r >= u15_golden::kMinCorrelation,
          "the image matches the known-good render (this is the check "
          "that tells a picture from a textured field)");
  } else {
    // SAID, not silent. A skipped golden is the difference between
    // "this render is right" and "nothing looked".
    std::printf("       golden SKIPPED: the generator settings were "
                "overridden, so the reference describes a different "
                "image\n");
  }

  const char* out = std::getenv("VPIPE_U15_TEST_OUT");
  if (out != nullptr && *out != '\0') {
    write_ppm(out, img, gp.height, gp.width);
    std::printf("       wrote %s\n", out);
  }

  std::printf("\n%d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
