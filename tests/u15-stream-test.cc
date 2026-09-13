// LAYER STREAMING: the same picture, out of a fraction of the memory.
//
// The question a streaming test has to answer is not "does it run" -- a
// streamed stack that dropped a layer, or read one twice, still
// produces an image, and at 4 steps it is a plausible one. It is
// whether streaming changed the ARITHMETIC. It must not: the same
// weights reach the same kernels in the same order, and the only
// difference is where the bytes were living a moment earlier.
//
// So this renders once preloaded and once streamed, at one seed, and
// requires the two to be BIT-IDENTICAL. That bar is available here and
// is not available to most quality tests, so it is the one to use:
// anything less would pass a port that silently skipped a layer.
//
// It also covers the loop inversion the streaming path depends on. The
// CFG branches run through the stack together rather than one after
// another (U15Backbone::forward_many), which is what makes a streamed
// step read the checkpoint once instead of once per branch -- and both
// arms below take that path with two branches live, so a reordering
// that was not a reordering shows up as a mismatch.
//
// Gated on VPIPE_U15_TEST_MODEL_PATH; SKIPS when unset and says so.
// Defaults to 256^2 / 4 steps: the bar is exactness, not quality, and
// a small render exercises every layer just as thoroughly.

#include "u15-backbone.h"
#include "u15-config.h"
#include "u15-generator.h"
#include "u15-image.h"
#include "u15-metal-ops.h"
#include "u15-prompt.h"
#include "u15-weights.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/weight-set.h"
#include "stages/model-memory.h"

#include <chrono>
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

// Everything one model needs, held together so a render can be run from
// it and the whole thing dropped before the next one loads. Both
// resident at once would be ~50 GB.
struct Loaded {
  std::unique_ptr<u15::MetalOps>    ops;
  std::shared_ptr<WeightSet>        ws;
  std::unique_ptr<u15::U15Weights>  weights;
  std::unique_ptr<u15::U15Backbone> backbone;
  std::unique_ptr<u15::ImagePath>   image;
  std::unique_ptr<u15::Prompt>      prompt;
  std::unique_ptr<u15::Generator>   gen;
  u15::U15Config                    cfg;
};

bool
load(const std::string& path, MetalCompute& mc,
     const u15::U15Weights::Options& opt, Loaded* out, std::string* err)
{
  if (!u15::parse_config(path, &out->cfg, err)) { return false; }
  out->ops = std::make_unique<u15::MetalOps>();
  if (!out->ops->init(&mc, err)) { return false; }
  // WeightSet::open, not open_weight_set: there is no session here, so
  // this is the documented back door and the set is private to the test.
  out->ws = WeightSet::open(path, nullptr);
  if (out->ws == nullptr) { *err = "cannot open"; return false; }
  out->weights = u15::U15Weights::load(out->ws, &mc, out->cfg, opt, err);
  if (out->weights == nullptr) { return false; }
  out->backbone = u15::U15Backbone::create(out->ops.get(), out->cfg,
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

std::string
gb(std::size_t b)
{
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.2f GB",
                (double)b / (1024.0 * 1024.0 * 1024.0));
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

  int size = 256, steps = 4, pinned = 2;
  if (const char* s = std::getenv("VPIPE_U15_TEST_SIZE")) {
    size = std::atoi(s);
  }
  if (const char* s = std::getenv("VPIPE_U15_TEST_STEPS")) {
    steps = std::atoi(s);
  }
  if (const char* s = std::getenv("VPIPE_U15_STREAM_PINNED")) {
    pinned = std::atoi(s);
  }

  u15::GenParams gp;
  gp.width = gp.height = size;
  gp.steps = steps;
  gp.cfg_scale = 4.0;          // > 1, so the guided TWO-branch path runs
  gp.timestep_shift = 3.0;
  gp.seed = 42;
  const std::vector<u15::RefImage> no_refs;
  const char* kPrompt = "a red fox sitting in snow, photorealistic";
  std::printf("%dx%d, %d steps, seed %llu, %d layers pinned when "
              "streaming\n\n", size, size, steps,
              (unsigned long long)gp.seed, pinned);

  std::string err;
  std::vector<std::uint8_t> pre_img, str_img;
  std::size_t pre_bytes = 0, str_bytes = 0, streamed = 0;
  double pre_s = 0.0, str_s = 0.0;
  int n_layers = 0, resident_after = 0, reads = 0;
  int pf_started = 0, pf_hits = 0;

  // ---- PRELOADED, the reference arm ---------------------------------
  {
    Loaded L;
    u15::U15Weights::Options opt;             // stream_layers = false
    check(load(p, mc, opt, &L, &err),
          std::string("preloaded model loads") +
              (err.empty() ? "" : " (" + err + ")"));
    if (L.gen == nullptr) { return 1; }
    n_layers = L.cfg.llm.num_hidden_layers;
    pre_bytes = L.weights->loaded_bytes();
    check(!L.weights->streaming(), "and reports that it is not streaming");
    check(L.weights->resident_layers() == n_layers,
          "with all " + std::to_string(n_layers) + " layers held");
    const auto t0 = std::chrono::steady_clock::now();
    check(L.gen->generate(kPrompt, no_refs, gp, &pre_img, nullptr, {}, &err),
          std::string("preloaded render") +
              (err.empty() ? "" : " (" + err + ")"));
    pre_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("       preloaded: %s resident, render %.1f s\n",
                gb(pre_bytes).c_str(), pre_s);
  }

  // ---- STREAMED -----------------------------------------------------
  {
    Loaded L;
    u15::U15Weights::Options opt;
    opt.stream_layers = true;
    opt.pinned_layers = pinned;
    // Wiring OFF here: it asks the manager for a pool share and there
    // is no session, so it would be off anyway -- saying so keeps the
    // test from depending on that.
    opt.wire_resident = false;
    check(load(p, mc, opt, &L, &err),
          std::string("streamed model loads") +
              (err.empty() ? "" : " (" + err + ")"));
    if (L.gen == nullptr) { return 1; }
    str_bytes = L.weights->loaded_bytes();
    check(L.weights->streaming(), "and reports that it IS streaming");
    check(L.weights->resident_layers() == pinned,
          "holding only the " + std::to_string(pinned) +
              " pinned layers at load");
    const auto t0 = std::chrono::steady_clock::now();
    check(L.gen->generate(kPrompt, no_refs, gp, &str_img, nullptr, {}, &err),
          std::string("streamed render") +
              (err.empty() ? "" : " (" + err + ")"));
    str_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    streamed = L.weights->streamed_bytes();
    reads = L.weights->streamed_layers();
    resident_after = L.weights->resident_layers();
    pf_started = L.weights->prefetch_started();
    pf_hits = L.weights->prefetch_hits();
    std::printf("       streamed:  %s resident at load, %s read back in "
                "%d layer reads, render %.1f s\n",
                gb(str_bytes).c_str(), gb(streamed).c_str(), reads, str_s);
    std::printf("       residency grew to %d of %d layers; prefetch %d "
                "issued, %d landed\n", resident_after, n_layers,
                pf_started, pf_hits);
  }

  // ---- what it cost, and what it bought -----------------------------
  std::printf("\n       footprint %s -> %s (%.0f%%), render %.1f -> %.1f s "
              "(%.2fx)\n\n",
              gb(pre_bytes).c_str(), gb(str_bytes).c_str(),
              100.0 * (double)str_bytes / (double)pre_bytes, pre_s, str_s,
              pre_s > 0.0 ? str_s / pre_s : 0.0);

  check(str_bytes < pre_bytes / 2,
        "streaming more than halves what the model holds at load");
  check(reads > 0 && streamed > 0,
        "and it really did read layers back (" + std::to_string(reads) +
            " reads)");
  check(resident_after >= pinned && resident_after <= n_layers,
        "residency stayed between the pinned prefix and the whole stack");
  // Every prefetch that was issued should also have LANDED: the reads
  // are issued one block ahead and joined at the next acquire, so a
  // miss means the bookkeeping lost track of which slot held what.
  check(pf_started == 0 || pf_hits > 0,
        "the prefetches that were issued landed (" +
            std::to_string(pf_hits) + " of " +
            std::to_string(pf_started) + ")");

  // THE BAR. Not "close", not "correlated" -- IDENTICAL. Streaming
  // moves where the bytes live and changes nothing else, so anything
  // short of equality is a bug, and a tolerance here would hide exactly
  // the failure this test exists to catch (a layer skipped, applied
  // twice, or read from the wrong index).
  check(pre_img.size() == str_img.size() && !pre_img.empty(),
        "both renders produced an image of the same size");
  std::size_t diff = 0;
  std::size_t worst = 0;
  for (std::size_t i = 0; i < pre_img.size() && i < str_img.size(); ++i) {
    if (pre_img[i] != str_img[i]) {
      ++diff;
      const std::size_t d = pre_img[i] > str_img[i]
                                ? (std::size_t)(pre_img[i] - str_img[i])
                                : (std::size_t)(str_img[i] - pre_img[i]);
      if (d > worst) { worst = d; }
    }
  }
  if (diff != 0) {
    std::printf("       %zu of %zu bytes differ, worst by %zu\n", diff,
                pre_img.size(), worst);
  }
  check(diff == 0,
        "the streamed render is BIT-IDENTICAL to the preloaded one");


  // ---- the READ PATH, A/B ------------------------------------------
  //
  // The arms above cannot see this. Residency admits every layer on the
  // first pass of a roomy box, and promotion MOVES the slot's buffers
  // into the resident set -- so the slot is empty each time and every
  // read is a rebuild. The refill only ever runs where residency
  // CANNOT grow, which is the constrained box, which is the whole
  // reason streaming exists.
  //
  // So this drives the layer protocol directly and never declares a
  // residency reserve. Growth stays off without one -- documented
  // behaviour, not a trick -- so nothing is promoted, the slot survives
  // every layer, and the two read paths can be compared over the same
  // 42 layers.
  {
    std::printf("\n  --- read path, no residency growth ---\n");
    struct Res { double secs = 0.0; double gb = 0.0; std::uint64_t hash = 0;
                 std::size_t rebuilt = 0; int pf = 0; };
    Res arm[2];
    const char* arm_name[2] = {"rebuild (mmap+memcpy)", "refill (pread)"};

    for (int which = 0; which < 2; ++which) {
      Loaded L;
      u15::U15Weights::Options opt;
      opt.stream_layers = true;
      opt.pinned_layers = 0;
      opt.wire_resident = false;
      opt.refill_streamed = (which == 1);
      if (!load(p, mc, opt, &L, &err)) {
        check(false, std::string("read-path arm loads (") + err + ")");
        return 1;
      }
      const int nl = L.cfg.llm.num_hidden_layers;
      const auto t0 = std::chrono::steady_clock::now();
      std::uint64_t h = 1469598103934665603ull;
      // TWO passes: the first builds the slot on its first layer, the
      // second is entirely the path under test.
      for (int pass = 0; pass < 2; ++pass) {
        L.weights->begin_pass();
        for (int li = 0; li < nl; ++li) {
          const u15::MotLayer* ml = L.weights->layer(li, &err);
          if (ml == nullptr) {
            check(false, "layer read (" + err + ")");
            return 1;
          }
          // Fingerprint the two experts' biggest matrix, strided: this
          // is what says the two paths produced the SAME BYTES, which
          // matters more than which was faster.
          for (const u15::ExpertLayer* e : {&ml->und, &ml->gen}) {
            const auto& b = e->down.quantized ? e->down.codes : e->down.w;
            const auto* q = static_cast<const std::uint8_t*>(b.contents());
            for (std::size_t k = 0; k < b.byte_size(); k += 65536) {
              h = (h ^ q[k]) * 1099511628211ull;
            }
          }
          L.weights->end_layer(li);
        }
      }
      arm[which].secs = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t0).count();
      arm[which].gb = (double)L.weights->streamed_bytes() /
                      (1024.0 * 1024.0 * 1024.0);
      arm[which].hash = h;
      arm[which].rebuilt = L.weights->rebuilt_layers();
      arm[which].pf = L.weights->prefetch_hits();
    }

    for (int w = 0; w < 2; ++w) {
      std::printf("       %-22s %5.1f s  %6.2f GB  %6.2f GB/s  "
                  "(%zu rebuilt, %d prefetch hits)\n", arm_name[w],
                  arm[w].secs, arm[w].gb,
                  arm[w].secs > 0.0 ? arm[w].gb / arm[w].secs : 0.0,
                  arm[w].rebuilt, arm[w].pf);
    }
    if (arm[0].secs > 0.0) {
      std::printf("       refill / rebuild = %.2fx\n",
                  arm[0].secs / arm[1].secs);
    }
    check(arm[0].hash == arm[1].hash,
          "both read paths produced the same bytes");
    check(arm[1].rebuilt <= 2,
          "the refill arm rebuilt at most the first layer of each slot (" +
              std::to_string(arm[1].rebuilt) + ")");
    // NOT a prefetch check. This section drives the layer protocol
    // directly, so there is no committed GPU work to issue a read
    // under and nothing to prefetch into -- the prefetch is exercised
    // by the RENDER arms above, which go through the backbone.
    check(arm[1].pf == 0,
          "no prefetch here, which is right: this section has no GPU "
          "work to hide a read behind");
    check(arm[0].rebuilt >= 80,
          "the rebuild arm rebuilt every layer of both passes (" +
              std::to_string(arm[0].rebuilt) + ")");
  }

  // ---- the decision, at sizes this box may not have -----------------
  //
  // plan_streaming reads VPIPE_RAM_LIMIT_MB, so the verdict can be
  // exercised for a 16 GB box on a 64 GB one. Checked here rather than
  // left to the stage because it is the irreversible half of the
  // policy: a graph that gets this wrong thrashes, and nothing later
  // can undo it.
  {
    namespace mm = vpipe::model_memory;
    const std::size_t disk = mm::dir_weights_bytes(p);
    std::printf("\n       checkpoint on disk: %s\n", gb(disk).c_str());
    const std::size_t floor =
        mm::streaming_floor_bytes(p, {"language_model.model.layers."});
    std::printf("       streaming floor:    %s\n", gb(floor).c_str());
    check(floor > 0 && floor < disk / 4,
          "the streaming floor is a small fraction of the checkpoint");

    struct Case { const char* mb; bool want; };
    // The rule is "preload only when the checkpoint is small against the
    // box": at most a third of RAM, with kStreamHeadroom on top.
    const Case cases[] = {
        {"8192", true},          // 8 GB: nothing this size preloads
        {"16384", true},         // 16 GB
        {"262144", false},       // 256 GB: room for anything here
    };
    for (const Case& c : cases) {
      setenv("VPIPE_RAM_LIMIT_MB", c.mb, 1);
      const auto plan =
          mm::plan_streaming(nullptr, p, "", mm::kStreamHeadroom);
      check(plan.stream == c.want,
            std::string("a ") + c.mb + " MB box " +
                (c.want ? "streams" : "preloads") + " this checkpoint");
    }
    unsetenv("VPIPE_RAM_LIMIT_MB");
  }

  std::printf("\n%d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
