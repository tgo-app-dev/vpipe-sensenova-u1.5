// HOW TO GET A TENSOR OFF DISK: mmap+memcpy against pread.
//
// The streaming path reads a layer per pass, so which primitive it uses
// decides what streaming costs. vpipe offers two and they are not
// equivalent:
//
//   MetalLlamaWeights::load()        allocate a SharedBuffer, memcpy
//                                    into it from the shard's MMAP.
//                                    What WeightSet::read/tensor/
//                                    stream_tensor all do, and what this
//                                    plugin's streaming path does today.
//   MetalLlamaWeights::pread_into()  pread(2) straight into a buffer the
//                                    caller already owns. What
//                                    streamed-refill.h is built on.
//                                    F_NOCACHE by DEFAULT, on its own fd.
//
// The claim under test is that mmap loses, and for a structural reason
// rather than a tuning one: the kernel tracks page residency at 4 KB
// granularity over files of tens of gigabytes, and that bookkeeping
// costs more than a kernel->user copy saves. If so, a streaming model
// should never reach for the mmap path.
//
// THREE THINGS THIS HAS TO SEPARATE, or the answer is an artifact:
//
//   the DEVICE      an internal SSD and a Thunderbolt external differ
//                   ~5x in raw pread. Pass both.
//   the PAGE CACHE  pread_into defaults to F_NOCACHE and mmap cannot,
//                   so a warm file flatters mmap for a reason that has
//                   nothing to do with the mechanism. The `pread cached`
//                   arm holds cache state equal so the mechanism shows
//                   on its own.
//   the ALLOCATION  load() allocates per call and pread_into does not.
//                   That IS part of the difference the design would
//                   capture, so it is measured -- and reported apart,
//                   via LoadCost, so it cannot be mistaken for I/O.
//
// Arms are INTERLEAVED and their tensor assignment ROTATES each round,
// so neither thermal drift nor one arm warming the cache for the next
// can be read as a result. Reports rates; asserts only that the arms
// return the same BYTES, because absolute speed is a property of the
// box and this is here to inform a decision, not to gate a build.
//
//   VPIPE_U15_IO_BENCH  colon-separated checkpoint dirs to test.
//                       Unset => uses VPIPE_U15_TEST_MODEL_PATH.
//   VPIPE_U15_IO_ROUNDS how many rotations (default 3).

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using vpipe::genai::MetalLlamaWeights;
using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

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

double
now_s()
{
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Arm {
  const char* name;
  double      bytes = 0.0;
  double      secs  = 0.0;
  double      alloc_ms = 0.0;     // load() only
  double gbps() const
  {
    return secs > 0.0 ? bytes / secs / (1024.0 * 1024.0 * 1024.0) : 0.0;
  }
};

// A cheap fingerprint, so "did the arms read the same thing" is a real
// check and not an assumption. Strided: reading every byte of 8 GB to
// verify a benchmark would cost more than the benchmark.
std::uint64_t
fingerprint(const void* p, std::size_t n)
{
  const auto* b = static_cast<const std::uint8_t*>(p);
  std::uint64_t h = 1469598103934665603ull;
  for (std::size_t i = 0; i < n; i += 4096) {
    h = (h ^ b[i]) * 1099511628211ull;
  }
  h = (h ^ b[n - 1]) * 1099511628211ull;
  return h;
}

std::vector<std::string>
split_paths(const std::string& s)
{
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i <= s.size()) {
    const std::size_t j = s.find(':', i);
    const std::string one =
        s.substr(i, j == std::string::npos ? std::string::npos : j - i);
    if (!one.empty()) { out.push_back(one); }
    if (j == std::string::npos) { break; }
    i = j + 1;
  }
  return out;
}

// The biggest tensors in the checkpoint, which is what a layer read is
// made of -- benchmarking the 8 KB norms would measure per-call
// overhead and call it bandwidth.
//
// Split into FOUR GROUPS OF EQUAL BYTES, one per arm, by greedy
// bin-packing over a size-sorted list. Equal COUNTS is not the same
// thing and is not good enough: this checkpoint has two ~1.2 GB tensors
// among thirty ~150 MB ones, so round-robin by index put 1.50 GB in two
// groups and 0.39 GB in the other two. With one round per arm that
// makes the arms measure different workloads -- one long sequential
// read is not the same as eight medium ones -- and the comparison stops
// being one.
std::vector<std::vector<std::string>>
tensor_groups(const MetalLlamaWeights& w, std::size_t want,
              std::size_t* max_bytes)
{
  std::vector<std::pair<std::size_t, std::string>> all;
  for (const std::string& n : w.tensor_names()) {
    const auto* ti = w.info(n);
    if (ti == nullptr || ti->shard < 0) { continue; }
    if (ti->nbytes < (4u << 20)) { continue; }
    all.emplace_back((std::size_t)ti->nbytes, n);
  }
  std::sort(all.begin(), all.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  if (all.size() > want) { all.resize(want); }
  *max_bytes = all.empty() ? 0 : all.front().first;

  std::vector<std::vector<std::string>> g(4);
  std::size_t load[4] = {0, 0, 0, 0};
  for (const auto& kv : all) {
    int lightest = 0;
    for (int i = 1; i < 4; ++i) {
      if (load[i] < load[lightest]) { lightest = i; }
    }
    g[(std::size_t)lightest].push_back(kv.second);
    load[lightest] += kv.first;
  }
  return g;
}

void
run_one(const std::string& dir, MetalCompute& mc, int rounds, int rot)
{
  std::printf("\n=== %s ===\n", dir.c_str());
  auto ws = WeightSet::open(dir, nullptr);
  if (ws == nullptr) {
    check(false, "cannot open " + dir);
    return;
  }
  const MetalLlamaWeights& src = ws->src();

  std::size_t max_bytes = 0;
  // Four byte-balanced groups, one per arm, so no arm ever re-reads
  // what another has just pulled into the page cache.
  const auto groups = tensor_groups(src, 32, &max_bytes);
  std::size_t n_all = 0;
  for (const auto& g : groups) { n_all += g.size(); }
  if (n_all < 4 || max_bytes == 0) {
    check(false, "not enough large tensors to benchmark");
    return;
  }
  std::printf("  %zu tensors, largest %.1f MB, group bytes", n_all,
              (double)max_bytes / (1024.0 * 1024.0));
  for (const auto& g : groups) {
    std::size_t b = 0;
    for (const std::string& n : g) {
      const auto* ti = src.info(n);
      if (ti != nullptr) { b += (std::size_t)ti->nbytes; }
    }
    std::printf(" %.2f", (double)b / (1024.0 * 1024.0 * 1024.0));
  }
  std::printf(" GB\n");

  // The reusable destination the pread arms write into -- the whole
  // point of the refill design, so it is allocated once here too.
  SharedBuffer dst = mc.make_shared_buffer(max_bytes);
  if (dst.empty()) {
    check(false, "cannot allocate the destination buffer");
    return;
  }

  // NOTE the fourth arm does LESS WORK than the other three: a wrap
  // copies nothing, so what it times is the page faults alone. It is
  // here because those faults are the bookkeeping under discussion, not
  // because its rate is comparable to a copy's.
  Arm arms[4] = {{"mmap+memcpy"}, {"pread F_NOCACHE"}, {"pread cached"},
                 {"mmap wrap+fault"}};
  bool bytes_agree = true;

  for (int r0 = 0; r0 < rounds; ++r0) {
    const int r = r0 + rot;
    for (int a = 0; a < 4; ++a) {
      // ROTATE, and take every fourth tensor of a SIZE-SORTED list
      // rather than a contiguous quarter of it. Contiguous quarters are
      // wildly unequal in bytes -- the first holds the eight largest
      // tensors and the last the eight smallest -- so an arm's total
      // would depend on which quarters it happened to draw. Round-robin
      // makes the four groups the same size to within one tensor.
      const auto& grp = groups[(std::size_t)(((a + r) % 4 + 4) % 4)];
      for (std::size_t gi = 0; gi < grp.size(); ++gi) {
        const std::string& nm = grp[gi];
        const auto* ti = src.info(nm);
        if (ti == nullptr) { continue; }
        const std::size_t nb = (std::size_t)ti->nbytes;

        std::uint64_t fp = 0;
        const double t0 = now_s();
        switch (a) {
          case 0: {
            MetalLlamaWeights::LoadCost c;
            SharedBuffer b = src.load(nm, &mc, &c);
            if (b.empty()) { bytes_agree = false; break; }
            fp = fingerprint(b.contents(), nb);
            arms[0].alloc_ms += c.alloc_ms;
            break;
          }
          case 1:
            if (!src.pread_into(nm, dst.contents(), nb, /*uncached=*/true)) {
              bytes_agree = false;
              break;
            }
            fp = fingerprint(dst.contents(), nb);
            break;
          case 2:
            if (!src.pread_into(nm, dst.contents(), nb, /*uncached=*/false)) {
              bytes_agree = false;
              break;
            }
            fp = fingerprint(dst.contents(), nb);
            break;
          default: {
            // The zero-copy wrap is only honest if the pages are then
            // TOUCHED -- an untouched mapping has done no I/O at all,
            // and the faults are exactly the cost under discussion.
            SharedBuffer b = src.load_mapped(nm, &mc);
            if (b.empty()) { bytes_agree = false; break; }
            fp = fingerprint(b.contents(), nb);
            break;
          }
        }
        const double dt = now_s() - t0;
        arms[a].secs += dt;
        arms[a].bytes += (double)nb;
        // Same bytes through every path: a fast arm that read the wrong
        // region would otherwise look like a win.
        static std::uint64_t seen[256] = {0};
        const std::size_t slot =
            (std::hash<std::string>{}(nm)) % 256;
        if (seen[slot] == 0) { seen[slot] = fp; }
        else if (seen[slot] != fp) { bytes_agree = false; }
      }
    }
  }

  std::printf("\n  %-18s %10s %12s %10s\n", "arm", "GB/s", "GB moved",
              "alloc");
  for (const Arm& a : arms) {
    std::printf("  %-18s %10.2f %12.2f %9.0f ms\n", a.name, a.gbps(),
                a.bytes / (1024.0 * 1024.0 * 1024.0), a.alloc_ms);
  }
  const double best_pread = std::max(arms[1].gbps(), arms[2].gbps());
  if (arms[0].gbps() > 0.0) {
    std::printf("\n  pread / mmap+memcpy = %.2fx   (cached-pread / "
                "mmap+memcpy = %.2fx)\n",
                best_pread / arms[0].gbps(),
                arms[2].gbps() / arms[0].gbps());
  }
  check(bytes_agree, "every arm returned the same bytes");
}

}  // namespace

int
main()
{
  std::string spec;
  if (const char* s = std::getenv("VPIPE_U15_IO_BENCH")) { spec = s; }
  if (spec.empty()) {
    if (const char* s = std::getenv("VPIPE_U15_TEST_MODEL_PATH")) {
      spec = s;
    }
  }
  if (spec.empty()) {
    std::printf("SKIPPED: VPIPE_U15_IO_BENCH / VPIPE_U15_TEST_MODEL_PATH "
                "unset -- this benchmark did NOT run.\n");
    return 0;
  }
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no usable Metal device. NOTHING was measured.\n");
    return 0;
  }
  int rounds = 3;
  if (const char* s = std::getenv("VPIPE_U15_IO_ROUNDS")) {
    rounds = std::max(1, std::atoi(s));
  }
  // Which tensor group each arm starts on. A COLD run is one round --
  // every arm reads its group exactly once, and only if the caller
  // evicted the page cache first is that group actually cold. Varying
  // this across cold runs is how an arm/group pairing gets checked for
  // being a result in itself.
  int rot = 0;
  if (const char* s = std::getenv("VPIPE_U15_IO_ROTATE")) {
    rot = std::atoi(s);
  }
  std::printf("%d rounds, arms interleaved, group offset %d\n", rounds,
              rot);
  if (rounds == 1) {
    std::printf("COLD MODE: every arm reads its group ONCE. This is only "
                "a cold measurement if the page cache was evicted "
                "first -- nothing here can check that for you.\n");
  }
  for (const std::string& d : split_paths(spec)) {
    run_one(d, mc, rounds, rot);
  }
  std::printf("\n%d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
