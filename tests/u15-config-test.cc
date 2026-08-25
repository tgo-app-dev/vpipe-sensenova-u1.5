// Checks the config + detection layer: the geometry rules, the
// resolution-dependent noise scale, and -- the part that matters most --
// that detection CANNOT pass by default.
//
// The geometry half runs everywhere. The checkpoint half is gated on
// VPIPE_U15_TEST_MODEL_PATH and SKIPS when it is unset, so read the
// output rather than the exit code: a vacuous skip looks exactly like a
// pass. It says which it did.

#include "u15-config.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

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

bool
close_(double a, double b, double tol = 1e-9)
{
  return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b));
}

void
write_(const fs::path& p, const std::string& s)
{
  std::ofstream f(p, std::ios::binary);
  f << s;
}

// A config.json that is structurally a U1.5 config. Callers mutate one
// field at a time to check that each clause of detect() is load-bearing.
std::string
base_config_(const char* model_type = "neo_chat",
             const char* arch = "NEOChatModel")
{
  std::string s = R"({
  "architectures": ["ARCH"],
  "model_type": "MTYPE",
  "downsample_ratio": 0.5,
  "patch_size": 16,
  "use_pixel_head": true,
  "use_adaLN": false,
  "timestep_shift": 1.0,
  "time_schedule": "standard",
  "t_eps": 0.05,
  "add_noise_scale_embedding": true,
  "noise_scale_mode": "resolution",
  "noise_scale": 1.0,
  "noise_scale_max_value": 16.0,
  "noise_scale_base_image_seq_len": 64,
  "concat_time_token_num": 0,
  "template": "neo1_0",
  "llm_config": {
    "hidden_size": 4096, "num_hidden_layers": 42,
    "num_attention_heads": 32, "num_key_value_heads": 8,
    "head_dim": 128, "intermediate_size": 12288,
    "vocab_size": 151936, "rms_norm_eps": 1e-06,
    "rope_theta": 5000000.0, "rope_theta_hw": 10000.0,
    "max_position_embeddings": 262144,
    "max_position_embeddings_hw": 10000
  },
  "vision_config": {
    "hidden_size": 1024, "llm_hidden_size": 4096, "patch_size": 16,
    "num_channels": 3, "downsample_ratio": 0.5,
    "rope_theta_vision": 10000.0,
    "max_position_embeddings_vision": 10000
  }
})";
  const auto sub = [&s](const std::string& from, const std::string& to) {
    const auto at = s.find(from);
    if (at != std::string::npos) { s.replace(at, from.size(), to); }
  };
  sub("ARCH", arch);
  sub("MTYPE", model_type);
  return s;
}

// The MoT signature detect() looks for in the tensor index.
constexpr const char* kMotIndex = R"({"weight_map": {
  "language_model.model.layers.0.mlp_mot_gen.gate_proj.weight": "a.safetensors",
  "fm_modules.fm_head.conv1.weight": "a.safetensors"
}})";

void
test_geometry()
{
  std::printf("geometry\n");
  check(u15::align_dimension(1024) == 1024, "1024 is already legal");
  check(u15::align_dimension(1000) == 1024, "1000 -> 1024");
  check(u15::align_dimension(33) == 64,     "33 -> 64");
  check(u15::align_dimension(32) == 32,     "32 -> 32");
  check(u15::align_dimension(0) == 32,      "0 -> 32 (never zero)");

  bool all_legal = true;
  for (int n = 1; n <= 4096; ++n) {
    const int a = u15::align_dimension(n);
    if (a % 32 != 0 || a < n) { all_legal = false; break; }
  }
  check(all_legal, "every output is a multiple of 32 and >= input");

  u15::U15Config c;
  check(c.token_h(1024) == 32 && c.token_w(512) == 16,
        "1024x512 -> 32x16 tokens (32 px per token)");
  check(c.grid_h(1024) == 64 && c.grid_w(512) == 32,
        "1024x512 -> 64x32 patch grid (16 px per patch)");
  check(c.vision.merge_size() == 2, "downsample_ratio 0.5 -> merge 2");
  check(c.llm.t_dim() == 64 && c.llm.h_dim() == 32 && c.llm.w_dim() == 32,
        "head_dim 128 splits 64/32/32 for t/h/w");
}

void
test_noise_scale()
{
  std::printf("initial noise scale\n");
  u15::GenConfig g;
  // The reference: sqrt((grid_h*grid_w)/merge^2/base) * noise_scale,
  // capped at noise_scale_max_value. (grid/merge^2) is the token count.
  check(close_(g.init_noise_scale(64), 1.0), "L == base -> 1.0");
  check(close_(g.init_noise_scale(256), 2.0), "L = 4*base -> 2.0");
  // 1024x1024 -> 32x32 = 1024 tokens -> sqrt(16) = 4.
  check(close_(g.init_noise_scale(1024), 4.0), "1024 tokens -> 4.0");
  // The cap binds well before 4K: 16^2 * 64 = 16384 tokens.
  check(close_(g.init_noise_scale(1 << 20), 16.0), "capped at 16");
  check(g.init_noise_scale(1 << 20) <= g.noise_scale_max_value,
        "never exceeds noise_scale_max_value");
}

void
test_detection(const fs::path& tmp)
{
  std::printf("detection cannot pass by default\n");
  fs::create_directories(tmp);

  // A complete, valid checkpoint stub.
  const fs::path good = tmp / "good";
  fs::create_directories(good);
  write_(good / "config.json", base_config_());
  write_(good / "model.safetensors.index.json", kMotIndex);
  std::string why;
  check(u15::detect(good.string(), &why),
        "a real U1.5 MoT stub is claimed" + (why.empty() ? "" : " (" + why + ")"));

  // Each clause removed in turn must REFUSE. This is the LTX-2.5 trap:
  // an identity check whose fields default to the value it compares
  // against can never fail, so it claims a sibling model's checkpoint.
  const fs::path no_idx = tmp / "no-index";
  fs::create_directories(no_idx);
  write_(no_idx / "config.json", base_config_());
  check(!u15::detect(no_idx.string(), nullptr),
        "REFUSED: no tensor index");

  const fs::path no_mot = tmp / "no-mot";
  fs::create_directories(no_mot);
  write_(no_mot / "config.json", base_config_());
  write_(no_mot / "model.safetensors.index.json",
         R"({"weight_map": {"language_model.model.layers.0.mlp.gate_proj.weight": "a",
                            "fm_modules.fm_head.conv1.weight": "a"}})");
  check(!u15::detect(no_mot.string(), nullptr),
        "REFUSED: neo_chat + NEOChatModel but NO _mot_gen (a non-MoT "
        "sibling)");

  const fs::path no_head = tmp / "no-head";
  fs::create_directories(no_head);
  write_(no_head / "config.json", base_config_());
  write_(no_head / "model.safetensors.index.json",
         R"({"weight_map": {"language_model.model.layers.0.mlp_mot_gen.gate_proj.weight": "a"}})");
  check(!u15::detect(no_head.string(), nullptr),
        "REFUSED: MoT but no fm_head (understanding-only)");

  const fs::path wrong_type = tmp / "wrong-type";
  fs::create_directories(wrong_type);
  write_(wrong_type / "config.json", base_config_("qwen3", "NEOChatModel"));
  write_(wrong_type / "model.safetensors.index.json", kMotIndex);
  check(!u15::detect(wrong_type.string(), nullptr),
        "REFUSED: model_type is not neo_chat");

  const fs::path wrong_arch = tmp / "wrong-arch";
  fs::create_directories(wrong_arch);
  write_(wrong_arch / "config.json",
         base_config_("neo_chat", "Qwen3ForCausalLM"));
  write_(wrong_arch / "model.safetensors.index.json", kMotIndex);
  check(!u15::detect(wrong_arch.string(), nullptr),
        "REFUSED: architectures does not list NEOChatModel");

  const fs::path empty = tmp / "empty";
  fs::create_directories(empty);
  check(!u15::detect(empty.string(), nullptr),
        "REFUSED: an empty directory");
}

void
test_structural_refusals(const fs::path& tmp)
{
  std::printf("structural switches are refused, not adapted\n");
  const auto one = [&](const char* name, const char* from, const char* to,
                       const char* what) {
    const fs::path d = tmp / name;
    fs::create_directories(d);
    std::string cfg = base_config_();
    const auto at = cfg.find(from);
    if (at != std::string::npos) { cfg.replace(at, std::strlen(from), to); }
    write_(d / "config.json", cfg);
    write_(d / "model.safetensors.index.json", kMotIndex);
    u15::U15Config c;
    check(!u15::parse_config(d.string(), &c, nullptr), what);
  };
  one("no-pixel-head", "\"use_pixel_head\": true",
      "\"use_pixel_head\": false",
      "REFUSED: use_pixel_head false (a different tensor set)");
  one("adaln", "\"use_adaLN\": false", "\"use_adaLN\": true",
      "REFUSED: use_adaLN true (SimpleMLPAdaLN not implemented)");
  one("dyn-sched", "\"time_schedule\": \"standard\"",
      "\"time_schedule\": \"dynamic\"",
      "REFUSED: a non-standard time schedule");
  one("concat-time", "\"concat_time_token_num\": 0",
      "\"concat_time_token_num\": 4",
      "REFUSED: concat_time_token_num != 0");
  one("odd-head", "\"head_dim\": 128", "\"head_dim\": 130",
      "REFUSED: head_dim not divisible by 4 (no t/h/w split)");
}

void
test_real_checkpoint()
{
  const char* p = std::getenv("VPIPE_U15_TEST_MODEL_PATH");
  if (p == nullptr || *p == '\0') {
    std::printf("real checkpoint: SKIPPED "
                "(VPIPE_U15_TEST_MODEL_PATH unset) -- this test did NOT "
                "run\n");
    return;
  }
  std::printf("real checkpoint at %s\n", p);
  std::string why;
  u15::U15Config c;
  const bool ok = u15::parse_config(p, &c, &why);
  check(ok, "config.json parses" + (why.empty() ? "" : " (" + why + ")"));
  if (!ok) { return; }

  check(c.model_type == "neo_chat", "model_type is neo_chat");
  check(c.llm.num_hidden_layers == 42, "42 layers");
  check(c.llm.hidden_size == 4096, "hidden 4096");
  check(c.llm.num_attention_heads == 32 && c.llm.num_key_value_heads == 8,
        "32 q heads / 8 kv heads");
  check(c.llm.head_dim == 128, "head_dim 128");
  check(c.llm.intermediate_size == 12288, "intermediate 12288");
  check(close_(c.llm.rope_theta, 5000000.0), "rope_theta 5e6 (t axis)");
  check(close_(c.llm.rope_theta_hw, 10000.0), "rope_theta_hw 1e4 (h/w)");
  check(c.vision.hidden_size == 1024 && c.vision.llm_hidden_size == 4096,
        "vision 1024 -> 4096");
  check(c.vision.patch_size == 16, "patch 16");
  check(c.gen.merge_size() == 2, "merge 2 -> 32 px per token");
  check(c.gen.use_pixel_head && !c.gen.use_adaLN,
        "pixel head, no adaLN");
  check(close_(c.gen.t_eps, 0.05), "t_eps 0.05");
  check(c.gen.tmpl == "neo1_0", "template neo1_0");

  std::string dwhy;
  check(u15::detect(p, &dwhy),
        "detect() claims it" + (dwhy.empty() ? "" : " (" + dwhy + ")"));
}

}  // namespace

int
main()
{
  const fs::path tmp =
      fs::temp_directory_path() / "vpipe-u15-config-test";
  std::error_code ec;
  fs::remove_all(tmp, ec);

  test_geometry();
  test_noise_scale();
  test_detection(tmp);
  test_structural_refusals(tmp);
  test_real_checkpoint();

  fs::remove_all(tmp, ec);
  std::printf("\n%d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
