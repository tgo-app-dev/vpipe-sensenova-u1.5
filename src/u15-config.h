#ifndef VPIPE_U15_CONFIG_H
#define VPIPE_U15_CONFIG_H

#include <string>
#include <vector>

namespace vpipe { class FlexData; }

namespace u15 {

// The family tag. One spelling, used by the model-config stage, the
// generate stage and the catalogue, so a config wired to the wrong
// checkpoint is REPORTED rather than half-applied.
inline constexpr const char* kFamily = "sensenova-u1.5";

// The plugin's own metallibs, registered at load and resolved by name
// through load_library(). bf16 is what the backbone runs; the f32 twin
// exists for the pieces the reference keeps in float -- the two scalar
// embedders and the 2-D vision RoPE (see ARCHITECTURE.md sections 5-7).
inline constexpr const char* kMetalLibBf16 = "u15_kernels_bf16";
inline constexpr const char* kMetalLibF32  = "u15_kernels_f32";

// ---------------------------------------------------------------------
// Geometry. patch_size 16 and downsample_ratio 0.5 give a 2x2 merge, so
// ONE LLM token covers a 32x32 pixel block. Both image dimensions must
// therefore be multiples of 32.
// ---------------------------------------------------------------------
inline constexpr int kPixelsPerToken = 32;

// Round a pixel dimension UP to a multiple of 32. Rounds rather than
// rejects: a graph must be able to change families without being
// re-authored.
int align_dimension(int px);

// ---------------------------------------------------------------------
// The backbone, as `llm_config` spells it. Field names track the JSON
// keys so the two can be compared by eye; the defaults are the shipped
// 8B-MoT values, which is what a missing key means.
// ---------------------------------------------------------------------
struct LlmConfig {
  int    hidden_size          = 4096;
  int    num_hidden_layers    = 42;
  int    num_attention_heads  = 32;
  int    num_key_value_heads  = 8;
  int    head_dim             = 128;
  int    intermediate_size    = 12288;
  int    vocab_size           = 151936;
  double rms_norm_eps         = 1e-6;

  // The 1-D sequence ("t") rope. Stock Qwen3.
  double rope_theta           = 5000000.0;
  int    max_position_embeddings = 262144;

  // The 2-D ("h"/"w") rope. NOT stock Qwen3 -- these two keys are what
  // make the head's 128 dims split three ways. See ARCHITECTURE.md #3.
  double rope_theta_hw        = 10000.0;
  int    max_position_embeddings_hw = 10000;

  // Derived, so the split is stated once rather than at every use site.
  int t_dim() const { return head_dim / 2; }   // 64, theta 5e6
  int h_dim() const { return head_dim / 4; }   // 32, theta 1e4
  int w_dim() const { return head_dim / 4; }   // 32, theta 1e4
};

// The patch embedder, as `vision_config` spells it. Both instances --
// understanding and generation -- share this shape and differ only in
// which weights they bind.
struct VisionConfig {
  int    hidden_size     = 1024;     // per-patch embedding width
  int    llm_hidden_size = 4096;     // after the 2x2 dense merge
  int    patch_size      = 16;
  int    num_channels    = 3;
  double downsample_ratio = 0.5;     // -> merge_size 2
  double rope_theta_vision = 10000.0;
  int    max_position_embeddings_vision = 10000;

  int merge_size() const;            // int(1 / downsample_ratio)
};

// The generation half of the top-level config.
struct GenConfig {
  int    patch_size       = 16;
  double downsample_ratio = 0.5;

  // use_pixel_head selects the ConvDecoder; use_adaLN false means the
  // SimpleMLPAdaLN head is absent from the checkpoint entirely. Both are
  // asserted at load rather than trusted, because the head they select
  // has a different tensor set.
  bool   use_pixel_head   = true;
  bool   use_adaLN        = false;

  // Sampling. NOTE `time_schedule` is read but NOT acted on: the
  // reference's _apply_time_schedule assigns "standard" on entry, so the
  // dynamic branch is dead code. Kept here so a checkpoint that says
  // something else can be REFUSED instead of silently mis-sampled.
  std::string time_schedule = "standard";
  double timestep_shift     = 1.0;
  double t_eps              = 0.05;

  // Resolution-dependent noise scale.
  bool   add_noise_scale_embedding    = true;
  std::string noise_scale_mode        = "resolution";
  double noise_scale                  = 1.0;
  double noise_scale_max_value        = 16.0;
  int    noise_scale_base_image_seq_len = 64;

  // The chat template name. "neo1_0" is ChatML.
  std::string tmpl = "neo1_0";

  // Must be 0 -- the reference asserts it in t2i_generate.
  int concat_time_token_num = 0;

  int merge_size() const;

  // The resolution-dependent initial-noise scale, given the image token
  // count L. sqrt(L / base), capped at noise_scale_max_value.
  double init_noise_scale(int image_tokens) const;
};

struct U15Config {
  LlmConfig    llm;
  VisionConfig vision;
  GenConfig    gen;

  // Straight off config.json, used only for identification.
  std::string model_type;                  // "neo_chat"
  std::vector<std::string> architectures;  // ["NEOChatModel"]

  // Token geometry for a pixel size. Both must be multiples of 32.
  int token_h(int height) const { return height / kPixelsPerToken; }
  int token_w(int width)  const { return width  / kPixelsPerToken; }
  int grid_h(int height)  const { return height / gen.patch_size; }
  int grid_w(int width)   const { return width  / gen.patch_size; }
};

// Parse <dir>/config.json. Returns false if the file is missing or is
// not a SenseNova-U1.5 config; `why` (when non-null) gets a one-line
// reason suitable for a log.
bool parse_config(const std::string& dir, U15Config* out, std::string* why);

// Does `dir` hold a SenseNova-U1.5 MoT checkpoint?
//
// This is deliberately NOT satisfiable by defaults. LTX-2.5 claimed a
// MiniMax-H3 repack because its class_name field DEFAULTED to the value
// the check compared against, so the check could never fail. Here every
// clause must be positively present:
//
//   * config.json says model_type == "neo_chat", AND
//   * architectures contains "NEOChatModel", AND
//   * the tensor index carries the MoT signature -- at least one
//     `_mot_gen` weight AND `fm_modules.fm_head.conv1.weight`.
//
// The last clause is what separates this from the non-MoT and the
// A3B-MoE members of the same family, which share the first two.
bool detect(const std::string& dir, std::string* why);

}  // namespace u15

#endif  // VPIPE_U15_CONFIG_H
