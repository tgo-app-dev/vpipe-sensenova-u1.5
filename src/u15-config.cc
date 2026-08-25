#include "u15-config.h"

#include "common/flex-data.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;
using vpipe::FlexData;

namespace u15 {

namespace {

bool
read_file_(const std::string& path, std::string* out)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { return false; }
  std::ostringstream ss;
  ss << f.rdbuf();
  *out = ss.str();
  return true;
}

// Read one key, leaving `out` alone when it is absent. "Leave alone"
// rather than "write the default": the struct already holds the shipped
// values, so a missing key means "as shipped" and a checkpoint that DOES
// say something always wins.
void
get_int_(const FlexData::ConstObjectView& o, const char* k, int& out)
{
  if (o.contains(k)) { out = (int)o.at(k).as_int(out); }
}

void
get_real_(const FlexData::ConstObjectView& o, const char* k, double& out)
{
  if (o.contains(k)) { out = o.at(k).as_real(out); }
}

void
get_bool_(const FlexData::ConstObjectView& o, const char* k, bool& out)
{
  if (o.contains(k)) { out = o.at(k).as_bool(out); }
}

void
get_str_(const FlexData::ConstObjectView& o, const char* k, std::string& out)
{
  if (o.contains(k)) { out = std::string(o.at(k).as_string(out)); }
}

// vision_config spells llm_hidden_size and downsample_ratio as ARRAYS
// (the reference indexes [0] on both), so accept either form.
void
get_scalar_or_first_(const FlexData::ConstObjectView& o, const char* k,
                     double& out)
{
  if (!o.contains(k)) { return; }
  const FlexData v = o.at(k);
  if (v.is_array()) {
    const auto arr = v.as_array();
    if (arr.size() > 0) { out = arr.at(0).as_real(out); }
    return;
  }
  out = v.as_real(out);
}

void
get_scalar_or_first_int_(const FlexData::ConstObjectView& o, const char* k,
                         int& out)
{
  double d = (double)out;
  get_scalar_or_first_(o, k, d);
  out = (int)d;
}

}  // namespace

int
align_dimension(int px)
{
  if (px <= 0) { return kPixelsPerToken; }
  const int r = px % kPixelsPerToken;
  return r == 0 ? px : px + (kPixelsPerToken - r);
}

int
VisionConfig::merge_size() const
{
  if (downsample_ratio <= 0.0) { return 2; }
  return (int)std::lround(1.0 / downsample_ratio);
}

int
GenConfig::merge_size() const
{
  if (downsample_ratio <= 0.0) { return 2; }
  return (int)std::lround(1.0 / downsample_ratio);
}

double
GenConfig::init_noise_scale(int image_tokens) const
{
  // The reference computes sqrt((grid_h*grid_w) / merge_size^2 / base),
  // and (grid_h*grid_w)/merge_size^2 IS the image token count -- so this
  // is sqrt(L / base). Written that way because L is what every caller
  // already has, and squaring it back would only lose the exactness.
  if (noise_scale_mode != "resolution" && noise_scale_mode != "dynamic" &&
      noise_scale_mode != "dynamic_sqrt") {
    return std::min(noise_scale, noise_scale_max_value);
  }
  const double base = (double)(noise_scale_base_image_seq_len > 0
                                   ? noise_scale_base_image_seq_len
                                   : 64);
  double s = std::sqrt((double)image_tokens / base) * noise_scale;
  if (noise_scale_mode == "dynamic_sqrt") { s = std::sqrt(s); }
  return std::min(s, noise_scale_max_value);
}

bool
parse_config(const std::string& dir, U15Config* out, std::string* why)
{
  const auto set_why = [why](const char* m) {
    if (why != nullptr) { *why = m; }
    return false;
  };
  if (out == nullptr) { return set_why("no output"); }

  std::string txt;
  const fs::path cj = fs::path(dir) / "config.json";
  if (!read_file_(cj.string(), &txt)) {
    return set_why("no config.json");
  }

  FlexData cfg;
  try { cfg = FlexData::from_json(txt); }
  catch (...) { return set_why("config.json is not valid JSON"); }
  if (!cfg.is_object()) { return set_why("config.json is not an object"); }

  // as_object() returns a VIEW into `cfg`, so `cfg` must outlive it --
  // it does, it is a local. Every nested view below is bound to a named
  // local for the same reason.
  const auto root = cfg.as_object();

  U15Config c;
  get_str_(root, "model_type", c.model_type);

  if (root.contains("architectures")) {
    const FlexData a = root.at("architectures");
    if (a.is_array()) {
      const auto arr = a.as_array();
      for (std::size_t i = 0; i < arr.size(); ++i) {
        c.architectures.push_back(std::string(arr.at(i).as_string("")));
      }
    }
  }

  // ---- the backbone -------------------------------------------------
  if (root.contains("llm_config")) {
    const FlexData lc = root.at("llm_config");
    if (lc.is_object()) {
      const auto l = lc.as_object();
      get_int_(l, "hidden_size", c.llm.hidden_size);
      get_int_(l, "num_hidden_layers", c.llm.num_hidden_layers);
      get_int_(l, "num_attention_heads", c.llm.num_attention_heads);
      get_int_(l, "num_key_value_heads", c.llm.num_key_value_heads);
      get_int_(l, "head_dim", c.llm.head_dim);
      get_int_(l, "intermediate_size", c.llm.intermediate_size);
      get_int_(l, "vocab_size", c.llm.vocab_size);
      get_real_(l, "rms_norm_eps", c.llm.rms_norm_eps);
      get_real_(l, "rope_theta", c.llm.rope_theta);
      get_int_(l, "max_position_embeddings",
               c.llm.max_position_embeddings);
      get_real_(l, "rope_theta_hw", c.llm.rope_theta_hw);
      get_int_(l, "max_position_embeddings_hw",
               c.llm.max_position_embeddings_hw);
    }
  }

  // ---- the patch embedder -------------------------------------------
  if (root.contains("vision_config")) {
    const FlexData vc = root.at("vision_config");
    if (vc.is_object()) {
      const auto v = vc.as_object();
      get_int_(v, "hidden_size", c.vision.hidden_size);
      get_scalar_or_first_int_(v, "llm_hidden_size",
                               c.vision.llm_hidden_size);
      get_int_(v, "patch_size", c.vision.patch_size);
      get_int_(v, "num_channels", c.vision.num_channels);
      get_scalar_or_first_(v, "downsample_ratio",
                           c.vision.downsample_ratio);
      get_real_(v, "rope_theta_vision", c.vision.rope_theta_vision);
      get_int_(v, "max_position_embeddings_vision",
               c.vision.max_position_embeddings_vision);
    }
  }

  // ---- generation ----------------------------------------------------
  get_int_(root, "patch_size", c.gen.patch_size);
  get_real_(root, "downsample_ratio", c.gen.downsample_ratio);
  get_bool_(root, "use_pixel_head", c.gen.use_pixel_head);
  get_bool_(root, "use_adaLN", c.gen.use_adaLN);
  get_str_(root, "time_schedule", c.gen.time_schedule);
  get_real_(root, "timestep_shift", c.gen.timestep_shift);
  get_real_(root, "t_eps", c.gen.t_eps);
  get_bool_(root, "add_noise_scale_embedding",
            c.gen.add_noise_scale_embedding);
  get_str_(root, "noise_scale_mode", c.gen.noise_scale_mode);
  get_real_(root, "noise_scale", c.gen.noise_scale);
  get_real_(root, "noise_scale_max_value", c.gen.noise_scale_max_value);
  get_int_(root, "noise_scale_base_image_seq_len",
           c.gen.noise_scale_base_image_seq_len);
  get_str_(root, "template", c.gen.tmpl);
  get_int_(root, "concat_time_token_num", c.gen.concat_time_token_num);

  // The two structural switches. REFUSE rather than adapt: each selects
  // a different tensor set, so running the wrong one does not degrade
  // the image, it reads weights that are not there.
  if (!c.gen.use_pixel_head) {
    return set_why("use_pixel_head is false: this build implements only "
                   "the ConvDecoder pixel head");
  }
  if (c.gen.use_adaLN) {
    return set_why("use_adaLN is true: the SimpleMLPAdaLN head is not "
                   "implemented");
  }
  if (c.gen.concat_time_token_num != 0) {
    return set_why("concat_time_token_num != 0 is not implemented");
  }
  if (c.gen.time_schedule != "standard") {
    return set_why("only the 'standard' time schedule is implemented");
  }

  // The head split has to divide evenly three ways: t = D/2, h = w =
  // D/4. Anything else is a different model wearing this config.
  if (c.llm.head_dim % 4 != 0) {
    return set_why("head_dim is not divisible by 4, so the t/h/w rope "
                   "split does not exist");
  }

  *out = c;
  return true;
}

bool
detect(const std::string& dir, std::string* why)
{
  const auto set_why = [why](const char* m) {
    if (why != nullptr) { *why = m; }
    return false;
  };

  U15Config c;
  if (!parse_config(dir, &c, why)) { return false; }

  if (c.model_type != "neo_chat") {
    return set_why("model_type is not 'neo_chat'");
  }
  bool has_arch = false;
  for (const auto& a : c.architectures) {
    if (a == "NEOChatModel") { has_arch = true; break; }
  }
  if (!has_arch) {
    return set_why("architectures does not list NEOChatModel");
  }

  // The structural clause. model_type and architectures are shared with
  // the non-MoT and A3B-MoE members of this family, so on their own they
  // would claim a checkpoint this code cannot run. The MoT signature is
  // the dual expert set; the pixel head is what makes it a GENERATION
  // checkpoint rather than an understanding-only one.
  std::string idx;
  const fs::path ij = fs::path(dir) / "model.safetensors.index.json";
  if (!read_file_(ij.string(), &idx)) {
    return set_why("no model.safetensors.index.json");
  }
  if (idx.find("_mot_gen") == std::string::npos) {
    return set_why("no _mot_gen tensors: not a MoT checkpoint");
  }
  if (idx.find("fm_modules.fm_head.conv1.weight") == std::string::npos) {
    return set_why("no fm_modules.fm_head: not a generation checkpoint");
  }
  return true;
}

}  // namespace u15
