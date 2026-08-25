#include "u15-model-config-stage.h"
#include "u15-config.h"

#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <string>
#include <utility>

using vpipe::ConfigKey;
using vpipe::ConfigType;
using vpipe::FlexData;
using vpipe::FlexDataPayload;
using vpipe::InEdge;
using vpipe::PortSpec;
using vpipe::SessionContextIntf;
using vpipe::StageCategory;
using vpipe::StageSpec;
using vpipe::model_config::make_config;

namespace u15 {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "cfg_scale", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance. 1 disables it and halves the cost -- "
          "there is then no unconditional pass. The reference's examples "
          "use 4.0. 0 means unset, so the model layer's default stands",
   .def_real = 0.0},

  {.key = "img_cfg_scale", .type = ConfigType::Real, .required = false,
   .doc = "the SECOND guidance axis, for EDITING only: how hard the "
          "result is pushed away from 'these reference images, no "
          "instruction'. 1 (the default) disables it and the "
          "unconditional prefix is never built -- so raising it costs a "
          "THIRD prefill and a third forward per step. Meaningless "
          "without a reference image, and ignored there. 0 means unset",
   .def_real = 0.0},

  {.key = "timestep_shift", .type = ConfigType::Real, .required = false,
   .doc = "how far the sampling schedule is pushed toward the noisy end: "
          "sigma <- s*sigma / (1 + (s-1)*sigma). 1 is the unshifted "
          "linear schedule; the reference's examples use 3.0. 0 means "
          "unset",
   .def_real = 0.0},

  {.key = "cfg_norm", .type = ConfigType::String, .required = false,
   .doc = "how the guided velocity is renormalised: 'none' (the "
          "reference default), 'global' (scale so its norm does not "
          "exceed the conditional's), 'channel' (the same per image "
          "token), or 'cfg_zero_star' (CFG-Zero* -- projects the "
          "unconditional onto the conditional first, and emits EXACTLY "
          "zero on the first step)",
   .def_str = ""},

  {.key = "cfg_interval_lo", .type = ConfigType::Real, .required = false,
   .doc = "guidance applies only while the timestep is in "
          "[lo, hi]. Outside it the conditional velocity is used "
          "unguided, which costs one forward instead of two. Defaults to "
          "the whole run",
   .def_real = -1.0},
  {.key = "cfg_interval_hi", .type = ConfigType::Real, .required = false,
   .doc = "the upper end of that interval", .def_real = -1.0},

  {.key = "t_eps", .type = ConfigType::Real, .required = false,
   .doc = "floor on the (1 - t) denominator that turns the head's x0 "
          "prediction into a velocity. Without it the last steps divide "
          "by ~0 and the trajectory blows up. The reference's example "
          "scripts leave it at 0.02; this checkpoint's config.json says "
          "0.05. 0 means unset",
   .def_real = 0.0},

  {.key = "init_noise", .type = ConfigType::String, .required = false,
   .doc = "debug/repro: a raw f32 [H][W][3] channel-last initial noise "
          "field, already scaled, in the model's [-1, 1] space. Exists so "
          "a run can be compared against the reference implementation, "
          "whose torch RNG this port does not reproduce -- so a matching "
          "seed does NOT give a matching image, and any comparison that "
          "assumes otherwise is measuring the RNG",
   .def_str = ""},
};

const PortSpec kIports[] = {
  {.name = "trigger",
   .doc = "OPTIONAL beat that gates re-emitting the config (a chrono "
          "tick, a prompt source, a feedback loop). Any payload -- "
          "receipt is the signal. Unwired, the stage emits once for the "
          "run",
   .type = nullptr, .clock_group = 0},
};

const PortSpec kOports[] = {
  {.name = "model_config",
   .doc = "SenseNova-U1.5 generation parameters as one FlexData object "
          "{model_family: sensenova-u1.5, ...}, for the generate stage's "
          "model_config iport. Only the keys the graph actually set are "
          "present, so the model layer's own defaults survive",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = U15ModelConfigStage::kTypeName,
  .doc = "Source: the SenseNova-U1.5-specific sampling parameters -- "
         "guidance scale and its renormalisation, the timestep shift, "
         "the guidance interval and the x0->velocity epsilon -- as one "
         "FlexData beat for the generate stage to latch. These do not "
         "live on the generate stage because none of them mean the same "
         "thing to a latent-diffusion family. One beat then done; with a "
         "trigger iport, one beat per inbound beat.",
  .display_name = "SenseNova-U1.5 Model Config",
  .category = StageCategory::ModelSpecificConfig,
  .iports = kIports,
  .oports = kOports,
  .attrs = kAttrs,
};

}  // namespace

U15ModelConfigStage::U15ModelConfigStage(const SessionContextIntf* s,
                                         std::string               id,
                                         std::vector<InEdge>       iports,
                                         FlexData                  config)
  : vpipe::ModelConfigSourceStage<U15ModelConfigStage>(s, std::move(id),
                                                       std::move(iports),
                                                       std::move(config))
{
  // The spec is DOCUMENTATION, and ModelConfigSourceStage does NOT do
  // this for its subclasses -- its header says so explicitly. Skip it
  // and the stage reports ZERO oports, so pipeline_from_spec refuses
  // every downstream edge with "oport 0 out of range" and no pipeline
  // launches at all.
  this->allocate_oports(std::size(kOports));

  // Which keys the GRAPH set, as opposed to which have a schema default.
  // That distinction is the whole contract: the model layer holds the
  // shipped numbers, and a source that emitted its own defaults would
  // overwrite them with something that merely looks configured.
  const auto was_set = [this](const char* k) {
    const FlexData& c = this->config();
    if (!c.is_object()) { return false; }
    return c.as_object().contains(k);
  };

  _cfg_scale = this->attr_real("cfg_scale");
  _img_cfg_scale = this->attr_real("img_cfg_scale");
  _timestep_shift = this->attr_real("timestep_shift");
  _t_eps = this->attr_real("t_eps");
  _cfg_norm = this->attr_str("cfg_norm");
  _cfg_lo = this->attr_real("cfg_interval_lo");
  _cfg_hi = this->attr_real("cfg_interval_hi");
  _init_noise = this->attr_str("init_noise");

  _has_cfg_scale = was_set("cfg_scale") && _cfg_scale > 0.0;
  _has_img_cfg = was_set("img_cfg_scale") && _img_cfg_scale > 0.0;
  _has_shift = was_set("timestep_shift") && _timestep_shift > 0.0;
  _has_t_eps = was_set("t_eps") && _t_eps > 0.0;
  _has_norm = was_set("cfg_norm") && !_cfg_norm.empty();
  _has_noise = was_set("init_noise") && !_init_noise.empty();
  // The interval is one FACT, not two: half of it is meaningless, so
  // both ends must be set for either to be emitted.
  _has_interval = was_set("cfg_interval_lo") && was_set("cfg_interval_hi") &&
                  _cfg_lo >= 0.0 && _cfg_hi >= 0.0;

  if (!_cfg_norm.empty() && _cfg_norm != "none" && _cfg_norm != "global" &&
      _cfg_norm != "channel" && _cfg_norm != "cfg_zero_star") {
    this->fail_config(vpipe::fmt(
        "cfg_norm must be one of none / global / channel / cfg_zero_star"));
  }
  if (was_set("cfg_interval_lo") != was_set("cfg_interval_hi")) {
    this->fail_config(vpipe::fmt(
        "cfg_interval_lo and cfg_interval_hi must be set together -- half "
        "an interval has no meaning"));
  }
}

const StageSpec&
U15ModelConfigStage::spec() const noexcept
{
  return kSpec;
}

const StageSpec*
U15ModelConfigStage::stage_spec() noexcept
{
  return &kSpec;
}

FlexData
U15ModelConfigStage::resolved_config() const
{
  FlexData fd = make_config(kFamily);
  auto o = fd.as_object();
  const auto put_real = [&o](bool set, const char* k, double v) {
    if (set) { o.insert_or_assign(k, FlexData::make_real(v)); }
  };
  put_real(_has_cfg_scale, "cfg_scale", _cfg_scale);
  put_real(_has_img_cfg, "img_cfg_scale", _img_cfg_scale);
  put_real(_has_shift, "timestep_shift", _timestep_shift);
  put_real(_has_t_eps, "t_eps", _t_eps);
  if (_has_norm) {
    o.insert_or_assign("cfg_norm", FlexData::make_string(_cfg_norm));
  }
  if (_has_interval) {
    FlexData arr = FlexData::make_array();
    auto a = arr.as_array();
    a.push_back(FlexData::make_real(_cfg_lo));
    a.push_back(FlexData::make_real(_cfg_hi));
    o.insert_or_assign("cfg_interval", std::move(arr));
  }
  if (_has_noise) {
    o.insert_or_assign("init_noise", FlexData::make_string(_init_noise));
  }
  return fd;
}

}  // namespace u15
