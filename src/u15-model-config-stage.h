#ifndef VPIPE_U15_MODEL_CONFIG_STAGE_H
#define VPIPE_U15_MODEL_CONFIG_STAGE_H

#include "common/flex-data.h"
#include "pipeline/stage-spec.h"
#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace u15 {

// SenseNova-U1.5's own generation knobs, as a config SOURCE.
//
// They live here rather than on the generate stage for the reason
// stages/model-config-source.h gives: a stage serving several families
// accumulates the union of their knobs, and each key is then inert --
// silently -- on whichever family is not resident. `cfg_norm` and
// `timestep_shift` mean nothing to a DiT family, and `guidance_scale`
// means something subtly different here.
//
// EVERY KEY IS EMITTED ONLY WHEN SET. The model layer holds the shipped
// defaults; a source that helpfully emitted its own would overwrite them
// with something that merely looks configured.
class U15ModelConfigStage
  : public vpipe::ModelConfigSourceStage<U15ModelConfigStage> {
public:
  static constexpr const char* kTypeName = "sensenova-u1.5-model-config";

  U15ModelConfigStage(const vpipe::SessionContextIntf* session,
                      std::string                      id,
                      std::vector<vpipe::InEdge>       iports,
                      vpipe::FlexData                  config);

  const vpipe::StageSpec& spec() const noexcept override;

  // The file-static spec, for the plugin's registration call. A plugin
  // attaches its spec through VpipePluginContext::register_stage rather
  // than the in-tree VPIPE_REGISTER_SPEC macro, and that takes a pointer
  // with static storage duration.
  static const vpipe::StageSpec* stage_spec() noexcept;

  vpipe::FlexData resolved_config() const;

private:
  double      _cfg_scale = 4.0;
  double      _img_cfg_scale = 1.0;
  double      _timestep_shift = 3.0;
  double      _t_eps = 0.05;
  std::string _cfg_norm;
  double      _cfg_lo = 0.0;
  double      _cfg_hi = 1.0;
  std::string _init_noise;

  bool _has_cfg_scale = false;
  bool _has_img_cfg = false;
  bool _has_shift = false;
  bool _has_t_eps = false;
  bool _has_norm = false;
  bool _has_interval = false;
  bool _has_noise = false;
};

}  // namespace u15

#endif  // VPIPE_U15_MODEL_CONFIG_STAGE_H
