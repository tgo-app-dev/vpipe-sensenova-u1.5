#include "u15-catalog.h"
#include "u15-config.h"

#include <vector>

using vpipe::ModelCatalogEntry;

namespace u15 {

std::vector<ModelCatalogEntry>
catalog_entries()
{
  std::vector<ModelCatalogEntry> v;

  // The whole repo is one entry because the parts are useless apart:
  // there is no separate text encoder or VAE to fetch, and the tokenizer
  // files sit beside the shards. `files` is left empty -- the repo is
  // 50.2 GB and ALL of it is needed, so pinning a subset would only
  // create a way to fetch something that cannot run.
  //
  // The size is what it is because the generation expert ships F32 while
  // the understanding expert ships BF16; this port converts the former
  // down at load, so the RESIDENT set is ~32 GB rather than 50.
  ModelCatalogEntry e;
  e.family        = "SenseNova-U";
  e.version       = "1.5";
  e.param_class   = "8B-MoT";
  e.variant       = "bf16 (text-to-image + editing)";
  e.hf_path       = "sensenova/SenseNova-U1.5-8B-MoT";
  e.model_type    = kFamily;
  e.name          = "SenseNova-U1.5-8B-MoT";
  e.inputs        = {"text", "image"};
  e.outputs       = {"image"};
  e.weight_format = "safetensors";
  v.push_back(std::move(e));

  return v;
}

}  // namespace u15
