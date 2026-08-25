#ifndef VPIPE_U15_CATALOG_H
#define VPIPE_U15_CATALOG_H

#include "stages/model-catalog.h"

#include <vector>

namespace u15 {

// The catalogue entries this plugin contributes, so the checkpoint is
// discoverable and fetchable the way a built-in model is.
std::vector<vpipe::ModelCatalogEntry> catalog_entries();

}  // namespace u15

#endif  // VPIPE_U15_CATALOG_H
