#pragma once

#include "diffusion/Diffusion.h"

namespace hpcse {

std::vector<Grid_t> DiffusionRows(const unsigned dim, const float d,
                                  const float dt,
                                  std::vector<float> const &snapshots);

std::vector<Grid_t> DiffusionGrid(const unsigned dim, const float d,
                                  const float dt,
                                  std::vector<float> const &snapshots);

} // End namespace hpcse
