#pragma once

#include "diffusion/Diffusion.h"

namespace hpcse {

std::vector<Grid_t> DiffusionMPI(const unsigned dim, const float d,
                                 const float dt,
                                 std::vector<float> const &snapshots);

} // End namespace hpcse
