/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date October 2015

#pragma once

#include <vector>
#include "diffusion/Diffusion.h"

namespace hpcse {

std::vector<Grid_t> DiffusionSequential(unsigned dim, const float dt,
                                        std::vector<float> const &snapshots);

} // End namespace hpcse
