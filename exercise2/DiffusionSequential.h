#pragma once

#include <vector>
#include "Diffusion.h"

std::vector<Grid_t> DiffusionSequential(unsigned dim, const float dt,
                                        std::vector<float> const &snapshots);
