#pragma once

#include <vector>
#include "Diffusion.h"

std::vector<Grid_t> DiffusionParallel(unsigned nCores, unsigned dim, float dt,
                                      std::vector<float> const &snapshots);
