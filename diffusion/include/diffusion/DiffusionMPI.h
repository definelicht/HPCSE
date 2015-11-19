#pragma once

#include "diffusion/Diffusion.h"

namespace hpcse {

std::vector<Grid_t> DiffusionRows(unsigned gridDim, float diffusionConstant,
                                  float timeStep,
                                  std::vector<float> const &timesToRecord);

std::vector<Grid_t> DiffusionGrid(unsigned gridDim, float diffusionConstant,
                                  float timeStep,
                                  std::vector<float> const &timesToRecord);

} // End namespace hpcse
