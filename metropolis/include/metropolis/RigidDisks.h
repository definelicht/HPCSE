#pragma once

#include <vector>

namespace hpcse {

std::vector<float> RigidDisks(unsigned nCores, unsigned nx, unsigned ny,
                              float l, float d0Factor,
                              unsigned stepsEquilibrium, unsigned steps,
                              unsigned nBins);

std::vector<float> RigidDisks(unsigned nx, unsigned ny, float l, float d0Factor,
                              unsigned stepsEquilibrium, unsigned steps,
                              unsigned nBins);

} // End namespace hpcse
