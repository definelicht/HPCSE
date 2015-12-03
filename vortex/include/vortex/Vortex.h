#pragma once

#include <vector>

namespace hpcse {

std::vector<std::vector<double>> Vortex(int nParticlesTotal, double lineLength,
                                        float timestep,
                                        std::vector<float> const &timeToRecord);

} // End namespace hpcse
