/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date October 2015

#pragma once

#include <vector>

namespace hpcse {

using Row_t = std::vector<float>;

using Grid_t = std::vector<Row_t>;

std::vector<Grid_t> Diffusion(unsigned dim, float d, float dt,
                              std::vector<float> const &snapshots);

std::vector<Grid_t> Diffusion(unsigned nCores, unsigned dim, float d, float dt,
                              std::vector<float> const &snapshots);

} // End namespace hpcse
