/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date October 2015

#pragma once

#include <vector>

namespace hpcse {

using Row_t = std::vector<float>;

using Grid_t = std::vector<Row_t>;

inline __attribute__((always_inline)) float DiffusionKernel(const float factor,
                                                            Grid_t const &grid,
                                                            const int i,
                                                            const int j) {
  return grid[i][j] +
         factor * (grid[i - 1][j] + grid[i][j - 1] - 4 * grid[i][j] +
                   grid[i][j + 1] + grid[i + 1][j]);
}

} // End namespace hpcse
