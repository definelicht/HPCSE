/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date October 2015

#include "diffusion/DiffusionSequential.h"

namespace hpcse {

Grid_t InitializeGrid(const unsigned dim);

void Diffuse(const float factor, Grid_t const &grid, Grid_t &buffer);

std::vector<Grid_t> DiffusionSequential(unsigned dim, const float d,
                                        const float dt,
                                        std::vector<float> const &snapshots) {
  auto grid = InitializeGrid(dim);
  std::vector<Grid_t> output(
      snapshots.size(), Grid_t(grid.size(), Grid_t::value_type(grid.size())));
  float t = 0;
  float ds = 2./dim;
  const float factor = d*dt/(ds*ds);
  auto snapshotItr = snapshots.cbegin();
  auto snapshotEnd = snapshots.cend();
  auto outputItr = output.begin();
  // Frame buffer to be swapped between iterations
  Grid_t buffer(grid);
  while (true) {
    if (t >= *snapshotItr) {
      *outputItr++ = grid;
      if (++snapshotItr == snapshotEnd) break; 
    }
    Diffuse(factor, grid, buffer);
    grid.swap(buffer);
    t += dt;
  }
  return output;
}

Grid_t InitializeGrid(const unsigned dim) {
  Grid_t grid(dim, Grid_t::value_type(dim));
  const float halfDim = (dim>>1) - 0.5;
  const float boundSquared = (dim*dim)>>4;
  const int dimSigned = dim;
  #pragma omp parallel for
  for (int i = 0; i < dimSigned; ++i) {
    float dx = i - halfDim;
    for (int j = 0; j < dimSigned; ++j) {
      float dy = j - halfDim;
      grid[i][j] = dx*dx + dy*dy < boundSquared;
    }
  }
  return grid;
}

void Diffuse(const float factor, Grid_t const &grid, Grid_t &buffer) {
  const int iEnd = grid.size()-1;
  #pragma omp parallel for
  for (int i = 1; i < iEnd; ++i) {
    for (int j = 1; j < iEnd; ++j) {
      buffer[i][j] =
          grid[i][j] +
          factor * (grid[i - 1][j] + grid[i][j - 1] - 4 * grid[i][j] +
                    grid[i][j + 1] + grid[i + 1][j]);
    }
  }
}

} // End namespace hpcse
