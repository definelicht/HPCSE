/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date October 2015

#include "diffusion/DiffusionJob.h"

namespace hpcse {

DiffusionJob::DiffusionJob(const unsigned rows, const unsigned cols,
                           const int rowOffset)
    : grid_(rows, Row_t(cols)), buffer_(grid_) {
  const float halfDim = (cols >> 1) - 0.5;
  const float boundSquared = (cols * cols) >> 4;
  const int dimSigned = cols;
  for (int i = 0, iEnd = rows; i < iEnd; ++i) {
    float dx = (rowOffset+i) - halfDim;
    for (int j = 0; j < dimSigned; ++j) {
      float dy = j - halfDim;
      grid_[i][j] = dx*dx + dy*dy < boundSquared;
    }
  }
}

std::vector<Grid_t> DiffusionJob::RunDiffusion(
    const std::shared_ptr<DiffusionJob> above,
    const std::shared_ptr<DiffusionJob> below, const float dt,
    const std::vector<float> snapshots, Barrier &barrier) {
  std::vector<Grid_t> output(
      snapshots.size(), Grid_t(grid_.size(), Row_t(grid_[0].size())));
  float t = 0;
  auto snapshotItr = snapshots.cbegin();
  auto snapshotEnd = snapshots.cend();
  auto outputItr = output.begin();
  float ds = 2./grid_[0].size();
  const float factor = dt/(ds*ds);
  const int iEnd = grid_.size()-1;
  const int jEnd = grid_[0].size()-1;
  while (true) {
    if (t >= *snapshotItr) {
      *outputItr++ = grid_;
      if (++snapshotItr == snapshotEnd) break; 
    }
    // Top row
    if (above != nullptr) {
      auto aboveRow = above->LastRow();
      for (int j = 1; j < jEnd; ++j) {
        buffer_[0][j] =
            grid_[0][j] +
            factor * (aboveRow[j] + grid_[0][j - 1] - 4 * grid_[0][j] +
                      grid_[0][j + 1] + grid_[1][j]);
      }
    }
    // Internal rows
    for (int i = 1; i < iEnd; ++i) {
      for (int j = 1; j < jEnd; ++j) {
        buffer_[i][j] =
            grid_[i][j] +
            factor * (grid_[i - 1][j] + grid_[i][j - 1] - 4 * grid_[i][j] +
                      grid_[i][j + 1] + grid_[i + 1][j]);
      }
    }
    // Bottom row
    if (below != nullptr) {
      auto belowRow = below->FirstRow();
      for (int j = 1; j < jEnd; ++j) {
        buffer_[iEnd][j] =
            grid_[iEnd][j] +
            factor * (grid_[iEnd - 1][j] + grid_[iEnd][j - 1] -
                      4 * grid_[iEnd][j] + grid_[iEnd][j + 1] + belowRow[j]);
      }
    }
    t += dt;
    // Synchronize before swapping frame buffer
    barrier.Synchronize();
    grid_.swap(buffer_);
    // Synchronize again to make sure all swaps have happened
    barrier.Synchronize();
  }
  return output;
}

} // End namespace hpcse
