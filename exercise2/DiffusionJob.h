#pragma once

#include "Barrier.h"
#include "Diffusion.h"

class DiffusionJob {

public:
  DiffusionJob(unsigned rows, unsigned cols, int rowOffset);

  inline Row_t const &FirstRow() const;

  inline Row_t const &LastRow() const;

  std::vector<Grid_t> RunDiffusion(std::shared_ptr<DiffusionJob> above,
                                   std::shared_ptr<DiffusionJob> below,
                                   float dt, std::vector<float> snapshots,
                                   Barrier &barrier);

  static inline std::shared_ptr<DiffusionJob>
  Allocate(unsigned cols, int rowBegin, int rowEnd);

private:
  Grid_t grid_, buffer_;
};

Row_t const& DiffusionJob::FirstRow() const {
  return *grid_.cbegin();
}

Row_t const& DiffusionJob::LastRow() const {
  return *(grid_.cend()-1);
}

std::shared_ptr<DiffusionJob>
DiffusionJob::Allocate(unsigned cols, int rowBegin, int rowEnd) {
  return std::make_shared<DiffusionJob>(cols, rowBegin, rowEnd);
}
