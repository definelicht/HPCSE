/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date October 2015

#include "diffusion/Diffusion.h"

namespace hpcse {

std::vector<Grid_t> DiffusionSequential(unsigned dim, float d, float dt,
                                        std::vector<float> const &snapshots);

std::vector<Grid_t> DiffusionParallel(unsigned nCores, unsigned dim, float d,
                                      float dt,
                                      std::vector<float> const &snapshots);

std::vector<Grid_t> Diffusion(unsigned dim, float d, float dt,
                              std::vector<float> const &snapshots) {
  return DiffusionSequential(dim, d, dt, snapshots);
}

std::vector<Grid_t> Diffusion(unsigned nCores, unsigned dim, float d, float dt,
                              std::vector<float> const &snapshots) {
  if (nCores > 1) {
    return DiffusionParallel(nCores, dim, d, dt, snapshots);
  } else {
    return DiffusionSequential(dim, d, dt, snapshots);
  }
}

} // End namespace hpcse
