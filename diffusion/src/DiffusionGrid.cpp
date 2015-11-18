#include <cmath>
#include "common/Timer.h"
#include "common/Mpi.h"
#include "diffusion/Diffusion.h"
#include "diffusion/DiffusionMPI.h"

namespace hpcse {

std::vector<Grid_t> DiffusionGrid(const unsigned dim, const float d,
                                  const float dt,
                                  std::vector<float> const &snapshots) {

  const int nHorizontal = std::sqrt(mpi::size());
  const int nVertical = mpi::size()/nHorizontal;
  mpi::CartesianGrid<2> grid({{nVertical, nHorizontal}}, false);
}

} // End namespace hpcse
