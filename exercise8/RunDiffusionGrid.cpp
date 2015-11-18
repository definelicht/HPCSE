#include <cassert>
#include <cmath>
#include <iostream>
#include "common/Mpi.h"

using namespace hpcse;

int main(int argc, char *argv[]) {
  mpi::Context context(&argc, &argv);
  const int nHorizontal = std::sqrt(mpi::size());
  const int nVertical = mpi::size()/nHorizontal;
  const int nThreads = nHorizontal*nVertical;
  if (nThreads < mpi::size()) {
    if (mpi::rank() == 0) {
      std::cerr << "MPI must be run with an even number of ranks to form a "
                   "cartesian grid.\n";
    }
    return 1;
  }
  mpi::CartesianGrid<2> grid({{nVertical, nHorizontal}}, false);
  std::cout << "Rank " << mpi::rank() << " has coordinates {" << grid.row()
            << ", " << grid.col() << "} / {" << grid.rowMax() << ", "
            << grid.colMax() << "}\n";
  return 0;
}
