#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "common/Mpi.h"
#include "common/Timer.h"
#include "diffusion/DiffusionMPI.h"

using namespace hpcse;

int main(int argc, char *argv[]) {

  mpi::Context context(argc, argv);

  if ((mpi::size() & 0x1) == 1) {
    if (mpi::rank() == 0) {
      std::cerr << "MPI must be run with an even number of ranks to form a "
                   "cartesian grid.\n";
    }
    return 1;
  }

  // Retrieve arguments
  if (argc < 6) {
    if (mpi::rank() == 0) {
      std::cerr << "Usage: <diffusion constant> <grid dimension> "
                   "<timestep> <output file> <time for "
                   "snapshot...>"
                << std::endl;
    }
    return 1;
  }
  float d = std::stof(argv[1]);
  unsigned dim = std::stoi(argv[2]);
  float dt = std::stof(argv[3]);
  std::string outPath(argv[4]);
  std::vector<float> timeToRecord;
  for (int i = 5; i < argc; ++i) {
    timeToRecord.push_back(std::stof(argv[i]));
  }

  Timer timer;
  auto snapshots = DiffusionGrid(dim, d, dt, timeToRecord);
  double elapsed = timer.Stop();

  if (mpi::rank() == 0) {
    std::cout << "Finished in " << elapsed << " seconds.\n";
    std::ofstream outputStream(outPath);
    assert(outputStream.is_open());
    for (auto &grid : snapshots) {
      for (auto &row : grid) {
        std::copy(row.cbegin(), row.cend()-1,
                  std::ostream_iterator<float>(outputStream, ","));
        outputStream << *(row.cend()-1) << "\n";
      }
      outputStream << "\n";
    }
  }

  return 0;
}
