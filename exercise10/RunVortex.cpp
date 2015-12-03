#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "common/Mpi.h"
#include "common/Timer.h"
#include "vortex/Vortex.h"

using namespace hpcse;

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr
        << "Usage: <number of vortices> <timestep> <times to record>...\n";
    return 1;
  }
  mpi::Context context;
  const int nParticles = std::stoi(argv[1]);
  const float timestep = std::stof(argv[2]);
  std::vector<float> timeToRecord;
  for (int i = 3; i < argc; ++i) {
    timeToRecord.emplace_back(std::stof(argv[i]));
  }
  const int nIterations = timeToRecord.back() / timestep + 1;
  Timer timer;
  if (mpi::rank() == 0) {
    std::cout << "Running " << nParticles << " particles for " << nIterations
              << " iterations..." << std::flush;
  }
  timer.Start();
  auto snapshots = Vortex(nParticles, 1, timestep, timeToRecord);
  double elapsed = timer.Stop();
  if (mpi::rank() == 0) {
    std::ofstream benchmarkFile("benchmarks.txt",
                                std::ofstream::out | std::ofstream::app);
    benchmarkFile << mpi::size() << "," << nParticles << "," << nIterations
                  << "," << elapsed << "\n";
    std::cout << " Finished in " << elapsed << " seconds.\n";
    std::ofstream snapshotFile("snapshots.txt",
                               std::ofstream::out | std::ofstream::trunc);
    for (size_t i = 0; i < snapshots.size(); ++i) {
      snapshotFile << timeToRecord[i];
      for (auto &j : snapshots[i]) {
        snapshotFile << "," << j;
      }
      snapshotFile << "\n";
    }
  }
  return 0;
}
