/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date October 2015

#include "diffusion/Diffusion.h"

#include <algorithm> // std::sort
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>    // std::stoi, std::stof
#include <vector>

using namespace hpcse;

int main(int argc, char const *argv[]) {
  if (argc < 7) {
    std::cerr << "Usage: <cores> <diffusion constant> <grid dimension> "
                 "<timestep> <output file> <time for "
                 "snapshot...>"
              << std::endl;
    return 1;
  }
  unsigned nCores = std::stoi(argv[1]);
  float d = std::stof(argv[2]);
  unsigned dim = std::stoi(argv[3]);
  float dt = std::stof(argv[4]);
  std::ofstream outputStream(argv[5]);
  assert(outputStream.is_open());
  std::vector<float> snapshots;
  for (int i = 6; i < argc; ++i) {
    snapshots.push_back(std::stof(argv[i]));
  }
  std::sort(snapshots.begin(), snapshots.end());
  std::cout << "Running on " << nCores << " core(s) for " << dim << "x" << dim
            << " grid with timestep " << dt << " for "
            << *(snapshots.cend() - 1) / dt << " iterations...\n";
  auto start = std::chrono::system_clock::now();
  auto results = Diffusion(nCores, dim, d, dt, snapshots);
  auto elapsed = 1e-6 *
                 std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::system_clock::now() - start)
                     .count();
  std::cout << "Finished in " << elapsed << " seconds." << std::endl;
  for (auto &grid : results) {
    for (auto &row : grid) {
      std::copy(row.cbegin(), row.cend()-1,
                std::ostream_iterator<float>(outputStream, ","));
      outputStream << *(row.cend()-1) << "\n";
    }
    outputStream << "\n";
  }
  return 0;
}


