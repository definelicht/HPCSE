#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include "common/AlignedAllocator.h"
#include "common/Timer.h"
#include "lennardjones/LennardJones.h"

using namespace hpcse;

using ContainerType = typename LennardJones::ContainerType;

std::pair<ContainerType, ContainerType>
LoadParticles(const std::string &path, const size_t nParticles);

int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cerr << "Input path to data file.\n";
    return 1;
  }

  LennardJones lennardJones(0.1, 5.0);

  constexpr size_t nParticles = 1000;
  auto particles = LoadParticles(argv[1], nParticles);

  const size_t n = particles.first.size();

  const auto newPos = std::make_pair<float>(particles.first[n - 1] + 0.01,
                                            particles.second[n - 1] + 0.01);
  constexpr int nIterations = 100;

  Timer timer;
  timer.Start();
  float energyDiffScalar = 0;
  for (int i = 0; i < nIterations; ++i) {
    energyDiffScalar +=
        lennardJones.Diff(particles.first.cbegin(), particles.first.cend(),
                          particles.second.cbegin(), newPos);
  }
  double elapsedScalar = timer.Stop();
  std::cout << "-- Scalar\n  Result: " << energyDiffScalar
            << "\n  Elapsed: " << elapsedScalar << "\n";

  timer.Start();
  float energyDiffAutoVec = 0;
  for (int i = 0; i < nIterations; ++i) {
    energyDiffAutoVec += lennardJones.DiffAutoVec(
        particles.first.cbegin(), particles.first.cend(),
        particles.second.cbegin(), newPos);
  }
  double elapsedAutoVec = timer.Stop();
  std::cout << "-- Autovectorized\n  Result: " << energyDiffAutoVec
            << "\n  Elapsed: " << elapsedAutoVec
            << "\n  Speedup: " << elapsedScalar/elapsedAutoVec << "\n";

#ifdef __AVX__ 
  float energyDiffAvx = 0;
  timer.Start();
  for (int i = 0; i < nIterations; ++i) {
    energyDiffAvx +=
        lennardJones.DiffAvx(particles.first.cbegin(), particles.first.cend(),
                             particles.second.cbegin(), newPos);
  }
  double elapsedAvx = timer.Stop();
  std::cout << "-- AVX\n  Result: " << energyDiffAvx
            << "\n  Elapsed: " << elapsedAvx 
            << "\n  Speedup: " << elapsedScalar/elapsedAvx << "\n";
#endif

  return 0;
}

std::pair<ContainerType, ContainerType> LoadParticles(std::string const &path,
                                                      const size_t nParticles) {
  const size_t n =
      nParticles * nParticles + (nParticles - 1) * (nParticles - 1);
  auto buffer = std::make_unique<float[]>(2 * n);
  std::ifstream infile(path);
  infile.read(reinterpret_cast<char *>(buffer.get()), 2 * n * sizeof(float));
  auto output = std::make_pair(ContainerType(n), ContainerType(n));
  for (size_t iIn = 0, iOut = 0; iOut < n; iIn += 2, ++iOut) {
    output.first[iOut] = buffer[iIn];
    output.second[iOut] = buffer[iIn + 1];
  }
  std::cout << "Loaded " << n << " particles from " << path << "." << std::endl;
  return output;
}
