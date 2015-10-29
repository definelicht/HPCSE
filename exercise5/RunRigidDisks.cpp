#include <fstream>
#include <iostream>
#include <string>
#include "metropolis/RigidDisks.h"

using namespace hpcse;

int main(int argc, char const *argv[]) {
  if (argc < 9) {
    std::cerr
        << "Usage: <number of cores> <number of particles in x> <number "
           "of particles in y> <square size> <diameter factor> <equilibrium "
           "steps> <steps> <number of bins> [<output file>]"
        << std::endl;
    return 1;
  }
  unsigned nCores = std::stoi(argv[1]);
  unsigned nDisksX = std::stoi(argv[2]);
  unsigned nDisksY = std::stoi(argv[3]);
  float l = std::stof(argv[4]);
  float d0Factor = std::stof(argv[5]);
  unsigned stepsEquilibrium = std::stoi(argv[6]);
  unsigned steps = std::stoi(argv[7]);
  unsigned nBins = std::stoi(argv[8]);
  std::string outPath("");
  if (argc >= 10) {
    outPath = argv[9]; 
  }
  auto histogram = RigidDisks(nCores, nDisksX, nDisksY, l, d0Factor,
                              stepsEquilibrium, steps, nBins);
  if (argc < 10) {
    std::cout << "Resulting histogram:\n";
    for (auto &b : histogram) {
      std::cout << b << " ";
    }
    std::cout << "\n";
  } else {
    std::ofstream outFile(outPath);
    outFile << (*histogram.cbegin());
    for (auto b = histogram.cbegin() + 1, bEnd = histogram.cend(); b != bEnd;
         ++b) {
      outFile << "," << *b;
    }
  }
  return 0;
}
