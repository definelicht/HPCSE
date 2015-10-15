#include <iostream>
#include <numeric>
#include <fstream>
#include <string> // std::stof
#include "diffusion/Diffusion.h"

using namespace hpcse;

int main(const int argc, char const *const argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: <number of cores> <number of samples to record> "
                 "<output file>"
              << std::endl;
    return 1;
  }
  unsigned nCores = std::stof(argv[1]);
  const unsigned nSamples = std::stoi(argv[2])+1;
  std::string path(argv[3]);
  const float tPerSample = 10. / (nSamples-1);
  std::vector<float> samples;
  for (unsigned i = 0; i < nSamples; ++i) {
    samples.emplace_back(i*tPerSample);
  }
  // std::vectorception...
  const std::vector<float> ds{1, 2, 5};
  std::vector<std::vector<Grid_t>> results;
  for (auto &d : ds) {
    std::cout << "Running diffusion for D = " << d << "...\n";
    results.emplace_back(Diffusion(nCores, 128, d, 1e-5, samples));
  }
  std::vector<std::vector<float>> n(3, std::vector<float>(nSamples));
  std::vector<std::vector<float>> mu(n);
  for (size_t i = 0, iEnd = ds.size(); i < iEnd; ++i) {
    std::cout << "Computing N and mu^2 for D = " << ds[i] << "...\n";
    #pragma omp parallel for
    for (size_t j = 0; j < nSamples; ++j) {
      float nSum = 0;
      float muSum = 0;
      for (size_t x = 0, xEnd = results[i][j].size(); x < xEnd; ++x) {
        for (size_t y = 0, yEnd = results[i][j][x].size(); y < yEnd; ++y) {
          nSum  += results[i][j][x][y];
          muSum += results[i][j][x][y]*(x*x + y*y);
        }
      } 
      n[i][j] = nSum;
      mu[i][j] = muSum;
    }
  }
  std::ofstream outfile(path);
  for (size_t i = 0, iEnd = ds.size(); i < iEnd; ++i) {
    for (size_t j = 0; j < nSamples; ++j) {
      outfile << ds[i] << "," << samples[j] << "," << n[i][j] << "," << mu[i][j]
              << "\n";
    }
  }
  return 0;
}
