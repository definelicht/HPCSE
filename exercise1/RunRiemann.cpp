/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date September 2015

#include <chrono>     // Tools for timing
#include <cmath>      // std::sqrt, std::log
#include <functional> // std::function
#include <iomanip>    // std::setprecision
#include <iostream>
#include <string>     // std::stod, std::stoi

#include "riemann/RiemannSum.h"

using namespace hpcse;

int main(int argc, char const *argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: <start> <end> <number of steps> <number of cores ...>"
              << std::endl;
    return 1;
  }
  double begin = std::stod(argv[1]);
  double end = std::stod(argv[2]);
  int n = std::stoi(argv[3]);
  if (begin >= end) {
    std::cerr << "End must be higher than beginning of integration."
              << std::endl;
    return 1;
  }
  if (n < 1) {
    std::cerr << "Number of steps must be >=1." << std::endl;
    return 1;
  }
  double result;
  // Function hardcoded here but kept as an argument for the sake of potential
  // extension 
  std::function<double(double)> f(
      [](double x) { return std::sqrt(x) * std::log(x); });
  for (int i = 4; i < argc; ++i) {
    int nThreads = std::stoi(argv[i]);
    std::cout << "-- Running on " << nThreads << " threads.\n";
    double elapsed;
    if (nThreads == 1) {
      auto start = std::chrono::system_clock::now();
      result = RiemannSequential(f, begin, end, n);
      elapsed = 1e-6 *
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - start)
                    .count();
    } else {
      if (n % nThreads > 0) {
        std::cerr << "Number of steps mod number of threads must be zero."
                  << std::endl;
        return 1;
      }
      auto start = std::chrono::system_clock::now();
      result = RiemannParallel(f, begin, end, n, nThreads);
      elapsed = 1e-6 *
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - start)
                    .count();
    }
    std::cout << "Finished in " << std::setprecision(4) << std::fixed << elapsed
              << " seconds.\n"
              << "Result: " << std::setprecision(16) << result << "\n";
  }
  return 0;
}
