/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date September 2015

#include "riemann/RiemannSum.h"

#include <chrono>     // Tools for timing
#include <cmath>      // std::sqrt, std::log
#include <functional> // std::function
#include <future>
#include <iomanip>    // std::setprecision
#include <iostream>
#include <string>     // std::stod, std::stoi
#include <vector>

namespace hpcse {

double RiemannSequential(std::function<double(double)> const &f, double begin,
                         double end, const int n) {
  const double step = (end - begin) / static_cast<double>(n);
  double sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += f(begin + static_cast<double>(i) * step + 0.5 * step) * step;
  }
  return sum;
}

double RiemannParallel(std::function<double(double)> const &f, double begin,
                       double end, const int n, const int nThreads) {
  double threadStep = (end - begin) / static_cast<double>(nThreads);
  int threadN = n / nThreads;
  std::vector<std::future<double>> futures;
  for (int i = 0; i < nThreads; ++i) {
    futures.push_back(std::async(std::launch::async, RiemannSequential, f,
                                 begin + i * threadStep,
                                 begin + (i + 1) * threadStep, threadN));
  }
  double sum = 0;
  for (auto &f : futures) {
    sum += f.get();
  }
  return sum;
}

} // End namespace hpcse
