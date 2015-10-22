#include "diffusion/RandomWalk.h"
#include <cmath>
#include <random>
#include <omp.h>

namespace hpcse {

// Per-thread kernel.
std::pair<double, double>
RandomWalkKernel(unsigned iterations, float d, std::pair<float, float> start,
                 std::pair<float, float> xBounds,
                 std::pair<float, float> yBounds,
                 std::function<float(float, float)> boundaryCondition);

std::pair<float, float>
RandomWalk(unsigned iterations, float d, std::pair<float, float> const &start,
           std::pair<float, float> const &xBounds,
           std::pair<float, float> const &yBounds,
           std::function<float(float, float)> const &boundaryCondition) {
  return RandomWalk(0, iterations, d, start, xBounds, yBounds,
                    boundaryCondition);
}

std::pair<float, float>
RandomWalk(unsigned nThreads, unsigned iterations, float d,
           std::pair<float, float> const &start,
           std::pair<float, float> const &xBounds,
           std::pair<float, float> const &yBounds,
           std::function<float(float, float)> const &boundaryCondition) {
  if (nThreads == 0) {
    nThreads = omp_get_max_threads();
  }
  const unsigned iterationsPerThread = iterations / nThreads;
  const unsigned iterationsTail =
      iterations + iterationsPerThread * (1 - nThreads);
  double sum = 0;
  double sumOfSquares = 0;
  #pragma omp parallel num_threads(nThreads) reduction(+ : sum, sumOfSquares)
  {
    unsigned localIterations =
        static_cast<unsigned>(omp_get_thread_num()) < nThreads - 1
            ? iterationsPerThread
            : iterationsTail;
    auto threadResult = RandomWalkKernel(localIterations, d, start,
                                         xBounds, yBounds, boundaryCondition);
    sum = threadResult.first;
    sumOfSquares = threadResult.second;
  }
  return {sum / iterations,
          (sumOfSquares - (sum * sum) / iterations) / (iterations - 1)};
}

std::pair<double, double>
RandomWalkKernel(const unsigned iterations, const float d,
                 const std::pair<float, float> start,
                 const std::pair<float, float> xBounds,
                 const std::pair<float, float> yBounds,
                 const std::function<float(float, float)> boundaryCondition) {
  std::mt19937 rng(std::random_device{}());
  static constexpr float kPi = 3.1415926535897932;
  std::uniform_real_distribution<float> distribution(0.0, kPi);
  double sum = 0;
  double sumOfSquares = 0;
  for (unsigned i = 0; i < iterations; ++i) {
    float x = start.first;
    float y = start.second;
    while (x >= xBounds.first && x <= xBounds.second && y >= yBounds.first &&
           y <= yBounds.second) {
      x += d*std::cos(distribution(rng));
      y += d*std::sin(distribution(rng));
    }
    float g = boundaryCondition(x, y);
    sum += g;
    sumOfSquares += g * g;
  }
  return {sum, sumOfSquares};
};

} // End namespace hpcse
