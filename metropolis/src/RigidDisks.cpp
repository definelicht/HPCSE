#include "metropolis/RigidDisks.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <random>
#include <utility>

namespace hpcse {

std::vector<std::pair<float, float>>
Initialize(unsigned nx, unsigned ny, float l);

std::vector<float> RigidDisks(const unsigned nThreads, const unsigned nx,
                              const unsigned ny, const float l,
                              const float diameterFactor,
                              const unsigned stepsEquilibrium,
                              const unsigned steps, const unsigned nBins) {

  const unsigned nTot = nx*ny;
  const float lDivNx = l/nx;
  const float diameter = diameterFactor*lDivNx;
  const float diameterSquared = diameter*diameter;
  const float alpha = lDivNx - diameter;
  static constexpr float kSqrtTwoInv = std::sqrt(0.5);
  const float rMax = kSqrtTwoInv*l;
  const float rMaxSquared = rMax*rMax;
  const float halfL = 0.5*l;
  auto disks = Initialize(nx, ny, l);

  std::random_device seeder;
  std::mt19937 rngIndex(seeder());
  std::mt19937 rngDisplacement(seeder());
  std::uniform_int_distribution<size_t> uniformIndex(0, nTot-1);
  std::uniform_real_distribution<float> uniformDisplacement(-alpha, alpha);

  auto distSquared = [&halfL](std::pair<float, float> const &a,
                              std::pair<float, float> const &b) {
    float dx = std::fabs(a.first - b.first);
    float dy = std::fabs(a.second - b.second);
    dx = dx <= halfL ? dx : dx - halfL;
    dy = dy <= halfL ? dy : dy - halfL;
    return dx * dx + dy * dy;
  };

  auto doStep = [l, nTot, diameterSquared, &distSquared](
      std::vector<std::pair<float, float>> &disks, std::mt19937 &rngIndex,
      std::mt19937 &rngDisplacement,
      std::uniform_int_distribution<size_t> &uniformIndex,
      std::uniform_real_distribution<float> &uniformDisplacement) {
    const auto iMoved = uniformIndex(rngIndex);
    std::pair<float, float> newPos = disks[iMoved];
    newPos.first += uniformDisplacement(rngDisplacement);
    newPos.second += uniformDisplacement(rngDisplacement);
    newPos.first = newPos.first >= 0
                       ? (newPos.first < l ? newPos.first : newPos.first - l)
                       : newPos.first + l;
    newPos.second = newPos.second >= 0
                       ? (newPos.second < l ? newPos.second : newPos.second - l)
                       : newPos.second + l;
    // TODO: replace with log(n) algorithm (kd-tree?)
    for (unsigned i = 0; i < nTot; ++i) {
      if (i != iMoved) {
        if (distSquared(newPos, disks[i]) < diameterSquared) { 
          return false;
        }
      }
    }
    disks[iMoved] = newPos;
    return true;
  };
  auto doStepSequential = std::bind(
      doStep, std::ref(disks), std::ref(rngIndex), std::ref(rngDisplacement),
      std::ref(uniformIndex), std::ref(uniformDisplacement));

  // Run to equilibrium
  for (unsigned i = 0; i < stepsEquilibrium; ++i) {
    for (unsigned j = 0; j < nTot; /* Only increment when successful */) {
      j += doStepSequential();
    }
  }

  // Run measurements
  const float binFactor = nBins / (rMaxSquared - diameterSquared);
  auto runMeasurements = [nBins, nTot, distSquared, diameterSquared, binFactor](
      const unsigned steps, std::vector<std::pair<float, float>> &disks,
      std::function<bool(void)> const &doStepLocal) {
    std::vector<float> histogram(nBins, 0);
    const auto jEnd = disks.cend() - 1;
    const auto kEnd = disks.cend();
    for (unsigned i = 0; i < steps; ++i) {
      for (unsigned j = 0; j < nTot; /* Only increment when successful */) {
        j += doStepLocal();
      }
      for (auto j = disks.cbegin(); j != jEnd; ++j) {
        // Compare only to other disks not already paired
        for (auto k = j + 1; k != kEnd; ++k) {
          const size_t index =
              (distSquared(*j, *k) - diameterSquared) * binFactor;
          histogram[index] += 1;
        }
      }
    }
    return histogram;
  };
  std::vector<float> histogram;
  if (nThreads > 1) {
    std::vector<std::future<std::vector<float>>> futures;
    for (unsigned i = 0; i < nThreads; ++i) {
      futures.emplace_back(std::async(std::launch::async, [&]() {
        // Copy disks from each thread
        std::vector<std::pair<float, float>> localDisks(disks);
        // Initialize local RNGs
        std::random_device seeder;
        std::mt19937 rngIndex(seeder());
        std::mt19937 rngDisplacement(seeder());
        std::uniform_int_distribution<size_t> uniformIndex(0, nTot-1);
        std::uniform_real_distribution<float> uniformDisplacement(-alpha,
                                                                  alpha);
        auto doStepLocal =
            std::bind(doStep, std::ref(localDisks), std::ref(rngIndex),
                      std::ref(rngDisplacement), std::ref(uniformIndex),
                      std::ref(uniformDisplacement));
        return runMeasurements((i + 1) * steps / nThreads -
                                   i * steps / nThreads,
                               localDisks, doStepLocal);
      }));
    }
    histogram = futures[0].get();
    for (unsigned i = 1; i < nThreads; ++i) {
      auto localHistogram = futures[i].get();
      for (unsigned j = 0; j < nBins; ++j) {
        histogram[j] += localHistogram[j];
      }
    }
  } else {
    histogram = runMeasurements(steps, disks, doStepSequential);
  }

  // Average over measurements
  const float stepsInv = 1. / steps;
  std::for_each(histogram.begin(), histogram.end(),
                [&stepsInv](float &x) { x *= stepsInv; });

  return histogram;
}

std::vector<float> RigidDisks(unsigned nx, unsigned ny, float l,
                              float diameterFactor, unsigned stepsEquilibrium,
                              unsigned steps, unsigned nBins) {
  return RigidDisks(0, nx, ny, l, diameterFactor, stepsEquilibrium, steps,
                    nBins);
}

std::vector<std::pair<float, float>>
Initialize(const unsigned nx, const unsigned ny, const float l) {
  std::vector<std::pair<float, float>> disks(nx*ny);
  const float distX = l / nx;
  const float distY = l / ny;
  for (unsigned i = 0; i < nx;  ++i) {
    for (unsigned j = 0; j < ny; ++j) {
      disks[i*ny + j] = std::pair<float, float>(i*distX, j*distY);
    }
  }
  return disks;
}

} // End namespace hpcse
