#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "common/Mpi.h"
#include "vortex/Vortex.h"

namespace hpcse {

std::vector<std::vector<double>>
Vortex(const int nParticlesTotal, const double lineLength, const float timestep,
       std::vector<float> const &timeToRecord) {

  // MPI range initialization
  const int mpiRank = mpi::rank();
  const int mpiSize = mpi::size();
  std::vector<int> beginAll(mpiSize);
  std::vector<int> endAll(mpiSize);
  std::vector<int> nParticlesAll(mpiSize);
  for (int i = 0; i < mpiSize; ++i) {
    beginAll[i] = nParticlesTotal * i / mpiSize;
    endAll[i] = nParticlesTotal * (i + 1) / mpiSize;
    nParticlesAll[i] = endAll[i] - beginAll[i];
  }
  const int begin = beginAll[mpiRank];
  const int end = endAll[mpiRank];
  const int nParticles = nParticlesAll[mpiRank];

  // Domain initialization
  std::vector<double> allPositions(nParticlesTotal);
  std::vector<double> allVelocities(nParticlesTotal);
  std::vector<double> allStrengths(nParticlesTotal);
  const double spacing = lineLength / nParticlesTotal;
  for (int i = 0; i < nParticlesTotal; ++i) {
    allPositions[i] = -0.5 + (i + 0.5) * spacing;
    // Gamma(x) = -d/dx(sqrt(1 - (x/0.5)^2)) = 4x / sqrt(1 - 4x^2)
    allStrengths[i] = spacing * 4 * allPositions[i] /
                      (std::sqrt(1 - 4 * allPositions[i] * allPositions[i]) +
                       std::numeric_limits<double>::epsilon());
  }

  // Output
  const int nSnapshots = timeToRecord.size();
  std::vector<std::vector<double>> positionSnapshots;
  if (mpiRank == 0) {
    positionSnapshots = std::vector<std::vector<double>>(
        nSnapshots, std::vector<double>(nParticles));
  }

  // Main loop
  float currentTime = 0;
  auto recordItr = timeToRecord.cbegin();
  auto outputItr = positionSnapshots.begin();
  const auto recordItrEnd = timeToRecord.cend(); 
  const auto allPosBegin = allPositions.begin();
  const auto posBegin = allPosBegin + begin;
  const auto posEnd = allPosBegin + end;
  std::vector<double> velocityBuffer(nParticles * (mpiSize - 1));
  const auto velBuffBegin = velocityBuffer.begin();
  const auto allVelBegin = allVelocities.begin();
  const auto allVelEnd = allVelocities.end();
  std::vector<MPI_Request> sendRequests(mpiSize - 1);
  std::vector<MPI_Request> receiveRequests(mpiSize - 1);
  while (true) {
    if (currentTime >= *recordItr) {
      if (mpiRank == 0) {
        std::copy(posBegin, posEnd, outputItr->begin());
      }
      if (++recordItr == recordItrEnd) {
        break;
      }
      ++outputItr;
    }
    // Compute local contributions to all global velocities
    std::fill(allVelBegin, allVelEnd, 0);
    for (int i = 0; i < nParticlesTotal; ++i) {
      for (int j = begin; j < end; ++j) {
        if (i != j) {
          allVelocities[i] +=
              allStrengths[j] / (allPositions[i] - allPositions[j]);
        }
      }
    }
    // Start sending velocities to all other ranks
    for (int i = 0, iSend = 0, iRecv = mpiSize - 2; i < mpiSize; ++i) {
      if (i != mpiRank) {
        sendRequests[iSend] = mpi::SendAsync(allVelBegin + beginAll[i],
                                             allVelBegin + endAll[i], i);
        receiveRequests[iRecv] =
            mpi::ReceiveAsync(velBuffBegin + iSend * nParticles,
                              velBuffBegin + (iSend + 1) * nParticles, i);
        ++iSend;
        --iRecv;
      }
    }
    // Receive and update velocity from one process at a time
    for (int i = 0, iMax = receiveRequests.size(); i < iMax; ++i) {
      mpi::Wait(receiveRequests[i]);
      for (int j = begin, jBuff = i * nParticles; j < end; ++j, ++jBuff) {
        allVelocities[j] += velocityBuffer[jBuff];
      }
    }
    // Update positions
    for (int i = begin; i < end; ++i) {
      constexpr double twoPiInv = 0.1591549430918953;
      allPositions[i] += twoPiInv * timestep * allVelocities[i];
    }
    // Get updated positions
    mpi::GatherAll(posBegin, posEnd, allPosBegin, nParticlesAll, beginAll);
    // Finalize sends
    mpi::WaitAll(sendRequests);
    currentTime += timestep;
  }

  return positionSnapshots;
}

} // End namespace hpcse
