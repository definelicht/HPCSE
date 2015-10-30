#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "diffusion/DiffusionMPI.h"

using namespace hpcse;

int main(int argc, char const *argv[]) {
  assert(MPI_Init(nullptr, nullptr) == MPI_SUCCESS);
  if (argc < 6) {
    std::cerr << "Usage: <diffusion constant> <grid dimension> "
                 "<timestep> <output file> <time for "
                 "snapshot...>"
              << std::endl;
    return 1;
  }
  float d = std::stof(argv[1]);
  unsigned dim = std::stoi(argv[2]);
  float dt = std::stof(argv[3]);
  std::string outPath(argv[4]);
  std::vector<float> timeToRecord;
  for (int i = 5; i < argc; ++i) {
    timeToRecord.push_back(std::stof(argv[i]));
  }
  int rank, nRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
  if (rank == 0) {
    std::cout << "Running on " << nRanks << " thread(s) for " << dim << "x" << dim
              << " grid with timestep " << dt << " for "
              << *(timeToRecord.cend() - 1) / dt << " iterations...\n";
  }
  auto start = std::chrono::system_clock::now();
  auto snapshots = DiffusionMPI(dim, d, dt, timeToRecord);
  auto elapsedInner = 1e-6 *
                      std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::system_clock::now() - start)
                          .count();
  if (rank != 0) {
    for (auto &s : snapshots) {
      for (auto row = s.cbegin()+1, rowEnd = s.cend()-1; row != rowEnd; ++row) {
        MPI_Ssend(row->data(), dim, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    // Gather results
    std::vector<Grid_t> results(snapshots.size(), Grid_t(dim, Row_t(dim, 0)));
    for (int i = 0, iMax = snapshots.size(); i < iMax; ++i) {
      for (unsigned j = 0, jEnd = snapshots[i].size()-2; j < jEnd; ++j) {
        results[i][j] = std::move(snapshots[i][j+1]);
      }
      for (int j = 1; j < nRanks; ++j) {
        const unsigned rowBegin = dim*j/nRanks;
        const unsigned rowEnd = dim*(j+1)/nRanks;
        const unsigned nRows = rowEnd - rowBegin;
        for (unsigned k = 0; k < nRows; ++k) {
          assert(MPI_Recv(results[i][rowBegin + k].data(), dim, MPI_FLOAT, j, 0,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS);
        }
      }
    }
    auto elapsedOuter = 1e-6 *
                   std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::system_clock::now() - start)
                       .count();
    std::cout << "Finished in " << elapsedOuter << " (" << elapsedInner
              << ") seconds.\n";
    std::ofstream outputStream(outPath);
    assert(outputStream.is_open());
    for (auto &grid : results) {
      for (auto &row : grid) {
        std::copy(row.cbegin(), row.cend()-1,
                  std::ostream_iterator<float>(outputStream, ","));
        outputStream << *(row.cend()-1) << "\n";
      }
      outputStream << "\n";
    }
  }
  MPI_Finalize();
  return 0;
}
