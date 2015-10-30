/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date October 2015

#include <cassert>
#include <memory>
#include <vector>
#include <mpi.h>
#include "diffusion/Diffusion.h"

namespace hpcse {

std::vector<Grid_t> DiffusionMPI(const unsigned dim, const float d,
                                 const float dt,
                                 std::vector<float> const &snapshots) {

  // MPI initialization
  int rank, nRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
  const unsigned rowBegin = dim*rank/nRanks;
  const unsigned rowEnd = dim*(rank+1)/nRanks;
  const unsigned nRows = rowEnd - rowBegin;
  Grid_t grid(nRows+2, Row_t(dim)); // Added padding

  // Initialize grid
  const int minCol = dim>>2;
  const int maxCol = dim - minCol;
  const int minRow = minCol - rowBegin;
  const int maxRow = (dim - minCol) - rowBegin;
  const int jMax = dim;
  for (int i = -1, iMax = nRows+1; i < iMax; ++i) {
    const bool inRow = i > minRow && i < maxRow;
    for (int j = 0; j < jMax; ++j) {
      grid[i+1][j] = inRow && j > minCol && j < maxCol;
    }
  }
  Grid_t buffer(grid);

  // Run diffusion
  std::vector<Grid_t> output(snapshots.size());
  auto outputItr = output.begin();
  auto snapshotItr = snapshots.cbegin();
  const auto snapshotEnd = snapshots.cend();
  const float ds = 2./dim;
  const float factor = d*dt/(ds*ds);
  const int iMax = nRows+1;
  float t = 0;
  while (true) {
    if (t >= *snapshotItr) {
      *outputItr++ = grid;
      if (++snapshotItr == snapshotEnd) break; 
    }
    for (int i = 1; i < iMax; ++i) {
      for (int j = 1; j < jMax; ++j) {
        buffer[i][j] =
            grid[i][j] +
            factor * (grid[i - 1][j] + grid[i][j - 1] - 4 * grid[i][j] +
                      grid[i][j + 1] + grid[i + 1][j]);
      }
    }
    MPI_Request northSend, southSend;
    if (rank != 0) {
      assert(MPI_Isend(buffer[1].data(), dim, MPI_FLOAT, rank - 1, 0,
                       MPI_COMM_WORLD, &northSend) == MPI_SUCCESS);
    } 
    if (rank != nRanks-1) {
      assert(MPI_Isend(buffer[nRows].data(), dim, MPI_FLOAT, rank + 1, 0,
                       MPI_COMM_WORLD, &southSend) == MPI_SUCCESS);
    }
    if (rank != 0) {
      MPI_Status northReceive;
      assert(MPI_Recv(buffer[0].data(), dim, MPI_FLOAT, rank - 1, 0,
                      MPI_COMM_WORLD, &northReceive) == MPI_SUCCESS);
    }
    if (rank != nRanks-1) {
      MPI_Status southReceive;
      assert(MPI_Recv(buffer[nRows].data(), dim, MPI_FLOAT, rank + 1, 0,
                      MPI_COMM_WORLD, &southReceive) == MPI_SUCCESS);
    }
    if (rank != 0) {
      MPI_Status northStatus;
      assert(MPI_Wait(&northSend, &northStatus) == MPI_SUCCESS);
    }
    if (rank != nRanks-1) {
      MPI_Status southStatus;
      assert(MPI_Wait(&southSend, &southStatus) == MPI_SUCCESS);
    }
    grid.swap(buffer);
    t += dt;
  }
  return output;
}

} // End namespace hpcse
