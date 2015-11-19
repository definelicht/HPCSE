#include <algorithm>
#include <cmath>
#include "common/Timer.h"
#include "common/Mpi.h"
#include "diffusion/Diffusion.h"
#include "diffusion/DiffusionMPI.h"

namespace hpcse {

std::vector<Grid_t> DiffusionGrid(const unsigned gridDim, const float d,
                                  const float dt,
                                  std::vector<float> const &_timesToRecord) {

  std::vector<float> timesToRecord(_timesToRecord);
  const int nSnapshots = timesToRecord.size();

  // Construct cartesian grid
  const int nHorizontal = std::sqrt(mpi::size());
  const int nVertical = mpi::size() / nHorizontal;
  mpi::CartesianGrid<2> mpiGrid({{nVertical, nHorizontal}}, false);

  // Determine local grid
  const int rowBegin = gridDim * mpiGrid.row() / mpiGrid.rowMax();
  const int rowEnd = gridDim * (mpiGrid.row() + 1) / mpiGrid.rowMax();
  const int nRows = rowEnd - rowBegin;
  const int colBegin = gridDim * mpiGrid.col() / mpiGrid.colMax();
  const int colEnd = gridDim * (mpiGrid.col() + 1) / mpiGrid.colMax();
  const int nCols = colEnd - colBegin;
  Grid_t grid(nRows + 2, Row_t(nCols + 2)); // Added padding

  // Initialize local grid values
  const int fillStart = gridDim / 4;
  const int fillEnd = gridDim - fillStart;
  const int minRow = fillStart - rowBegin;
  const int maxRow = fillEnd - rowBegin;
  const int minCol = fillStart - colBegin;
  const int maxCol = fillEnd - colBegin;
  const int iMax = nRows + 1;
  const int jMax = nCols + 1;
  for (int i = -1; i < iMax; ++i) {
    const bool inRow = i > minRow && i < maxRow;
    for (int j = -1; j < jMax; ++j) {
      grid[i + 1][j + 1] = inRow && j > minCol && j < maxCol;
    }
  }
  Grid_t gridBuffer(grid); // Copy into gridBuffer

  // Run diffusion
  std::sort(timesToRecord.begin(), timesToRecord.end());
  std::vector<Grid_t> localSnapshots(nSnapshots);
  auto outputItr = localSnapshots.begin();
  auto timeItr = timesToRecord.cbegin();
  const auto timeItrEnd = timesToRecord.cend();
  const float ds = 2. / gridDim;
  const float factor = d * dt / (ds * ds);
  const int iMaxInner = iMax - 1;
  const int jMaxInner = jMax - 1;
  float t = 0;
  auto diffuse = [&grid, &factor](const int i, const int j) {
    return grid[i][j] +
           factor * (grid[i - 1][j] + grid[i][j - 1] - 4 * grid[i][j] +
                     grid[i][j + 1] + grid[i + 1][j]);
  };
  const std::array<std::pair<int, bool>, 4> neighbors = {
      {mpiGrid.up(), mpiGrid.down(), mpiGrid.left(), mpiGrid.right()}};
  std::vector<float> bufferSendLeft(nRows+2);
  std::vector<float> bufferSendRight(nRows+2);
  std::vector<float> bufferReceiveLeft(nRows+2);
  std::vector<float> bufferReceiveRight(nRows+2);

  for (;;t += dt) {

    if (t >= *timeItr) {
      // Collect snapshot
      *outputItr++ = grid;
      if (++timeItr == timeItrEnd) {
        break;
      }
    }

    // Handle edges
    std::vector<MPI_Request> colReceive;
    std::vector<MPI_Request> requests;
    // Horizontal edges 
    for (int i = 1; i < iMax; ++i) {
      bufferSendLeft[i]  = diffuse(i, 1);
      bufferSendRight[i] = diffuse(i, nCols);
    }
    if (neighbors[2].second) {
      requests.emplace_back(mpi::SendAsync(bufferSendLeft.cbegin(),
                                           bufferSendLeft.cend(),
                                           neighbors[2].first));
      colReceive.emplace_back(mpi::ReceiveAsync(bufferReceiveLeft.begin(),
                                                bufferReceiveLeft.end(),
                                                neighbors[2].first));
    }
    if (neighbors[3].second) {
      requests.emplace_back(mpi::SendAsync(bufferSendRight.cbegin(),
                                           bufferSendRight.cend(),
                                           neighbors[3].first));
      colReceive.emplace_back(mpi::ReceiveAsync(bufferReceiveRight.begin(),
                                                bufferReceiveRight.end(),
                                                neighbors[3].first));
    }
    // Top edge
    for (int j = 1; j < jMax; ++j) {
      gridBuffer[1][j] = diffuse(1, j);
    }
    if (neighbors[0].second) {
      requests.emplace_back(mpi::SendAsync(
          gridBuffer[1].cbegin(), gridBuffer[1].cend(), neighbors[0].first));
      requests.emplace_back(mpi::ReceiveAsync(
          gridBuffer[0].begin(), gridBuffer[0].end(), neighbors[0].first));
    }
    // Bottom edge
    for (int j = 1; j < jMax; ++j) {
      gridBuffer[nRows][j] = diffuse(nRows, j);
    }
    if (neighbors[1].second) {
      requests.emplace_back(mpi::SendAsync(gridBuffer[nRows].cbegin(),
                                           gridBuffer[nRows].cend(),
                                           neighbors[1].first));
      requests.emplace_back(mpi::ReceiveAsync(gridBuffer[nRows + 1].begin(),
                                              gridBuffer[nRows + 1].end(),
                                              neighbors[1].first));
    }

    // Retrieve the cols so they can be copied to the grid while computing the
    // bulk
    mpi::WaitAll(colReceive);

    // Compute the bulk
    for (int i = 2; i < iMaxInner; ++i) {
      gridBuffer[i][0] = bufferReceiveLeft[i];
      gridBuffer[i][1] = bufferSendLeft[i];
      for (int j = 2; j < jMaxInner; ++j) {
        gridBuffer[i][j] = diffuse(i, j);
      }
      gridBuffer[i][nCols] = bufferSendRight[i];
      gridBuffer[i][nCols + 1] = bufferReceiveRight[i];
    }

    // Wait for all remaining sends and receives to finish
    mpi::WaitAll(requests);

    gridBuffer.swap(grid);

  } // End main loop

  // Gather results
  const MPI_Comm rowComm = mpiGrid.Partition<0>();
  const MPI_Comm colComm = mpiGrid.Partition<1>();
  const int rank = mpi::rank();
  const int colRank = mpi::rank(colComm);
  const int rowRank = mpi::rank(rowComm);
  std::vector<Grid_t> totalSnapshots;
  if (rank == 0) {
    totalSnapshots =
        std::vector<Grid_t>(nSnapshots, Grid_t(gridDim, Row_t(gridDim)));
  }
  std::vector<Grid_t> rowSnapshots;
  if (colRank == 0) {
    rowSnapshots =
        std::vector<Grid_t>(nSnapshots, Grid_t(nRows, Row_t(gridDim)));
  }
  std::vector<MPI_Request> requests;
  for (int s = 0; s < nSnapshots; ++s) {
    // Gather grid rows across columns in each row of MPI ranks
    for (int i = 0; i < nRows; ++i) {
      typename Row_t::iterator target;
      if (colRank == 0) {
        target = rowSnapshots[s][i].begin();
      }
      mpi::Gather(localSnapshots[s][i + 1].cbegin() + 1,
                  localSnapshots[s][i + 1].cend() - 1, target, 0, colComm);
    }
    // Gather all rows in root rank
    if (colRank == 0) {
      if (rowRank != 0) {
        for (int i = 0; i < nRows; ++i) {
          mpi::Send(rowSnapshots[s][i].cbegin(), rowSnapshots[s][i].cend(), 0,
                    0, rowComm);
        }
      } else {
        int globalRow = 0;
        for (int i = 0; i < nRows; ++i) {
          totalSnapshots[s][globalRow] = std::move(rowSnapshots[s][i]);
          ++globalRow;
        }
        for (int r = 1, rMax = mpiGrid.rowMax(); r < rMax; ++r) {
          const int currRowBegin = gridDim*r/nVertical;
          const int currRowEnd = gridDim*(r+1)/nVertical;
          const int currNRows = currRowEnd - currRowBegin;
          for (int i = 0; i < currNRows; ++i) {
            requests.emplace_back(mpi::ReceiveAsync(
                totalSnapshots[s][globalRow].begin(),
                totalSnapshots[s][globalRow].end(), r, 0, rowComm));
            ++globalRow;
          }
        }
      }
    }
  }
  mpi::WaitAll(requests);

  return totalSnapshots;
}

} // End namespace hpcse
