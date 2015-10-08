#include "Diffusion.h"

#include "DiffusionJob.h"
#include <future>
#include <memory>
#include <vector>

std::shared_ptr<DiffusionJob>
DiffusionAllocate(unsigned cols, unsigned rowBegin, unsigned rowEnd);

std::vector<Grid_t> DiffusionParallel(unsigned nCores, unsigned dim, float dt,
                                      std::vector<float> const &snapshots) {
  unsigned rowsPerCore = dim / nCores;
  std::vector<std::shared_ptr<DiffusionJob>> workers;
  { 
    // Let each worker allocate and initialize their own set of rows
    std::vector<std::future<std::shared_ptr<DiffusionJob>>> futures;
    for (unsigned i = 0; i < nCores; ++i) {
      futures.emplace_back(std::async(std::launch::async,
                                      DiffusionJob::Allocate, rowsPerCore, dim,
                                      i * rowsPerCore));
    }
    for (unsigned i = 0; i < nCores; ++i) {
      workers.emplace_back(futures[i].get());
    }
  }
  std::vector<Grid_t> output(snapshots.size());
  {
    // Let workers run independently, synchronizing at each timestep
    std::vector<std::future<std::vector<Grid_t>>> futures;
    Barrier barrier(nCores);
    for (unsigned i = 0; i < nCores; ++i) {
      futures.emplace_back(std::async(
          std::launch::async,
          [dt, &snapshots, &barrier](std::shared_ptr<DiffusionJob> job,
                                     std::shared_ptr<DiffusionJob> above,
                                     std::shared_ptr<DiffusionJob> below) {
            return job->RunDiffusion(above, below, dt, snapshots, barrier);
          },
          workers[i], i > 0 ? workers[i - 1] : nullptr,
          i < nCores - 1 ? workers[i + 1] : nullptr));
    }
    for (auto &f : futures) {
      auto partialSnapshot = f.get();
      for (int i = 0, iEnd = snapshots.size(); i < iEnd; ++i) {
        output[i].insert(output[i].end(), partialSnapshot[i].cbegin(),
                         partialSnapshot[i].cend());
      }
    }
  }
  return output;
}