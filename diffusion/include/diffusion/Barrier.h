/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date October 2015

#pragma once

#include <mutex>
#include <condition_variable>

namespace hpcse {

class Barrier {

public:
  
  inline Barrier(int nThreads);

  inline void Synchronize();

private:

  int nThreads_, threadsLeft_;
  bool step_{true};
  std::mutex mutex_{};
  std::condition_variable cv_{};

};

Barrier::Barrier(int nThreads) : nThreads_(nThreads), threadsLeft_(nThreads) {}

void Barrier::Synchronize() {
  std::unique_lock<std::mutex> lock(mutex_);
  --threadsLeft_;
  if (threadsLeft_ > 0) {
    auto sleepStep = step_;
    do {
      cv_.wait(lock);
    } while (sleepStep == step_);
  } else {
    threadsLeft_ = nThreads_;
    step_ = !step_;
    cv_.notify_all();
  }
}

} // End namespace hpcse
