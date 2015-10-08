#pragma once

#include <mutex>
#include <condition_variable>

class Barrier {

public:
  
  inline Barrier(int nThreads);

  inline void Synchronize(int step);

private:

  int nThreads_, threadsLeft_, step_;
  std::mutex mutex_{};
  std::condition_variable cvSync_{}, cvAhead_{};

};

Barrier::Barrier(int nThreads)
    : nThreads_(nThreads), threadsLeft_(nThreads), step_(0) {}

void Barrier::Synchronize(int step) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (step > step_) {
    cvAhead_.wait(lock);
  }
  --threadsLeft_;
  if (threadsLeft_ > 0) {
    do {
      cvSync_.wait(lock);
    } while (step >= step_);
  } else {
    ++step_;
    threadsLeft_ = nThreads_;
    cvSync_.notify_all();
    cvAhead_.notify_all();
  }
}
