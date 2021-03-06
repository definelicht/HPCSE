/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date October 2015

#include "diffusion/Barrier.h"

#include <iostream>
#include <thread>
#include <vector>

using namespace hpcse;

void Race() {
  auto printStep = []() {
    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 10; ++j) {
        std::cout << j;
      }
    }
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back(printStep);
  }
  for (auto &t : threads) {
    t.join();
  }
}

void NoRace() {
  Barrier barrier(8);
  auto printStep = [&barrier]() {
    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 10; ++j) {
        std::cout << j;
        barrier.Synchronize();
      }
    }
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back(printStep);
  }
  for (auto &t : threads) {
    t.join();
  }
}

int main() {
  std::cout << "Running without barrier:\n";
  Race();
  std::cout << "\nRunning with barrier:\n";
  NoRace();
  std::cout << "\n";
  return 0;
}
