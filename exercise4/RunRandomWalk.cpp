#include "diffusion/RandomWalk.h"
#include <chrono>
#include <iostream>
#include <string>

using namespace hpcse;

int main(int argc, char const *argv[]) {
  if (argc > 3) {
    std::cerr << "Usage: <number of cores> <number of iterations>" << std::endl;
    return 1;
  }
  unsigned nThreads = std::stoi(argv[1]);
  unsigned iterations = std::stoi(argv[2]);
  auto start = std::chrono::system_clock::now();
  auto result = RandomWalk(nThreads, iterations, 0.01, {0.3, 0.4}, {0, 1},
                           {0, 1}, [](float x, float) { return x; });
  auto elapsed = 1e-6 *
                 std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::system_clock::now() - start)
                     .count();
  std::cout << nThreads << "," << iterations << "," << result.first << ","
            << result.second << "," << elapsed << "\n";
  return 0;
}
