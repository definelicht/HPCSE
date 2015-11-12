#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include "common/Timer.h"
#include "immintrin.h"

using namespace hpcse;

void CopyLoop(const size_t numFloats, const float src[], float tgt[]) {
  for (size_t i = 0; i < numFloats; ++i) {
    tgt[i] = src[i];
  }
}

void CopyStl(const size_t numFloats, const float src[], float tgt[]) {
  std::copy(src, src+numFloats, tgt);
}

void CopyC(const size_t numFloats, const float src[], float tgt[]) {
  std::memcpy(tgt, src, numFloats*sizeof(float));
}

int main(int argc, char const *argv[]) {
  if (argc < 3) {
    std::cerr << "Specify number of iterations and memory size(s).\n";
    return 1;
  }
  const size_t iMax =  std::stol(argv[1]);
  for (int i = 2; i < argc; ++i) {
    const size_t memSize = std::stol(argv[i]);
    const size_t numFloats = memSize/4;
    float *src = new float[numFloats]; 
    std::fill(src, src+numFloats, 1.);
    auto doCopy = [&](std::function<void(size_t, float[], float[])> const &f,
                      std::string const &name) {
      float *tgt = new float[numFloats];
      Timer timer;
      for (size_t i = 0; i < iMax; ++i) {
        f(numFloats, src, tgt);
      }
      double elapsed = timer.Stop();
      for (size_t i = 0; i < numFloats; ++i) {
        assert(tgt[i] == 1);
      }
      double elapsedAvg = elapsed / iMax;
      std::cout << name << ": " << memSize << "B done in " << elapsedAvg
                << " seconds: " << 1e-9 * memSize / elapsedAvg << " GB/s.\n";
      delete[] tgt;
    };
    doCopy(CopyLoop, "Loop");
    doCopy(CopyStl, "std::copy");
    doCopy(CopyC, "std::memcpy");
    delete[] src;
  }
}
