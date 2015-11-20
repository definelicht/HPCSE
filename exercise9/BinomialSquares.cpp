#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <immintrin.h>
#include "common/Timer.h"

using namespace hpcse;

constexpr size_t elementsPerRun = 2<<11;

void BinomialSquares(const size_t iMax, const float x[], const float y[],
                     float z[]);

#ifdef __AVX__
void BinomialSquaresAvx(const size_t iMax, const float x[], const float y[],
                        float z[]);
#endif

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cerr << "Specify number of iterations to run.\n";
    return 1;
  }
  const size_t iMax = std::stol(argv[1]);
  alignas(64) float x[elementsPerRun];
  alignas(64) float y[elementsPerRun];
  std::fill(x, x + elementsPerRun, 1);
  std::fill(y, y + elementsPerRun, 2);

  alignas(64) float zNaive[elementsPerRun];
  Timer timer;
  BinomialSquares(iMax, x, y, zNaive);
  double elapsedNaive = timer.Stop();
  std::cout << "Naive finished in " << elapsedNaive << " seconds.\n";

#ifdef __AVX__
  alignas(64) float zAvx[elementsPerRun];
  timer.Start();
  BinomialSquaresAvx(iMax, x, y, zAvx);
  double elapsedAvx = timer.Stop();
  for (size_t i = 0; i < elementsPerRun; ++i) {
    assert(zNaive[i] == zAvx[i]);
  }
  std::cout << "AVX finished in " << elapsedAvx << " seconds.\n"
            << "Speedup: " << elapsedNaive / elapsedAvx << ".\n";
#endif

  return 0;
}

__attribute__((optimize("no-tree-vectorize")))
void BinomialSquares(const size_t iMax, const float x[], const float y[],
                     float z[]) {
  for (size_t i = 0; i < iMax; ++i) {
    #pragma clang loop vectorize(disable)
    for (size_t j = 0; j < elementsPerRun; ++j) {
      z[j] = x[j]*x[j] + y[j]*y[j] + 2.*x[j]*y[j];
    }
  }
}

#ifdef __AVX__
void BinomialSquaresAvx(const size_t iMax, const float x[], const float y[],
                        float z[]) {
  const __m256 two = _mm256_set1_ps(2.);
  for (size_t i = 0; i < iMax; ++i) {
    for (size_t j = 0; j < elementsPerRun; j += 8) {
      __m256 xVec = _mm256_load_ps(x+j); 
      __m256 yVec = _mm256_load_ps(y+j);
      __m256 zVec = _mm256_mul_ps(xVec, yVec); 
      xVec = _mm256_mul_ps(xVec, xVec);
      yVec = _mm256_mul_ps(yVec, yVec);
      zVec = _mm256_mul_ps(zVec, two);
      zVec = _mm256_add_ps(zVec, xVec);
      zVec = _mm256_add_ps(zVec, yVec);
      _mm256_store_ps(z+j, zVec);
    }
  }
}
#endif
