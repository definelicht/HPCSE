#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include "common/Timer.h"
#include "immintrin.h"

using namespace hpcse;

#if defined(__SSE__)
#define HPCSE_PUSHFORROOF_ENABLE_SSE
#endif

#if defined(__AVX__)
#define HPCSE_PUSHFORROOF_ENABLE_AVX
#endif

constexpr size_t elementsPerRun = 2<<11;

__attribute__((optimize("no-tree-vectorize")))
double Vanilla(const size_t iMax, const float source[], float target[]) {
  Timer timer;
  for (size_t i = 0; i < iMax; ++i) {
    #pragma clang loop vectorize(disable)
    for (size_t j = 0; j < elementsPerRun; ++j) {
      target[j] += source[j];
    }
  }
  double elapsed = timer.Stop();
  return elapsed;
}

double AutoVectorization(const size_t iMax, const float source[], float target[]) {
  Timer timer;
  for (size_t i = 0; i < iMax; ++i) {
    #pragma clang loop vectorize(enable)
    for (size_t j = 0; j < elementsPerRun; ++j) {
      target[j] += source[j];
    }
  }
  double elapsed = timer.Stop();
  return elapsed;
}

#ifdef HPCSE_PUSHFORROOF_ENABLE_SSE
double SseIntrinsics(const size_t iMax, const float source[], float target[]) {
  Timer timer;
  for (size_t i = 0; i < iMax; ++i) {
    for (size_t j = 0; j < elementsPerRun; j += 4) {
      __m128 s = _mm_load_ps(source+j); 
      __m128 t = _mm_load_ps(target+j);
      s = _mm_add_ps(s, t);
      _mm_store_ps(target+j, s);
    }
  }
  double elapsed = timer.Stop();
  return elapsed;
}
#endif

#ifdef HPCSE_PUSHFORROOF_ENABLE_AVX
double AvxIntrinsics(const size_t iMax, const float source[], float target[]) {
  Timer timer;
  for (size_t i = 0; i < iMax; ++i) {
    for (size_t j = 0; j < elementsPerRun; j += 8) {
      __m256 s = _mm256_load_ps(source+j); 
      __m256 t = _mm256_load_ps(target+j);
      s = _mm256_add_ps(s, t);
      _mm256_store_ps(target+j, s);
    }
  }
  double elapsed = timer.Stop();
  return elapsed;
}
#endif

void VerifyOutput(const float expected, float const target[]) {
  for (size_t i = 0; i < elementsPerRun; ++i) {
    if (target[i] != expected) {
      std::cout << target[i] << " vs. " << expected << "\n";
      assert(false);
    }
  }
}

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cerr << "Specify number of iterations to run.\n";
    return 1;
  }
  const size_t iMax = std::stol(argv[1]);
  double nFlops = elementsPerRun*iMax;

  // Align arrays for AVX
  float source[elementsPerRun] __attribute__((aligned(64)));
  std::fill(source, source+elementsPerRun, 1.);

  auto runBenchmark = [&source, iMax, nFlops](
      std::string const &name,
      std::function<double(size_t, const float[], float[])> const &f) {
    float target[elementsPerRun] __attribute__((aligned(64)));
    std::fill(target, target+elementsPerRun, 0.);
    double elapsed = f(iMax, source, target);
    VerifyOutput(iMax, target);
    std::cout << name << ": " << (nFlops / elapsed) << " FLOPS.\n";
  };

  runBenchmark("Vanilla", Vanilla);
  runBenchmark("Autovectorization", AutoVectorization);
  #ifdef HPCSE_PUSHFORROOF_ENABLE_SSE
  runBenchmark("SSE Intrinsics", SseIntrinsics);
  #endif
  #ifdef HPCSE_PUSHFORROOF_ENABLE_AVX
  runBenchmark("AVX Intrinsics", AvxIntrinsics);
  #endif

  return 0;
}
