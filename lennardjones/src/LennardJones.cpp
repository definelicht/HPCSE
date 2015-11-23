#include <cassert>
#include <iostream>
#include <immintrin.h>
#include "lennardjones/LennardJones.h"

namespace hpcse {

LennardJones::LennardJones(const float distMin, const float epsilon)
    : distMinSquared_(distMin * distMin), epsilon_(epsilon) {}

namespace {
#define HPCSE_LENNARDJONES_FUNCTION_NAME LennardJonesKernelScalar 
#include "LennardJones.inl"
#undef HPCSE_LENNARDJONES_FUNCTION_NAME
#define HPCSE_LENNARDJONES_VECTORIZE
#define HPCSE_LENNARDJONES_FUNCTION_NAME LennardJonesKernelAutoVec 
#include "LennardJones.inl"
#undef HPCSE_LENNARDJONES_VECTORIZE
#undef HPCSE_LENNARDJONES_FUNCTION_NAME
} // End anonymous namespace

float LennardJones::Diff(const ContainerItr x, const ContainerItr xEnd,
                         const ContainerItr y,
                         std::pair<float, float> const &newPos) const {
  const float *xPtr = &x[0];
  const float *yPtr = &y[0];
  const int n = std::distance(x, xEnd) - 1;
  const float newPosX = newPos.first;
  const float newPosY = newPos.second;
  return epsilon_ * LennardJonesKernelScalar(distMinSquared_, xPtr, yPtr, n,
                                             newPosX, newPosY);
}

float LennardJones::DiffAutoVec(const ContainerItr x, const ContainerItr xEnd,
                                const ContainerItr y,
                                std::pair<float, float> const &newPos) const {
  const float *xPtr = &x[0];
  const float *yPtr = &y[0];
  const int n = std::distance(x, xEnd) - 1;
  const float newPosX = newPos.first;
  const float newPosY = newPos.second;
  return epsilon_ * LennardJonesKernelAutoVec(distMinSquared_, xPtr, yPtr, n,
                                              newPosX, newPosY);
}

#ifdef __AVX__
float LennardJones::DiffAvx(const ContainerItr x, const ContainerItr xEnd,
                            const ContainerItr y,
                            std::pair<float, float> const &newPos) const {

  const size_t n = std::distance(x, xEnd) - 1;

  __m256 dE = _mm256_setzero_ps();
  const __m256 newPosX = _mm256_set1_ps(newPos.first);
  const __m256 newPosY = _mm256_set1_ps(newPos.second);
  const __m256 xLast = _mm256_set1_ps(x[n]);
  const __m256 yLast = _mm256_set1_ps(y[n]);
  const __m256 distMinSquared = _mm256_set1_ps(distMinSquared_);
  const __m256 two = _mm256_set1_ps(2.);
  const __m256 epsilon = _mm256_set1_ps(epsilon_);

  size_t i = 0;
  const float *xPtr = &x[0];
  const float *yPtr = &y[0];
  for (; i+8 <= n; i += 8, xPtr += 8, yPtr += 8) {

    const __m256 xVec = _mm256_load_ps(xPtr);
    const __m256 yVec = _mm256_load_ps(yPtr);

    // dx_0^2 = (x_n - x_i)^2
    const __m256 dx0 = _mm256_sub_ps(xLast, xVec);
    const __m256 dx0Squared = _mm256_mul_ps(dx0, dx0);

    // dx_1^2 = (x_(i+1) - x_i)^2
    const __m256 dx1 = _mm256_sub_ps(newPosX, xVec);
    const __m256 dx1Squared = _mm256_mul_ps(dx1, dx1);

    // dy_0^2 = (y_n - y_i)^2
    const __m256 dy0 = _mm256_sub_ps(yLast, yVec);
    const __m256 dy0Squared = _mm256_mul_ps(dy0, dy0);

    // dy_1^2 = (y_(i+1) - y_i)^2
    const __m256 dy1 = _mm256_sub_ps(newPosY, yVec);
    const __m256 dy1Squared = _mm256_mul_ps(dy1, dy1);

    // r_0^2 = r_(m0)^2 / (dx_0^2 + dy_0^2)
    const __m256 r0Divisor = _mm256_add_ps(dx0Squared, dy0Squared);
    const __m256 r0Squared = _mm256_div_ps(distMinSquared, r0Divisor);

    // r_1^2 = r_(m1)^2 / (dx_1^2 + dy_1^2)
    const __m256 r1Divisor = _mm256_add_ps(dx1Squared, dy1Squared);
    const __m256 r1Squared = _mm256_div_ps(distMinSquared, r1Divisor);

    // r_0^6 = (r_0^2)^3
    const __m256 r0Fourth = _mm256_mul_ps(r0Squared, r0Squared);
    const __m256 r0Sixth = _mm256_mul_ps(r0Fourth, r0Squared);

    // r_1^6 = (r_1^2)^3
    const __m256 r1Fourth = _mm256_mul_ps(r1Squared, r1Squared);
    const __m256 r1Sixth = _mm256_mul_ps(r1Fourth, r1Squared);

    // r_0^12 = (r_0^6)^2
    const __m256 r0Twelfth = _mm256_mul_ps(r0Sixth, r0Sixth);

    // r_1^12 = (r_1^12)^2
    const __m256 r1Twelfth = _mm256_mul_ps(r1Sixth, r1Sixth);

    // dE = eps * (r_1^12 - 2 * r_1^6 - r_0^12 - 2 * r_0^6);
    const __m256 twoR0Sixth = _mm256_mul_ps(r0Sixth, two);
    const __m256 twoR1Sixth = _mm256_mul_ps(r1Sixth, two);
    __m256 energy = _mm256_sub_ps(r1Twelfth, twoR0Sixth);
    energy = _mm256_sub_ps(energy, r0Twelfth);
    energy = _mm256_add_ps(energy, twoR1Sixth);
    energy = _mm256_mul_ps(energy, epsilon);

    dE = _mm256_add_ps(dE, energy);
  }

  // Collapse and store (http://stackoverflow.com/a/13222410)
  const __m128 lowerQuad = _mm256_castps256_ps128(dE);
  const __m128 upperQuad = _mm256_extractf128_ps(dE, 1);
  const __m128 sumQuad = _mm_add_ps(lowerQuad, upperQuad);
  const __m128 lowerDual = sumQuad;
  const __m128 upperDual = _mm_movehl_ps(sumQuad, sumQuad);
  const __m128 sumDual = _mm_add_ps(lowerDual, upperDual);
  const __m128 lower = sumDual;
  const __m128 upper = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  const __m128 sum = _mm_add_ss(lower, upper);
  float result = _mm_cvtss_f32(sum);

  float tail = Diff(x+i, xEnd, y+i, newPos);

  return result + tail;
}
#endif

float LennardJones::operator()(ContainerType::const_iterator x,
                               const ContainerType::const_iterator xEnd,
                               ContainerType::const_iterator y) const {

  const size_t n = std::distance(x, xEnd);

  float energy = 0;

  for (size_t i = 0; i < n; ++i) {
    float energyPart = 0;
    for (size_t j = 0; j < i; ++j) {
      float const dx = x[i] - x[j];
      float const dy = y[i] - y[j];
      float const r0Squared = distMinSquared_ / (dx * dx + dy * dy);
      float const r0Sixth = r0Squared * r0Squared * r0Squared;
      energyPart += epsilon_ * (r0Sixth * r0Sixth - 2 * r0Sixth);
    }
    energy += energyPart;
  }
  return energy;
}

} // End namespace hpcse
