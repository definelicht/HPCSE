#ifdef HPCSE_LENNARDJONES_VECTORIZE
__attribute__((optimize("tree-vectorize")))
#else
__attribute__((optimize("no-tree-vectorize")))
#endif
float HPCSE_LENNARDJONES_FUNCTION_NAME (
    const float distMinSquared, const float *__restrict__ x,
    const float *__restrict__ y, const int n, const float newPosX,
    const float newPosY) {

  float dE = 0.0;

#ifdef HPCSE_LENNARDJONES_VECTORIZE
  #pragma clang loop vectorize(enable) interleave(enable)
  #pragma GCC ivdep
#else
  #pragma clang loop vectorize(disable)
#endif
  for (int i = 0; i < n; ++i) {
    const float dx = x[n] - x[i];
    const float dxNew = newPosX - x[i];
    const float dy = y[n] - y[i];
    const float dyNew = newPosY - y[i];
    const float r0Squared = distMinSquared / (dx * dx + dy * dy);
    const float r1Squared = distMinSquared / (dxNew * dxNew + dyNew * dyNew);
    const float r0Sixth = r0Squared * r0Squared * r0Squared;
    const float r1Sixth = r1Squared * r1Squared * r1Squared;

    dE +=
        ((r1Sixth * r1Sixth - 2 * r1Sixth) - (r0Sixth * r0Sixth - 2 * r0Sixth));
  }

  return dE;
}
