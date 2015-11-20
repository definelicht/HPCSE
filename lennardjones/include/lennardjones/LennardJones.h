#pragma once

#include <vector>
#include "common/AlignedAllocator.h"
#include "common/Common.h"

namespace hpcse {

class LennardJones {

public:

  using ContainerType = std::vector<float, AlignedAllocator<float, 64>>;
  using ContainerItr = typename ContainerType::const_iterator;

  LennardJones(const float distMin, const float epsilon);

  float Diff(ContainerItr x, ContainerItr xEnd, ContainerItr y,
             std::pair<float, float> const &newPos) const;

#ifdef __AVX__
  float DiffAvx(ContainerItr x, ContainerItr xEnd, ContainerItr y,
                std::pair<float, float> const &newPos) const;
#endif

  float operator()(ContainerItr x, const ContainerItr xEnd,
                   ContainerItr y) const;

private:
  const float distMinSquared_;
  const float epsilon_;
};

} // End namespace hpcse
