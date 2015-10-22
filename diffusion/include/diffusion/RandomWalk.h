#pragma once

#include <functional>
#include <utility>

namespace hpcse {

std::pair<float, float>
RandomWalk(unsigned iterations, float d, std::pair<float, float> const &start,
           std::pair<float, float> const &xBounds,
           std::pair<float, float> const &yBounds,
           std::function<float(float, float)> const &boundaryCondition);

std::pair<float, float>
RandomWalk(unsigned nCores, unsigned iterations, float d,
           std::pair<float, float> const &start,
           std::pair<float, float> const &xBounds,
           std::pair<float, float> const &yBounds,
           std::function<float(float, float)> const &boundaryCondition);

} // End namespace hpcse
