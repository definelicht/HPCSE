/// \author Johannes de Fine Licht (definelj@student.ethz.ch)
/// \date September 2015

#include <functional>

namespace hpcse {

double RiemannSequential(std::function<double(double)> const &f, double begin,
                         double end, const int n);

double RiemannParallel(std::function<double(double)> const &f, double begin,
                       double end, const int n, const int nThreads);

} // End namespace hpcse
