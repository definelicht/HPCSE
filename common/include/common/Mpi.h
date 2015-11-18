#pragma once

#include <array>
#include <iterator>
#include <stdexcept>
#include <mpi.h>

namespace hpcse::mpi {

namespace {

template <typename T>
struct MpiType;

#define CPPUTILS_MPI_SENDBACKEND(TYPE, MPI_TYPE_NAME)                          \
  template <> struct MpiType<TYPE> {                                           \
    static MPI_Datatype value() { return MPI_TYPE_NAME; }                      \
  }
CPPUTILS_MPI_SENDBACKEND(int, MPI_INT);
CPPUTILS_MPI_SENDBACKEND(unsigned, MPI_UNSIGNED);
CPPUTILS_MPI_SENDBACKEND(long, MPI_LONG);
CPPUTILS_MPI_SENDBACKEND(char, MPI_CHAR);
CPPUTILS_MPI_SENDBACKEND(float, MPI_FLOAT);
CPPUTILS_MPI_SENDBACKEND(double, MPI_DOUBLE);
#undef CPPUTILS_MPI_SENDBACKEND

} // End anonymous namespace

inline int rank(MPI_Comm const &comm) {
  int output;
  MPI_Comm_rank(comm, &output);
  return output;
}

inline int rank() {
  return rank(MPI_COMM_WORLD);
}

inline int size(MPI_Comm const &comm) {
  int output;
  MPI_Comm_size(comm, &output);
  return output;
}

inline int size() {
  return size(MPI_COMM_WORLD);
}

template <typename IteratorType,
          typename = typename std::enable_if<
              std::is_base_of<std::random_access_iterator_tag,
                              typename std::iterator_traits<IteratorType>::
                                  iterator_category>::value>::type>
void Send(const IteratorType begin, const IteratorType end,
          const int destination, const int tag = 0,
          const MPI_Comm comm = MPI_COMM_WORLD) {
  using T = typename std::iterator_traits<const IteratorType>::value_type;
  MPI_Send(&(*begin), std::distance(begin, end), MpiType<T>::value(),
           destination, tag, comm);
}

template <typename IteratorType,
          typename = typename std::enable_if<
              std::is_base_of<std::random_access_iterator_tag,
                              typename std::iterator_traits<IteratorType>::
                                  iterator_category>::value>::type>
MPI_Status Receive(const IteratorType begin, const IteratorType end,
                   const int source, const int tag = 0,
                   const MPI_Comm comm = MPI_COMM_WORLD) {
  using T = typename std::iterator_traits<IteratorType>::value_type;
  MPI_Status status;
  MPI_Recv(&(*begin), std::distance(begin, end), MpiType<T>::value(), source,
           tag, comm, &status);
  return status;
}

class Context {

public:
  inline Context() {
    MPI_Init(nullptr, nullptr);
  }
  inline Context(int *argc, char ***argv) {
    MPI_Init(argc, argv);
  }
  inline ~Context() {
    MPI_Finalize();
  }
  Context(Context const &) = delete;
  Context(Context &&) = delete;
  Context &operator=(Context const &) = delete;
  Context &operator=(Context &&) = delete;
};


template <size_t Dim>
class CartesianGrid {

  static_assert(Dim > 0, "Cartesian grid must have a least one dimension.");

public:
  CartesianGrid(std::array<int, Dim> const &dimensions,
                const bool periodic = false,
                const MPI_Comm comm = MPI_COMM_WORLD)
      : dimensions_(dimensions) {
    std::fill(periods_.begin(), periods_.end(), periodic);
    MPI_Cart_create(comm, Dim, dimensions_.data(), periods_.data(), false,
                    &cartComm_);
    MPI_Cart_get(cartComm_, Dim, dimensions_.data(), periods_.data(),
                 coords_.data());
  }

  template <size_t GetDim>
  int get() const {
    return coords_[GetDim];
  }

  int get(const size_t dim) const {
    return coords_[dim];
  }

  template <size_t GetDim>
  int getMax() const {
    static_assert(GetDim < Dim, "Requested dimension is out of bounds.");
    return dimensions_[GetDim];
  }

  int getMax(const size_t dim) const {
    return dimensions_[dim];
  }

  typename std::enable_if<!std::less<size_t>()(Dim, 1), int>::type row() const {
    return coords_[Dim-2];
  }

  typename std::enable_if<!std::less<size_t>()(Dim, 1), int>::type col() const {
    return coords_[Dim-1];   
  }

  typename std::enable_if<!std::less<size_t>()(Dim, 1), int>::type
  rowMax() const {
    return dimensions_[Dim-2];
  }

  typename std::enable_if<!std::less<size_t>()(Dim, 1), int>::type
  colMax() const {
    return dimensions_[Dim-1];
  }

  template <size_t ShiftDim>
  std::pair<int, bool> shift(const int amount) const {
    static_assert(ShiftDim < Dim, "Requested dimension is out of bounds.");
    std::pair<int, bool> output;
    int source;
    MPI_Cart_shift(cartComm_, ShiftDim, amount, &source, &output.first);
    output.second = output.first != MPI_PROC_NULL;
    return output;
  }

  std::pair<int, bool> shift(const size_t dim, const int amount) const {
    std::pair<int, bool> output;
    int source;
    MPI_Cart_shift(cartComm_, dim, amount, &source, &output.first);
    output.second = output.first != MPI_PROC_NULL;
    return output;
  }

  typename std::enable_if<!std::less<size_t>()(Dim, 1),
                          std::pair<int, bool>>::type
  left(const int amount = 1) const {
    return shift<Dim - 1>(-amount);
  }

  typename std::enable_if<!std::less<size_t>()(Dim, 1),
                          std::pair<int, bool>>::type
  right(const int amount = 1) const {
    return shift<Dim - 1>(amount);
  }

  typename std::enable_if<!std::less<size_t>()(Dim, 1),
                          std::pair<int, bool>>::type
  up(const int amount = 1) const {
    return shift<Dim - 2>(-amount);
  }

  typename std::enable_if<!std::less<size_t>()(Dim, 1),
                          std::pair<int, bool>>::type
  down(const int amount = 1) const {
    return shift<Dim - 2>(amount);
  }

private:
  std::array<int, Dim> dimensions_; 
  std::array<int, Dim> periods_;
  std::array<int, Dim> coords_;
  MPI_Comm cartComm_{};

};

} // End namespace hpcse::mpi
