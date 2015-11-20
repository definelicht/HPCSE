#pragma once

#include <cstdlib>
#include <memory>
#include <new>

namespace hpcse {

template <typename T, unsigned Alignment = 64>
class AlignedAllocator {

public:
  using pointer = T*;
  using const_pointer = T const*;
  using reference = T&;
  using const_reference = T const&;
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  inline explicit AlignedAllocator() = default; 

  template <typename U>
  inline explicit AlignedAllocator(AlignedAllocator<U> const&) {}

  template <typename U>
  inline explicit AlignedAllocator(AlignedAllocator<U> &&) {}

  inline ~AlignedAllocator() = default;

  pointer address(reference x) const {
    return &x;
  }

  const_pointer address(const_reference x) const {
    return &x;
  }

  pointer allocate(size_type n) {
    pointer p;
    if (posix_memalign(reinterpret_cast<void**>(&p), Alignment, n*sizeof(T))) {
      throw std::bad_alloc();
    }
    return p;
  }

  void deallocate(pointer p, size_type) {
    std::free(p);
  }

  size_type max_size() {
    return std::allocator<T>().max_size();
  }

  template <typename U, class... Args>
  void construct(U *p, Args&&... args) {
    new (reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
  }

  template <typename U>
  void destroy(U *p) {
    p->~U();
  }

  template <typename T1, unsigned Alignment1>
  constexpr bool operator==(AlignedAllocator<T1, Alignment1> const &) {
    return true;
  }

  template <typename T1, unsigned Alignment1>
  constexpr bool operator!=(AlignedAllocator<T1, Alignment1> const &) {
    return false;
  }

};

} // End namespace hpcse
