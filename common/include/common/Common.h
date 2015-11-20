#pragma once

#include <iterator>
#include <type_traits>

namespace hpcse {

template <typename IteratorType>
using CheckRandomAccess = typename std::enable_if<std::is_base_of<
    std::random_access_iterator_tag,
    typename std::iterator_traits<IteratorType>::iterator_category>::value>::
    type;

} // End namespace hpcse
