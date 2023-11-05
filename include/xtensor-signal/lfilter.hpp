#ifndef XTENSOR_SIGNAL_LFILTER_HPP
#define XTENSOR_SIGNAL_LFILTER_HPP

#include <type_traits>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>

namespace xt {
namespace signal {
namespace detail {
template <typename E1, typename E2, typename E3, typename E4>
auto lfilter(E1 &&b, E2 &&a, E3 &&x, E4 zi) {
  using value_type = typename std::decay_t<E3>::value_type;
  xt::xtensor<value_type, 1> out = xt::zeros<value_type>({x.shape(0)});

  for (int i = 0; i < x.shape(0); i++) {
    value_type tmp = 0;
    if (i < zi.shape(0)) {
      tmp = zi(i);
    }
    for (int j = 0; j < b.shape(0); j++) {
      if (!(i - j < 0)) {
        tmp += b(j) * x(i - j);
      }
    }

    for (int j = 1; j < a.shape(0); j++) {
      if (!(i - j < 0)) {
        tmp -= a(j) * out(i - j);
      }
    }

    tmp /= a(0);
    out(i) = tmp;
  }
  return out;
}
} // namespace detail

/*
 * @brief performs a 1D filter operation along the specified axis.
 * @param b the numerator of the filter expression
 * @param a the denominator of the filter expression
 * @param x input dataset
 * @param axis the axis along which to perform the filter operation
 * @param zi initial condition of the filter accumulator
 * @return filtered version of x
 * @todo Add implementation bound to MKL or HPC library for IIR and FIR
 */
template <typename E1, typename E2, typename E3,
          typename E4 = decltype(xt::xnone())>
auto lfilter(E1 &&b, E2 &&a, E3 &&x, std::ptrdiff_t axis = -1,
             E4 zi = xt::xnone()) {
  using value_type = typename std::decay_t<E3>::value_type;
  xt::xarray<value_type> out(x);
  auto saxis = xt::normalize_axis(out.dimension(), axis);
  auto begin = xt::axis_slice_begin(out, saxis);
  auto end = xt::axis_slice_end(out, saxis);

  for (auto iter = begin; iter != end; iter++) {
    if constexpr (std::is_same<typename std::decay<E4>::type,
                               decltype(xt::xnone())>::value == false) {
      xt::noalias(*iter) = detail::lfilter(b, a, *iter, zi);
    } else {
      xt::noalias(*iter) = detail::lfilter(
          b, a, *iter, xt::zeros<value_type>({(*iter).shape(0)}));
    }
  }
  return out;
}
} // namespace signal
} // namespace xt

#endif
