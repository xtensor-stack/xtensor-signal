#ifndef XTENSOR_SIGNAL_LFILTER_HPP
#define XTENSOR_SIGNAL_LFILTER_HPP

#include <type_traits>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xexception.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>

namespace xt {
namespace signal {
namespace detail {
template <typename E1, typename E2, typename E3, typename E4>
inline auto lfilter(E1 &&b, E2 &&a, E3 &&x, E4 zi) {
  using value_type = typename std::decay_t<E3>::value_type;
  using size_type = typename std::decay_t<E3>::size_type;
  if (zi.shape(0) != x.shape(0)) {
    XTENSOR_THROW(
        std::runtime_error,
        "Accumulator initialization must be the same length as the input");
  }
  if (x.dimension() != 1) {
    XTENSOR_THROW(std::runtime_error,
                  "Implementation only works on 1D arguments");
  }
  if (a.dimension() != 1) {
    XTENSOR_THROW(std::runtime_error,
                  "Implementation only works on 1D arguments");
  }
  if (b.dimension() != 1) {
    XTENSOR_THROW(std::runtime_error,
                  "Implementation only works on 1D arguments");
  }
  xt::xtensor<value_type, 1> out = xt::zeros_like(x);
  auto padded_x = xt::pad(x, b.shape(0) - 1);
  auto padded_out = xt::pad(out, a.shape(0) - 1);
  for (size_type i = 0; i < x.shape(0); i++) {
    auto b_accum =
        xt::sum(b *
                xt::flip(xt::view(padded_x, xt::range(i, i + b.shape(0))))) +
        zi(i);
    auto a_accum =
        b_accum - xt::sum(xt::view(a, xt::range(1, xt::placeholders::_)) *
                          xt::flip(xt::view(padded_out,
                                            xt::range(i, i + a.shape(0) - 1))));
    auto result = a_accum / a(0);
    out(i) = result();
  }
  return out;
}
} // namespace detail

/*
 * @brief performs a 1D filter operation along the specified axis. Performs
 * operations immediately.
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
inline auto lfilter(E1 &&b, E2 &&a, E3 &&x, std::ptrdiff_t axis = -1,
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
