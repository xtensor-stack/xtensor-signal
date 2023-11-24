
#ifdef XTENSOR_USE_TBB
#include <oneapi/tbb.h>
#endif
#include <stdexcept>
#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xcomplex.hpp>

namespace xt::fft {
namespace detail {
template <class E,
          typename std::enable_if<
              xtl::is_complex<typename std::decay<E>::type::value_type>::value,
              bool>::type = true>
inline auto fft(E &&e) {
  using namespace xt::placeholders;
  using namespace std::complex_literals;
  using value_type = typename std::decay_t<E>::value_type;
  using precision = typename value_type::value_type;
  auto N = e.size();
  const bool powerOfTwo = !(N == 0) && !(N & (N - 1));
  // check for power of 2
  if (!powerOfTwo || N == 0) {
    // TODO: Replace implementation with dft
    XTENSOR_THROW(std::runtime_error, "FFT Implementation requires power of 2");
  }
  auto pi = xt::numeric_constants<precision>::PI;
  xt::xtensor<value_type, 1> ev = e;
  if (N <= 1) {
    return ev;
  } else {
#ifdef XTENSOR_USE_TBB
    xt::xtensor<value_type, 1> even;
    xt::xtensor<value_type, 1> odd;
    oneapi::tbb::parallel_invoke(
        [&] { even = fft(xt::view(ev, xt::range(0, _, 2))); },
        [&] { odd = fft(xt::view(ev, xt::range(1, _, 2))); });
#else
    auto even = fft(xt::view(ev, xt::range(0, _, 2)));
    auto odd = fft(xt::view(ev, xt::range(1, _, 2)));
#endif

    auto range = xt::arange<double>(N / 2);
    auto exp = xt::exp(static_cast<value_type>(-2i) * pi * range / N);
    auto t = exp * odd;
    auto first_half = even + t;
    auto second_half = even - t;
    // TODO: should be a call to stack if performance was improved
    auto spectrum = xt::xtensor<value_type, 1>::from_shape({N});
    xt::view(spectrum, xt::range(0, N / 2)) = first_half;
    xt::view(spectrum, xt::range(N / 2, N)) = second_half;
    return spectrum;
  }
}
} // namespace detail

/**
* @breif 1D FFT of an Nd array along a specified axis
* @param e an Nd expression to be transformed to the fourier domain
* @param axis the axis along which to perform the 1D FFT
* @return a transformed xarray of the specified precision
*/
template <class E,
          typename std::enable_if<
              xtl::is_complex<typename std::decay<E>::type::value_type>::value,
              bool>::type = true>
inline auto fft(E &&e, std::ptrdiff_t axis = -1) {
  using value_type = typename std::decay_t<E>::value_type;
  using precision = typename value_type::value_type;
  xt::xarray<std::complex<precision>> out = xt::eval(e);
  auto saxis = xt::normalize_axis(e.dimension(), axis);
  auto begin = xt::axis_slice_begin(out, saxis);
  auto end = xt::axis_slice_end(out, saxis);
  for (auto iter = begin; iter != end; iter++) {
    xt::noalias(*iter) = detail::fft(*iter);
  }
  return out;
}

/**
* @breif 1D FFT of an Nd array along a specified axis
* @param e an Nd expression to be transformed to the fourier domain
* @param axis the axis along which to perform the 1D FFT
* @return a transformed xarray of the specified precision
*/
template <class E,
          typename std::enable_if<
              !xtl::is_complex<typename std::decay<E>::type::value_type>::value,
              bool>::type = true>
inline auto fft(E &&e, std::ptrdiff_t axis = -1) {
  using value_type = typename std::decay<E>::type::value_type;
  return fft(xt::cast<std::complex<value_type>>(e), axis);
}

} // namespace xt::fft
