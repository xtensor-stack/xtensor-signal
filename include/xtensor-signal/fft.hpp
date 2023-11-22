
#ifdef XTENSOR_USE_TBB
#include <oneapi/tbb.h>
#endif
#include <stdexcept>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xcomplex.hpp>

namespace xt::fft {

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
  auto pi = xt::numeric_constants<precision>::PI;
  auto ev = xt::eval(e);
  if (N <= 1) {
    return ev;
  } else {
#ifdef XTENSOR_USE_TBB
    xt::xarray<value_type> even;
    xt::xarray<value_type> odd;
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
    auto spectrum = xt::xtensor<value_type, 1>::from_shape({N});
    xt::view(spectrum, xt::range(0, N / 2)) = first_half;
    xt::view(spectrum, xt::range(N / 2, N)) = second_half;
    return spectrum;
  }
}

template <class E,
          typename std::enable_if<
              !xtl::is_complex<typename std::decay<E>::type::value_type>::value,
              bool>::type = true>
inline auto fft(E &&e) {
  using value_type = typename std::decay<E>::type::value_type;
  return fft(xt::cast<std::complex<value_type>>(e));
}
} // namespace xt::fft
