
#ifdef XTENSOR_USE_TBB
    #include <oneapi/tbb.h>
#endif
#include <xtensor/xview.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>

#include <stdexcept>

namespace xt::fft
{
    template<class E>
    inline auto fft(E&& e)
    {
        using namespace xt::placeholders;
        using namespace std::complex_literals;
        auto N = e.size();
        auto pi = xt::numeric_constants<double>::PI;
        auto ev = xt::eval(e);
        if (N <= 1)
        {
            return ev;
        }
        else
        {
    #ifdef XTENSOR_USE_TBB
            xt::xarray<std::complex<float>> even;
            xt::xarray<std::complex<float>> odd;
            oneapi::tbb::parallel_invoke([&]{even = fft(xt::view(ev, xt::range(0, _, 2))); },
                [&]{odd = fft(xt::view(ev, xt::range(1, _, 2)));});
    #else
            auto even = fft(xt::view(ev, xt::range(0, _, 2)));
            auto odd = fft(xt::view(ev, xt::range(1, _, 2)));
    #endif

            auto range = xt::arange<double>(N / 2);
            auto exp = xt::exp(-2i * pi * range / N);
            auto t = exp * odd;
            auto first_half = even + t;
            auto second_half = even - t;
            auto spectrum = xt::xtensor<std::complex<float>, 1>::from_shape({N});
            xt::view(spectrum, xt::range(0, N / 2)) = first_half;
            xt::view(spectrum, xt::range(N / 2, N)) = second_half;
            return spectrum;
        }
    }
}

