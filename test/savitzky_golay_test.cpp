

#include "doctest/doctest.h"
#include "xtensor-signal/xtensor_signal.hpp"
#include "xtensor-signal/savitzky_golay.hpp"


namespace xtensor_signal
{
    TEST_SUITE("savitzky-golay")
    {
        TEST_CASE("filter")
        {
            //random values
            xt::xarray<float> data = {
                -1.930768602393657940e-01,
                -1.154657924401388813e+00,
                2.078154006282262767e+00,
                1.216556965350956476e+00,
                2.484586854455282656e-01,
                1.836468974787362485e-01,
                1.503777948531001663e+00,
                -3.054547397849587953e-01,
                1.599710779135675942e+00,
                1.795197067887975562e-01
            };

            //generated from python
            xt::xarray<float> expectation = {
                -2.122379190259373327e-01,
                - 1.039691571682408666e+00,
                1.790738124484921867e+00,
                1.599778141080939964e+00,
                9.872231835611133821e-02,
                6.343365726323109133e-01,
                6.034399242764377780e-01,
                3.697987784059255656e-01,
                1.329609371859030942e+00,
                2.245366080006883092e-01
            };

           auto res = xt::signal::savgol_filter(data, 7, 5);
           
           auto indexes = xt::linspace(std::size_t(0), res.shape(0) - 1, res.shape(0));

           for (const auto& index : indexes)
           {
               REQUIRE(res(index) == doctest::Approx(expectation(index)));
           }
        }

        TEST_CASE("coeffs")
        {
            size_t window = 7;
            size_t order = 5;

             //generated from python
            xt::xarray<float> expectation = {
                2.164502164502158515e-02,
                -1.298701298701295803e-01,
                3.246753246753243394e-01,
                5.670995670995665483e-01,
                3.246753246753247280e-01,
                -1.298701298701295803e-01,
                2.164502164502157475e-02
            };

            auto res = xt::signal::savgol_coeffs(window, order);
            auto indexes = xt::linspace(std::size_t(0), res.shape(0) - 1, res.shape(0));
            for (const auto& index : indexes)
            {
                REQUIRE(res(index) == doctest::Approx(expectation(index)));
            }
        }
    }
}

