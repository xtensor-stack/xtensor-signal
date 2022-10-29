

#include "doctest/doctest.h"
#include "xtensor-signal/xtensor_signal.hpp"
#include "xtensor-signal/savitzky_golay.hpp"


namespace xtensor_signal
{
    TEST_SUITE("savitzky-golay")
    {
        TEST_CASE("first test")
        {
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

           auto res = savgol_filter(data, 3, 5);

           std::cout << res;
        }
    }
}

