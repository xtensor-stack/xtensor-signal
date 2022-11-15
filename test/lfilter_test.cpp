
#include <doctest/doctest.h>
#include <xtensor-signal/lfilter.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xio.hpp>

namespace xtensor_signal
{
    TEST_SUITE("lfilter")
    {
        TEST_CASE("filter")
        {
            //define the signal 
           	xt::xtensor<float, 1> sig = {-0.917843918645, 0.141984778794,1.20536903482,0.190286794412,-0.662370894973,-1.00700480494,
                -0.404707073677,0.800482325044,0.743500089861,1.01090520172,0.741527555207,
                0.277841675195,0.400833448236,-0.2085993586,-0.172842103641,-0.134316096293,
                0.0259303398477,0.490105989562,0.549391221511,0.9047198589};

            xt::xtensor<float, 1> expectation = {-0.152974,-0.435258,-0.136043,0.697503,0.656445,
                                                -0.435483,-1.08924,-0.537677,0.51705,1.05225,0.961854,0.69569,0.424356,
                                                0.196262,-0.0278351,-0.211722,-0.174746,0.0692584,0.385446,0.651771};

            //Constants for a Butterworth filter (order 3, low pass)
            xt::xtensor<float, 1> a = {1.00000000, -2.77555756e-16, 3.33333333e-01, -1.85037171e-17};
            xt::xtensor<float, 1> b = {0.16666667, 0.5, 0.5, 0.16666667};
        
            auto res = xt::signal::lfilter(b, a, sig);

            auto indexes = xt::linspace(std::size_t(0), res.shape(0) - 1, res.shape(0));

            std::cout << res << std::endl;
           for (const auto& index : indexes)
           {
               REQUIRE(res(index) == doctest::Approx(expectation(index)));
           }
        }
    }
}

