
#include "doctest/doctest.h"
#include "xtensor-signal/xtensor_signal.hpp"
#include "xtensor-signal/find_peaks.hpp"

namespace xtensor_signal
{
    TEST_SUITE("find_peaks")
    {

		TEST_CASE("FindSinglePeak")
		{
			//generate a gaussian curve
			//defines
			auto mu = 1;
			auto sigma = 2;
			auto start = -5;
			auto end = 5;
			auto numsamples = 10000;
			auto x = xt::linspace<double>(start, end, numsamples);
			auto y = (1 / (sigma * std::sqrt(2 * xt::numeric_constants<double>::PI))) * xt::exp(-.5 * xt::pow((x - mu), 2) / (std::pow(sigma, 2)));

			//we should have a single peak at x = 1
			auto peaks = xt::signal::find_peaks(y);

			//assert that we have one peak
			REQUIRE(peaks.shape(0) == doctest::Approx(1.0).epsilon(.001));

			//assert that the value is x = 0
			REQUIRE(x(peaks(0)) == doctest::Approx(1.0).epsilon(.001));
		}

		TEST_CASE("RandNoise")
		{
			//run through some random noise and make sure the
			// algorithm doesn't dies on random noise.
			//defines
			auto start = -5;
			auto end = 5;
			auto numsamples = 10000;
			auto x = xt::linspace<double>(start, end, numsamples);
			auto y = xt::random::randn<double>(x.shape());

			auto peaks = xt::signal::find_peaks(y);
		}

		TEST_CASE("BiModal")
		{
			//generate a gaussian curve
			//defines
			auto mu = 1;
			auto sigma = 2;
			auto start = -5;
			auto end = 15;
			auto numsamples = 10000;
			auto x = xt::linspace<double>(start, end, numsamples);
			xt::xarray<double> y1 = (1 / (sigma * std::sqrt(2 * xt::numeric_constants<double>::PI))) * xt::exp(-.5 * xt::pow((x - mu), 2) / (std::pow(sigma, 2)));

			mu = 10;
			xt::xarray<double> y2 = (1 / (sigma * std::sqrt(2 * xt::numeric_constants<double>::PI))) * xt::exp(-.5 * xt::pow((x - mu), 2) / (std::pow(sigma, 2)));

			auto y = y1 + y2;

			//we should have a single peak at x = 1
			auto peaks = xt::signal::find_peaks(y);

			//assert that we have two peaks
			REQUIRE(peaks.shape(0) == doctest::Approx(2));

			//assert that the value is x = 1
			auto res = x(peaks(0));
			REQUIRE(res == doctest::Approx(1.0).epsilon(.001));

			//assert that the value is x = 10
			res = x(peaks(1));
			REQUIRE(res == doctest::Approx(10.0).epsilon(.001));
		}
    }
}