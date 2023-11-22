
#include "doctest/doctest.h"
#include "xtensor-signal/fft.hpp"
#include <xtensor/xio.hpp>

TEST_SUITE("fft") {

  TEST_CASE("fft_single") {
    xt::xtensor<float, 1> input = {1,1,1,1,0,0,0,0};
    xt::xtensor<float, 1> expectation = {4.000, 2.613, 0.000, 1.082, 0.000, 1.082, 0.000, 2.613};
    auto result = xt::fft::fft(input);
    REQUIRE(xt::all(xt::isclose(xt::abs(result), expectation, .001)));
  }
  TEST_CASE("fft_double") {
    xt::xtensor<double, 1> input = {1,1,1,1,0,0,0,0};
    xt::xtensor<double, 1> expectation = {4.000, 2.613, 0.000, 1.082, 0.000, 1.082, 0.000, 2.613};
    auto result = xt::fft::fft(input);
    REQUIRE(xt::all(xt::isclose(xt::abs(result), expectation, .001)));
  }
  TEST_CASE("fft_csingle") {
    xt::xtensor<std::complex<float>, 1> input = {1,1,1,1,0,0,0,0};
    xt::xtensor<float, 1> expectation = {4.000, 2.613, 0.000, 1.082, 0.000, 1.082, 0.000, 2.613};
    auto result = xt::fft::fft(input);
    REQUIRE(xt::all(xt::isclose(xt::abs(result), expectation, .001)));
  }
  TEST_CASE("fft_cdouble") {
    xt::xtensor<std::complex<double>, 1> input = {1,1,1,1,0,0,0,0};
    xt::xtensor<double, 1> expectation = {4.000, 2.613, 0.000, 1.082, 0.000, 1.082, 0.000, 2.613};
    auto result = xt::fft::fft(input);
    REQUIRE(xt::all(xt::isclose(xt::abs(result), expectation, .001)));
  }
}
