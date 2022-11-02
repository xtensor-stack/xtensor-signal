/***************************************************************************
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "doctest/doctest.h"

#include "xtensor-signal/xtensor_signal.hpp"

namespace xt
{
    TEST_SUITE("config")
    {
        TEST_CASE("first test")
        {
            int i = 0;
            REQUIRE(i == 0);
        }
    }
}

