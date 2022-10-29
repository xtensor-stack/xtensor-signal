############################################################################
# Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xtensor-signal-test)

    find_package(xsimd REQUIRED)
    set(XSIMD_INCLUDE_DIR ${xsimd_INCLUDE_DIRS})

    find_package(xtensor REQUIRED CONFIG)
    set(XTENSOR_INCLUDE_DIR ${xtensor_INCLUDE_DIRS})

    find_package(xtensor-signal REQUIRED CONFIG)
    set(XTENSOR_SIGNAL_INCLUDE_DIR ${xtensor-signal_INCLUDE_DIRS})
endif ()

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion -fvisibility=hidden")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)

    if (HAS_CPP14_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        message(FATAL_ERROR "Unsupported compiler -- xtensor-signal requires C++14 support!")
    endif()
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

find_package(doctest REQUIRED)
find_package(Threads)

include_directories(${GTEST_INCLUDE_DIRS})

set(XTENSOR_SIGNAL_TESTS
    main.cpp
    test_config.cpp
    savitzky_golay_test
)

add_executable(test_xtensor_signal ${XTENSOR_SIGNAL_TESTS} ${XTENSOR_SIGNAL_HEADERS})
target_link_libraries(test_xtensor_signal PRIVATE xtensor-signal doctest::doctest ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(xtest COMMAND ./test_xtensor_signal DEPENDS test_xtensor_signal)

