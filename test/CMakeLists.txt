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

    find_package(xtensor-io REQUIRED)
    set(XTENSOR_IO_INCLUDE_DIR ${xtensor_io_INCLUDE_DIRS})
endif ()

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

set (CMAKE_CXX_STANDARD 17)

find_package(ZLIB REQUIRED)
find_package(doctest REQUIRED)
find_package(Threads)

set(XTENSOR_SIGNAL_TESTS
    main.cpp
    test_config.cpp
    find_peaks_test.cpp
)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} /MP /bigobj")
endif()

file(COPY "test_data" DESTINATION "${CMAKE_BINARY_DIR}/test")

add_executable(test_xtensor_signal ${XTENSOR_SIGNAL_TESTS} ${XTENSOR_SIGNAL_HEADERS})
target_link_libraries(test_xtensor_signal PRIVATE ZLIB::ZLIB xtensor-signal doctest::doctest ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(xtest COMMAND ./test_xtensor_signal DEPENDS test_xtensor_signal)
