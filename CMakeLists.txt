############################################################################
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)
project(xtensor-signal)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(XTENSOR_SIGNAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versioning
# ==========

set(XTENSOR_SIGNAL_CONFIG_FILE
    "${XTENSOR_SIGNAL_INCLUDE_DIR}/xtensor-signal/xtensor_signal.hpp")
file(STRINGS ${XTENSOR_SIGNAL_CONFIG_FILE} xtensor_signal_version_defines
    REGEX "#define XTENSOR_SIGNAL_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${xtensor_signal_version_defines})
    if(ver MATCHES "#define XTENSOR_SIGNAL_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(XTENSOR_SIGNAL_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
    ${XTENSOR_SIGNAL_VERSION_MAJOR}.${XTENSOR_SIGNAL_VERSION_MINOR}.${XTENSOR_SIGNAL_VERSION_PATCH})
message(STATUS "xtensor-signal v${${PROJECT_NAME}_VERSION}")

# Dependencies
# ============

set(xsimd_REQUIRED_VERSION 9.0.1)
if(TARGET xsimd)
    set(xsimd_VERSION ${XSIMD_VERSION_MAJOR}.${XSIMD_VERSION_MINOR}.${XSIMD_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if(${xsimd_VERSION} VERSION_LESS ${xsimd_REQUIRED_VERSION})
        message(ERROR "Mismatch xsimd versions. Found '${xsimd_VERSION}' but requires: '${xsimd_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xsimd v${xsimd_VERSION}")
    endif()
else()
    find_package(xsimd ${xsimd_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xsimd: ${xsimd_INCLUDE_DIRS}/xsimd")
endif()


set(xtensor_REQUIRED_VERSION 0.24.6)
if(TARGET xtensor)
    set(xtensor_VERSION ${XTENSOR_VERSION_MAJOR}.${XTENSOR_VERSION_MINOR}.${XTENSOR_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if( NOT ${xtensor_VERSION} VERSION_GREATER_EQUAL ${xtensor_REQUIRED_VERSION})
        message(ERROR "Mismatch xtensor versions. Found '${xtensor_VERSION}' but requires: '${xtensor_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xtensor v${xtensor_VERSION}")
    endif()
else()
    find_package(xtensor ${xtensor_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}/xtensor")
endif()

# Build
# =====

set(XTENSOR_SIGNAL_HEADERS
    ${XTENSOR_SIGNAL_INCLUDE_DIR}/xtensor-signal/xtensor_signal.hpp
    ${XTENSOR_SIGNAL_INCLUDE_DIR}/xtensor-signal/find_peaks.hpp
    )

add_library(xtensor-signal INTERFACE)

target_include_directories(xtensor-signal INTERFACE
    $<BUILD_INTERFACE:${XTENSOR_SIGNAL_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(xtensor-signal INTERFACE xtensor xsimd)

OPTION(BUILD_TESTS "xtensor test suite" OFF)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS xtensor-signal
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(FILES ${XTENSOR_SIGNAL_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xtensor-signal)

set(XTENSOR_SIGNAL_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for xtensor-signalConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${XTENSOR_SIGNAL_CMAKECONFIG_INSTALL_DIR})

# xtensor-signal is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from xtensor-signalConfigVersion.cmake so that an xtensor-signalConfig.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_XTENSOR_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_XTENSOR_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${XTENSOR_SIGNAL_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${XTENSOR_SIGNAL_CMAKECONFIG_INSTALL_DIR})
