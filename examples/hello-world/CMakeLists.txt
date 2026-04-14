# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS "../..")
set (CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{PRJ_BUILD_TARGET} AND NOT "$ENV{PRJ_BUILD_TARGET}" STREQUAL "")
    set(PRJ_BUILD_TARGET "$ENV{PRJ_BUILD_TARGET}")
    set(PRJ_TARGET_SDKCONFIG "${CMAKE_SOURCE_DIR}/sdkconfig.${PRJ_BUILD_TARGET}")

    if(EXISTS "${PRJ_TARGET_SDKCONFIG}")
        message(STATUS "PRJ_BUILD_TARGET=${PRJ_BUILD_TARGET}")
        message(STATUS "Using extra sdkconfig defaults: ${PRJ_TARGET_SDKCONFIG}")
        set(SDKCONFIG_DEFAULTS
            "sdkconfig.defaults"
            "${PRJ_TARGET_SDKCONFIG}"
        )
    else()
        message(FATAL_ERROR
            "PRJ_BUILD_TARGET is set to '${PRJ_BUILD_TARGET}', "
            "but file '${PRJ_TARGET_SDKCONFIG}' does not exist."
        )
    endif()
endif()

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(hello-world)

add_definitions(-DEI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=0)
add_definitions(-DEIDSP_USE_ESP_DSP=0)