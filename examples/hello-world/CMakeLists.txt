# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

# Remove this when using the component in your application
set(EXTRA_COMPONENT_DIRS "../..")

# Default to building for esp32s3, but allow overriding with environment variable
set(PRJ_BUILD_TARGET "esp32s3")
set(PRJ_TARGET_SDKCONFIG "${CMAKE_SOURCE_DIR}/sdkconfig.${PRJ_BUILD_TARGET}")

# Select wokwi or real hardware build based on environment variable PRJ_BUILD_TARGET
if(DEFINED ENV{PRJ_BUILD_TARGET})
    set(PRJ_BUILD_TARGET "$ENV{PRJ_BUILD_TARGET}")
    set(PRJ_TARGET_SDKCONFIG "${CMAKE_SOURCE_DIR}/sdkconfig.${PRJ_BUILD_TARGET}")

    if(NOT EXISTS "${PRJ_TARGET_SDKCONFIG}")
        message(FATAL_ERROR
            "PRJ_BUILD_TARGET is set to '${PRJ_BUILD_TARGET}', "
            "but file '${PRJ_TARGET_SDKCONFIG}' does not exist."
        )
    endif()
endif()

message(STATUS ">>> PRJ_BUILD_TARGET=${PRJ_BUILD_TARGET}")
set(SDKCONFIG_DEFAULTS "sdkconfig.defaults" "${PRJ_TARGET_SDKCONFIG}")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(hello-world)
