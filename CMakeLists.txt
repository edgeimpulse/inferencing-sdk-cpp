set(EI_COMPONENT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(EI_SDK_FOLDER ${EI_COMPONENT_DIR}/src/edge-impulse-sdk)

if(NOT CMAKE_BUILD_EARLY_EXPANSION)
if(CONFIG_EI_DISABLE_HW_ACCEL)
add_definitions(-DEI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=0)
add_definitions(-DEIDSP_USE_ESP_DSP=0)
else()
add_definitions(-DEI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=1)
add_definitions(-DEIDSP_USE_ESP_DSP=1)
endif()
endif()

set(include_dirs
    ${EI_SDK_FOLDER}
    ${EI_COMPONENT_DIR}/src
    ${EI_SDK_FOLDER}/classifier
    ${EI_SDK_FOLDER}/dsp
    ${EI_SDK_FOLDER}/porting
    ${EI_SDK_FOLDER}/tensorflow
    ${EI_SDK_FOLDER}/third_party
    ${EI_SDK_FOLDER}/third_party/ruy
    ${EI_SDK_FOLDER}/third_party/gemmlowp
    ${EI_SDK_FOLDER}/third_party/flatbuffers/include
)

include(${EI_SDK_FOLDER}/cmake/utils.cmake)

RECURSIVE_FIND_FILE_EXCLUDE_DIR(SOURCE_FILES "${EI_SDK_FOLDER}" "CMSIS" "*.cpp")
RECURSIVE_FIND_FILE_EXCLUDE_DIR(CC_FILES "${EI_SDK_FOLDER}" "CMSIS" "*.cc")
RECURSIVE_FIND_FILE_EXCLUDE_DIR(C_FILES "${EI_SDK_FOLDER}" "CMSIS" "*.c")
list(APPEND C_FILES "${EI_SDK_FOLDER}/CMSIS/DSP/Source/StatisticsFunctions/arm_rms_f32.c")
RECURSIVE_FIND_FILE(PORTING_FILES "${EI_SDK_FOLDER}/porting/espressif" "*.cpp")


list(APPEND SOURCE_FILES ${C_FILES})
list(APPEND SOURCE_FILES ${CC_FILES})
list(APPEND SOURCE_FILES ${PORTING_FILES})

idf_component_register(SRCS ${SOURCE_FILES}
                       INCLUDE_DIRS ${include_dirs}
                       REQUIRES esp_timer)

target_compile_options(${COMPONENT_LIB} PRIVATE -std=gnu++17)
idf_build_set_property(COMPILE_OPTIONS "-fdiagnostics-color=always" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-Wno-unused-variable" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-Wno-maybe-uninitialized" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-Wno-error=format=" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-Wno-unused-but-set-parameter" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-Wno-error=nonnull-compare" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-Wno-error=stringop-truncation" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-Wno-error=uninitialized" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-Wno-stringop-truncation" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-Wno-deprecated-declarations" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-fno-if-conversion" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-Wno-error=stringop-overflow=" APPEND)                       