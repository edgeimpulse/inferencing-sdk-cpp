cmake_minimum_required(VERSION 3.16)

set(EI_COMPONENT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(EI_SDK_FOLDER ${EI_COMPONENT_DIR}/src/edge-impulse-sdk)

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


if(CONFIG_EI_DISABLE_HW_ACCEL)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC
        EI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=0
        EIDSP_USE_ESP_DSP=0
    )
else()
    target_compile_definitions(${COMPONENT_LIB} PUBLIC
        EI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=1
        EIDSP_USE_ESP_DSP=1
    )
endif()

target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_17)
target_compile_options(${COMPONENT_LIB} PRIVATE
     -Wno-error=missing-field-initializers
     -Wno-error=unused-function
     -Wno-maybe-uninitialized
     -Wno-missing-field-initializers
     -Wno-unused-function
     -Wno-unused-variable
     -Wno-deprecated-declarations
)
