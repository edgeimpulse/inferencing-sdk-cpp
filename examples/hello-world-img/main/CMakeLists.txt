
set(MODEL_FOLDER ..)
set(PRJ_INCL
    ${MODEL_FOLDER}
    ${MODEL_FOLDER}/tflite-model
    ${MODEL_FOLDER}/model-parameters 
    ".")

if(NOT CMAKE_BUILD_EARLY_EXPANSION)
if(CONFIG_EI_DISABLE_HW_ACCEL)
    add_definitions(-DEI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=0)
    add_definitions(-DEIDSP_USE_ESP_DSP=0)
else()
    add_definitions(-DEI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=1)
    add_definitions(-DEIDSP_USE_ESP_DSP=1)
endif()
endif()

idf_component_register(SRCS "main.cpp" 
        "${MODEL_FOLDER}/tflite-model/tflite_learn_904830_19_compiled.cpp"
        INCLUDE_DIRS ${PRJ_INCL})
