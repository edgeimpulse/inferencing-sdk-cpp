
set(MODEL_FOLDER ..)
set(PRJ_INCL
    ${MODEL_FOLDER}
    ${MODEL_FOLDER}/tflite-model
    ${MODEL_FOLDER}/model-parameters 
    ".")

idf_component_register(SRCS "main.cpp" 
        "${MODEL_FOLDER}/tflite-model/tflite_learn_44_13_compiled.cpp"
        INCLUDE_DIRS ${PRJ_INCL})
