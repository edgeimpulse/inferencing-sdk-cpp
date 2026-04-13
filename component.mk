COMPONENT_ADD_INCLUDEDIRS := src \
                            src/CMSIS/Core/Include \
                            src/CMSIS/DSP/Include \
                            src/CMSIS/NN/Include \
                            src/third_party/flatbuffers/include \
                            src/porting/espressif/ESP-NN \
                            src/porting/espressif/ESP-NN/include \
                            src/porting/espressif/esp-dsp/modules/common/include \
                            src/porting/espressif/esp-dsp/modules/fft/include

COMPONENT_PRIV_INCLUDEDIRS := src/porting/espressif/ESP-NN/src/common

_EI_SOURCE_ROOTS := src/classifier \
                    src/dsp \
                    src/tensorflow \
                    src/third_party \
                    src/CMSIS/DSP/Source \
                    src/CMSIS/NN/Source \
                    src/porting/espressif

# Collect only directories that actually contain source files used by the ESP32 fork.
COMPONENT_SRCDIRS := $(sort $(foreach dir,$(_EI_SOURCE_ROOTS),$(shell if [ -d "$(COMPONENT_PATH)/$(dir)" ]; then find "$(COMPONENT_PATH)/$(dir)" -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.s' -o -name '*.S' \) ! -name 'test_*' ! -name '*_test*' -printf '%h\n'; fi | sed 's|^$(COMPONENT_PATH)/||')))

