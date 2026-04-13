# Run this script to convert the edge-impulse-sdk folder into a library that can be consumed by the Arduino IDE
# it renames files (e.g. *.cpp to *.c), removes features (uTensor), and updates include paths

# exit when any command fails
set -e

cleanup() {
    echo ""
    echo "Terminated by user"
    exit 1
}
trap cleanup INT TERM

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

if [[ "$OSTYPE" == "darwin"* ]]; then
    SEDCMD="sed -i '' -e"
    ECHOCMD="echo"
    LC_CTYPE=C
    LANG=C
else
    SEDCMD="sed -i -e"
    ECHOCMD="echo -e"
fi

rm -rf $SCRIPTPATH/tensorflow/lite/micro/mbed/
find "$SCRIPTPATH/porting/" -mindepth 1 -maxdepth 1 ! -name 'arduino' ! -name 'particle' ! -name 'espressif' -type d -exec rm -rf {} +

rm -rf $SCRIPTPATH/classifier/ei_run_classifier_c*
rm -rf $SCRIPTPATH/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal2.S
rm -rf $SCRIPTPATH/third_party/arc_mli_package/

# rename all .cc files to .cpp, and do an inplace change of the headers
find . -name '*.cc' -exec sh -c 'mv "$0" "${0%.cc}.cpp"' {} \;

# make sure that abs is undefined on arduino
find $SCRIPTPATH/ -name 'compatibility.h' -exec bash -c "$SEDCMD 's/#include <cstdint>/#include <cstdint>\\
#include \"edge-impulse-sdk\/tensorflow\/lite\/portable_type_to_tflitetype.h\"/' {}" {} \;
find $SCRIPTPATH/ -name 'micro_utils.h' -exec bash -c "$SEDCMD 's/#include <cstdint>/#include <cstdint>\\
#include \"edge-impulse-sdk\/tensorflow\/lite\/portable_type_to_tflitetype.h\"/' {}" {} \;

# remove all the -e files
find $SCRIPTPATH/ -name "*-e" -exec rm -f {} \;
