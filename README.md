# EdgeImpulse Inference SDK for ESP32

## Overview

This repository packages the Edge Impulse C++ inferencing SDK as an ESP-IDF component for ESP32 targets.

The shared SDK is vendored under `src/edge-impulse-sdk/`, and the component builds that tree together with the Espressif port in `src/edge-impulse-sdk/porting/espressif/`.

This repo is primarily a packaging fork. It is useful when your ESP-IDF project already contains model-specific Edge Impulse export files and you want to consume the shared runtime through the Espressif Component Registry.

## Current State

- ESP-IDF component metadata is defined in `idf_component.yml`.
- The registry package currently targets ESP-IDF `>=5.5` and is developed against ESP-IDF `v5.5.4`.
- The component declares `espressif/esp-dsp` and `espressif/esp-nn` as dependencies.
- Hardware acceleration can be disabled with `CONFIG_EI_DISABLE_HW_ACCEL`, which controls both `EI_CLASSIFIER_TFLITE_ENABLE_ESP_NN` and `EIDSP_USE_ESP_DSP`.
- The repository includes two local example apps under `examples/` for validation and reference:
  - `examples/hello-world`: offline keyword spotting sample
  - `examples/hello-world-img`: offline image classification sample with higher memory needs


## Installation

### Staging prerelease

Use the staging registry while validating prerelease uploads:

```yaml
dependencies:
  ozanoner/edgeimpulse-inference-sdk:
    version: "0.1.0-rc1"
    registry_url: https://components-staging.espressif.com
```

This repository uses plain component versions such as `0.1.0` and prerelease versions such as `0.1.0-rc1`.

### Production release

After the stable release is published, add the component from the IDF Component Registry:

```bash
idf.py add-dependency "ozanoner/edgeimpulse-inference-sdk"
```

Or declare it directly in your project's `idf_component.yml`:

```yaml
dependencies:
  ozanoner/edgeimpulse-inference-sdk: "^0.1.0"
```

For maintainers, prereleases should use matching Git tags such as `v0.1.0-rc1`.

## Using the Component

This component packages the shared SDK only. Your application still needs the model-specific files exported from Edge Impulse, such as the generated model sources, model parameters, and the application code that calls `run_classifier()`.

The core integration point matches the local examples:

```cpp
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

signal_t signal{};
signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
signal.get_data = &get_signal_data;

ei_impulse_result_t result{};
EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
```

See `examples/hello-world` and `examples/hello-world-img` for complete ESP-IDF applications that wire the classifier into offline audio and image sample pipelines.


## Local Example Builds

This repository is not a standalone firmware app at the root. Build from one of the example projects instead.

If `idf.py` is not already on your shell path, load the ESP-IDF environment first:

```bash
source ~/.espressif/v5.5.4/esp-idf/export.sh
```

### `examples/hello-world`

This example runs keyword spotting from an offline audio sample.   

Build for ESP32-S3:

```bash
rm -f sdkconfig && PRJ_BUILD_TARGET=esp32s3 idf.py reconfigure build
```

Build for Wokwi:

```bash
rm -f sdkconfig && PRJ_BUILD_TARGET=wokwi idf.py reconfigure build
```


### `examples/hello-world-img`

This example runs image classification from offline feature data.   

Build for ESP32-S3:

```bash
rm -f sdkconfig && PRJ_BUILD_TARGET=esp32s3 idf.py reconfigure build
```

Build for Wokwi:

```bash
rm -f sdkconfig && PRJ_BUILD_TARGET=wokwi idf.py reconfigure build
```

Notes:

- It enables PSRAM-oriented configuration in its sdkconfig defaults.
- On ESP32-S3, the quantized `int8` impulse does not currently work when hardware acceleration is enabled. Use an unoptimized `float32` impulse for this example.

## References

- Staging registry: https://components-staging.espressif.com/
- Production registry: https://components.espressif.com/
- Fork repository: https://github.com/ozanoner/edgeimpulse-inferencing-sdk-cpp
- Upstream SDK: https://github.com/edgeimpulse/inferencing-sdk-cpp
- ESP-IDF Getting Started: https://docs.espressif.com/projects/esp-idf/en/latest/get-started/


## Reporting Issues

If you find a packaging or ESP32 integration issue in this fork, open an issue at https://github.com/ozanoner/edgeimpulse-inferencing-sdk-cpp/issues.

If the issue appears to belong to the upstream shared SDK instead of this ESP-IDF packaging fork, include that context and link the upstream project when possible.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This repository is derived from the Edge Impulse inferencing SDK and includes bundled third-party source code. See `LICENSE`, `LICENSE.3-clause-bsd-clear`, and any license files shipped with bundled dependencies for the applicable terms.
