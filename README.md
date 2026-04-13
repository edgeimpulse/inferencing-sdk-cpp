[![Component Registry](https://components.espressif.com/components/ozanoner/edgeimpulse-inference-sdk/badge.svg)](https://components.espressif.com/components/ozanoner/edgeimpulse-inference-sdk)

# EdgeImpulse Inference SDK for ESP32


## Overview

EdgeImpulse Inference SDK for ESP32 packages this fork of the Edge Impulse C++ inferencing SDK as an ESP-IDF component.

The component bundles the runtime, DSP helpers, TensorFlow Lite Micro sources, CMSIS libraries, and the Espressif-specific integrations already vendored in this repository, including the embedded ESP-DSP and ESP-NN ports used by the SDK on ESP32 targets.

This repository is intended for ESP-IDF projects that already have model-specific Edge Impulse code or exported impulse assets and want to consume the shared SDK through the Espressif Component Registry.


## Documentation

- Component Registry: https://components.espressif.com/components/ozanoner/edgeimpulse-inference-sdk
- Fork repository: https://github.com/ozanoner/edgeimpulse-inferencing-sdk-cpp
- Upstream SDK: https://github.com/edgeimpulse/inferencing-sdk-cpp
- ESP-IDF Getting Started: https://docs.espressif.com/projects/esp-idf/en/latest/get-started/

## Installation and Usage

This package is consumed like any other component in the [ESP-IDF build system](https://docs.espressif.com/projects/esp-idf/en/latest/api-guides/build-system.html).

The recommended way to use it is through the [IDF Component Registry](https://components.espressif.com/components/ozanoner/edgeimpulse-inference-sdk).

### Adding the component to an existing project

In the project directory, run:
```bash
idf.py add-dependency "ozanoner/edgeimpulse-inference-sdk"
```
This adds the SDK component as a dependency of your `main` component. You can also add it manually in your project's `idf_component.yml`.

Example dependency declaration:

```yaml
dependencies:
  ozanoner/edgeimpulse-inference-sdk: "^0.1.0"
```

### Integrating the SDK

This component packages the shared SDK sources only. Your application still needs the model-specific files generated for your Edge Impulse project, along with the code that calls into the runtime.

The vendored sources in this fork are laid out under `src/` and include:

- `classifier/` for inference entry points and model-facing APIs
- `dsp/` for DSP helpers and signal processing utilities
- `tensorflow/` and `third_party/` for TensorFlow Lite Micro and bundled dependencies
- `CMSIS/` for CMSIS-DSP and CMSIS-NN sources
- `porting/espressif/` for ESP32-specific integrations

This fork does not currently publish downloadable registry examples from the root `examples/` directory.

### Building and running an ESP-IDF project

```bash
idf.py reconfigure
idf.py build
idf.py -p PORT flash monitor
```

where `PORT` is the UART port name of your development board, such as `/dev/ttyUSB0` or `COM1`.

Note that you need to set up ESP-IDF before building the project. Refer to the [ESP-IDF Getting Started Guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html) if you don't have the environment set up yet.

## Reporting Issues

If you find a packaging or ESP32 integration issue in this fork, please use the [Issues](https://github.com/ozanoner/edgeimpulse-inferencing-sdk-cpp/issues) section on GitHub.

If the problem is clearly upstream to the shared SDK rather than this ESP-IDF packaging fork, include that context in the issue and link the upstream project where appropriate.

## Contributing

Please check [CONTRIBUTING.md](CONTRIBUTING.md) if you'd like to contribute to this fork.

## License

This repository is derived from the Edge Impulse inferencing SDK and includes bundled third-party source code. See `LICENSE`, `LICENSE.3-clause-bsd-clear`, and any license files shipped with bundled dependencies for the applicable terms.
