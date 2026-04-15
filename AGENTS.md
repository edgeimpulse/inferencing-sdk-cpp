# AGENTS.md

## Build and verify
- This repository is an ESP-IDF component, not a standalone app. Do not validate changes from the repo root; build from `examples/hello-world` or `examples/hello-world-img`.
- If `idf.py` is unavailable, source `~/.espressif/v5.5.4/esp-idf/export.sh`.
- Before `idf.py reconfigure`, remove the example's generated `sdkconfig` so target-specific defaults are reapplied cleanly.
- Focused verification commands:
  - `examples/hello-world`: `rm -f sdkconfig && PRJ_BUILD_TARGET=esp32s3 idf.py reconfigure build`
  - `examples/hello-world`: `rm -f sdkconfig && PRJ_BUILD_TARGET=wokwi idf.py reconfigure build`
  - `examples/hello-world-img`: `rm -f sdkconfig && PRJ_BUILD_TARGET=esp32s3 idf.py reconfigure build`
  - `examples/hello-world-img`: `rm -f sdkconfig && PRJ_BUILD_TARGET=wokwi idf.py reconfigure build`
- `PRJ_BUILD_TARGET` selects extra sdkconfig defaults (`sdkconfig.esp32s3`, `sdkconfig.wokwi`). Verify target-specific changes with the matching env var.

## Repo structure
- The packaged SDK sources are vendored under `src/edge-impulse-sdk/`. The root `CMakeLists.txt` compiles that tree plus `src/edge-impulse-sdk/porting/espressif/`.
- Example apps are the real local entrypoints:
  - `examples/hello-world`: offline keyword spotting sample
  - `examples/hello-world-img`: image/classification sample with heavier memory needs

## Hardware accel and target quirks
- `CONFIG_EI_DISABLE_HW_ACCEL` controls both `EI_CLASSIFIER_TFLITE_ENABLE_ESP_NN` and `EIDSP_USE_ESP_DSP`.
- `examples/hello-world/CMakeLists.txt` hard-disables HW acceleration at the project level; Kconfig alone will not re-enable it there.
- `examples/hello-world-img/README.md` notes that the quantized int8 impulse does not work on ESP32-S3 when HW acceleration is enabled; use an unoptimized float32 impulse for that example.


## Packaging and release-sensitive files
- Keep these aligned when changing component metadata, packaging, or published behavior:
  - `README.md`
  - `component.mk`
  - `.github/workflows/upload_component.yml`
- Publishing is tag-driven: pushing a `v*` tag triggers `.github/workflows/upload_component.yml`, which uploads namespace `ozanoner` / component `edgeimpulse-inference-sdk` using `IDF_COMPONENT_API_TOKEN`.

## Editing cautions
- Avoid any refactors in `src/edge-impulse-sdk/`; this repo is primarily a packaging fork over vendored upstream code.
- Do not run repo-wide formatting by default. `.clang-format-ignore` excludes all of `src/` and generated model directories (`**/model-parameters/**`, `**/tflite-model/**`).
