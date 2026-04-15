# Hello World Img

Offline image classification example for this ESP-IDF component.

The app loads a bundled test image, runs a single inference, logs memory usage and timing information, and reports the top prediction.

## Notes

- Impulse: https://studio.edgeimpulse.com/studio/904830/impulse/1/deployment
- Build as Unoptimized (float32), since the Quantized (int8) build does not work on ESP32-S3 when HW acceleration is enabled.

## Build

Note: Default build (ie. when no `PRJ_BUILD_TARGET` is provided) is for hardware.

For Wokwi:

```bash
rm -f sdkconfig
PRJ_BUILD_TARGET=wokwi idf.py reconfigure build
```

For ESP32-S3:

```bash
rm -f sdkconfig
PRJ_BUILD_TARGET=esp32s3 idf.py reconfigure build
```

If `idf.py` is unavailable, source `~/.espressif/v5.5.4/esp-idf/export.sh` first.

## Run

For Wokwi:

```bash
wokwi-cli . --timeout 150000 --fail-text "Backtrace:" --expect-text "main_task: Returned from app_main()"
```

For hardware:

```bash
idf.py flash monitor
```

## Example output

For Wokwi:

``` text
I (1254) main_task: Started on CPU0
I (1264) esp_psram: Reserving pool of 32K of internal memory for DMA/internal allocations
I (1264) main_task: Calling app_main()
I (1264) hello-world-img: Hardware acceleration is disabled for this build.
I (1374) hello-world-img: Edge Impulse standalone inferencing (Espressif ESP32)
I (1374) hello-world-img: Heap (internal): free=176107, largest=81920 bytes
I (1374) hello-world-img: PSRAM: enabled, free=8386140, largest=8257536 bytes
I (1384) hello-world-img: Model arena target: 1106406 bytes
I (114714) hello-world-img: Timing: DSP 176 ms, inference 113157 ms, anomaly 0 ms
I (114724) hello-world-img: Predictions:
I (114724) hello-world-img:   lamp: 0.00009
I (114724) hello-world-img:   plant: 0.99970
I (114724) hello-world-img:   unknown: 0.00020
I (114734) hello-world-img: Top prediction: plant (0.99970)
I (114734) hello-world-img: Expected label: plant -> PASS
I (114744) main_task: Returned from app_main()


Expected text found: "main_task: Returned from app_main()"
TEST PASSED.
```

For hardware:

``` text
I (1347) main_task: Started on CPU0
I (1357) esp_psram: Reserving pool of 32K of internal memory for DMA/internal allocations
I (1357) main_task: Calling app_main()
I (1457) hello-world-img: Edge Impulse standalone inferencing (Espressif ESP32)
I (1457) hello-world-img: Heap (internal): free=176059, largest=81920 bytes
I (1457) hello-world-img: PSRAM: enabled, free=8386140, largest=8257536 bytes
I (1457) hello-world-img: Model arena target: 1106406 bytes
I (12677) hello-world-img: Timing: DSP 22 ms, inference 11184 ms, anomaly 0 ms
I (12677) hello-world-img: Predictions:
I (12677) hello-world-img:   lamp: 0.00009
I (12677) hello-world-img:   plant: 0.99970
I (12677) hello-world-img:   unknown: 0.00020
I (12687) hello-world-img: Top prediction: plant (0.99970)
I (12687) hello-world-img: Expected label: plant -> PASS
I (12697) main_task: Returned from app_main()
```
