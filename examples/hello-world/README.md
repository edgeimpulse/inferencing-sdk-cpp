# Hello World

Offline keyword spotting example for this ESP-IDF component.

The app loads a bundled offline audio sample, runs a single inference, logs basic model information, and prints the classifier output.

## Build

Note: Default build (ie. when no PRJ_BUILD_TARGET is provided) is for hardware.

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
wokwi-cli . --timeout 120000 --fail-text "Backtrace:" --expect-text "main_task: Returned from app_main()"
```

For hardware:

```bash
idf.py flash monitor
```

## Example output

For Wokwi:

``` text
<removed>
I (256) main_task: Started on CPU0
I (266) main_task: Calling app_main()
W (266) hello-world: Hardware acceleration is disabled for this build. This may cause the classifier to run significantly slower than expected.
I (306) hello-world: Model: Demo: Keyword Spotting
I (306) hello-world: Labels: 3
Timing: DSP 5911 ms, inference 301 ms, anomaly 0 ms, postprocessing 55 us
#Classification predictions:
  helloworld: 0.832031
  noise: 0.000000
  unknown: 0.167969
I (6526) main_task: Returned from app_main()


Expected text found: "main_task: Returned from app_main()"
TEST PASSED.
```

   
For hardware:

``` text
I (276) main_task: Started on CPU0
I (286) main_task: Calling app_main()
I (286) hello-world: Model: Demo: Keyword Spotting
I (286) hello-world: Labels: 3
Timing: DSP 116 ms, inference 3 ms, anomaly 0 ms, postprocessing 36 us
#Classification predictions:
  helloworld: 0.980469
  noise: 0.019531
  unknown: 0.000000 
I (416) main_task: Returned from app_main()

Done
```