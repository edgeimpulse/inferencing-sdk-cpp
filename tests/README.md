# Software RFFT plan regression test

From the repository root, on a host with GCC:

```sh
build_dir=$(mktemp -d)
ln -s "$PWD" "$build_dir/edge-impulse-sdk"
g++ -std=c++11 -O2 -I"$build_dir" tests/rfft_plan.cpp \
    dsp/kissfft/kiss_fft.cpp dsp/kissfft/kiss_fftr.cpp -o "$build_dir/rfft-plan"
"$build_dir/rfft-plan"
```

The test compares cached and uncached power spectra and orthonormal DCT results
bit for bit for silence and generated signals. It exercises repeated use, FFT
size changes, allocation failure/retry, and scoped allocation cleanup.
This exercises the generic software backend; target performance and hardware
backends need separate validation.
