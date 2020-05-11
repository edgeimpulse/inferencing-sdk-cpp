/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdio.h>
#include <stdarg.h>
#include "tensorflow/lite/micro/debug_log.h"

// Use standard IO on Posix systems for debugging
extern "C" void DebugLog(const char* s) {
  printf("%s", s);
}

void ei_printf(const char *format, ...) {
  va_list myargs;
  va_start(myargs, format);
  vprintf(format, myargs);
  va_end(myargs);
}
