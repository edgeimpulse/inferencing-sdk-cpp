/*
 * SPDX-FileCopyrightText: Copyright 2021, 2023, 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ETHOSU_LOG_H
#define ETHOSU_LOG_H

/******************************************************************************
 * Includes
 ******************************************************************************/

#include <stdio.h>
#include <string.h>

/******************************************************************************
 * Defines
 ******************************************************************************/

// Log severity levels
#define ETHOSU_LOG_ERR 0
#define ETHOSU_LOG_WARN 1
#define ETHOSU_LOG_INFO 2
#define ETHOSU_LOG_DEBUG 3

// Define default log severity
#ifndef ETHOSU_LOG_SEVERITY
#define ETHOSU_LOG_SEVERITY ETHOSU_LOG_WARN
#endif

// Logs enabled by default
#ifndef ETHOSU_LOG_ENABLE
#define ETHOSU_LOG_ENABLE 1
#endif

#define LOG_COMMON_NOP(s, f, ...)                                                                                      \
    if (0)                                                                                                             \
    (void)fprintf(s, "%s" f, "", ##__VA_ARGS__)

#if ETHOSU_LOG_ENABLE
#define LOG_COMMON(s, f, ...) (void)fprintf(s, f, ##__VA_ARGS__)
#else
#define LOG_COMMON(s, f, ...) LOG_COMMON_NOP(s, f, ##__VA_ARGS__)
#endif

// Log formatting
#define LOG(f, ...) LOG_COMMON(stdout, f, ##__VA_ARGS__)

#if ETHOSU_LOG_SEVERITY >= ETHOSU_LOG_ERR
#ifdef __FILE_NAME__
#define LOG_ERR(f, ...) LOG_COMMON(stderr, "E: %s:%d: " f "\n", __FILE_NAME__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_ERR(f, ...) LOG_COMMON(stderr, "E: %s:%d: " f "\n", strrchr("/" __FILE__, '/') + 1, __LINE__, ##__VA_ARGS__)
#endif
#else
#define LOG_ERR(f, ...) LOG_COMMON_NOP(stderr, f, ##__VA_ARGS__)
#endif

#if ETHOSU_LOG_SEVERITY >= ETHOSU_LOG_WARN
#define LOG_WARN(f, ...) LOG_COMMON(stdout, "W: " f "\n", ##__VA_ARGS__)
#else
#define LOG_WARN(f, ...) LOG_COMMON_NOP(stdout, f, ##__VA_ARGS__)
#endif

#if ETHOSU_LOG_SEVERITY >= ETHOSU_LOG_INFO
#define LOG_INFO(f, ...) LOG_COMMON(stdout, "I: " f "\n", ##__VA_ARGS__)
#else
#define LOG_INFO(f, ...) LOG_COMMON_NOP(stdout, f, ##__VA_ARGS__)
#endif

#if ETHOSU_LOG_SEVERITY >= ETHOSU_LOG_DEBUG
#define LOG_DEBUG(f, ...) LOG_COMMON(stdout, "D: %s(): " f "\n", __FUNCTION__, ##__VA_ARGS__)
#else
#define LOG_DEBUG(f, ...) LOG_COMMON_NOP(stdout, f, ##__VA_ARGS__)
#endif

#endif
