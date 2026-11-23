/**
 * @file moonshine_parallel.h
 * @brief Lightweight dual-core work dispatcher for ESP32-P4
 */

#ifndef MOONSHINE_PARALLEL_H
#define MOONSHINE_PARALLEL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MOONSHINE_PAR_PARTS 2

typedef void (*moonshine_par_fn)(void *ctx, int part, int parts);

/**
 * @brief Initialize parallel execution worker task pinned to Core 1.
 * Safe to call multiple times (idempotent).
 */
void moonshine_parallel_init(void);

/**
 * @brief Run fn on 2 cores.
 * Part 0 runs on caller (Core 0), part 1 runs on worker (Core 1).
 * Blocks until both parts finish.
 * 
 * @param fn Function to execute: fn(ctx, part, parts)
 * @param ctx Context pointer passed to fn
 */
void moonshine_parallel_run(moonshine_par_fn fn, void *ctx);

/**
 * @brief Helper to split range [0, n) into parts with given alignment.
 */
static inline void moonshine_par_range(int n, int part, int parts, int align, int *lo, int *hi) {
    int units = (n + align - 1) / align;
    int a = units * part / parts;
    int b = units * (part + 1) / parts;
    *lo = (a * align < n) ? (a * align) : n;
    *hi = (b * align < n) ? (b * align) : n;
}

#ifdef __cplusplus
}
#endif

#endif // MOONSHINE_PARALLEL_H
