#ifndef GGML_ARM_NEON_COMPAT_H
#define GGML_ARM_NEON_COMPAT_H

// ARM NEON compatibility header for Android build
// Prevents redefinition conflicts and ensures proper ARM NEON support

#ifdef __ARM_NEON
#include <arm_neon.h>

// Check if vcvtnq_s32_f32 is available in the system
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

// Only define our implementation if the builtin is not available
#if !__has_builtin(__builtin_vcvtnq_s32_f32) && !defined(vcvtnq_s32_f32)
// Custom implementation will be provided by GGML
#define GGML_NEED_VCVTNQ_S32_F32_IMPL
#endif

#endif // __ARM_NEON

#endif // GGML_ARM_NEON_COMPAT_H
