cmake_minimum_required(VERSION 3.4.1)

project(llama_cpp_flutter)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# Add compile definitions for CPU-only build
add_definitions(-DGGML_USE_CALL_LOCAL)
add_definitions(-DGGML_CPU_ONLY)
add_definitions(-DGGML_DISABLE_CUSTOM_NEON_FUNCS)

# Performance optimizations - mobile-friendly settings
add_definitions(-DGGML_USE_OPENMP)
add_definitions(-DNDEBUG)

# ARM architecture detection and optimization
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_definitions(-DGGML_USE_ACCELERATE)
    # Conditional NEON support to avoid redefinition conflicts
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        # Check for NEON availability without conflicting with GGML implementation
        include(CheckCSourceCompiles)
        check_c_source_compiles("
            #include <arm_neon.h>
            int main() { return 0; }
        " HAVE_ARM_NEON)
        if(HAVE_ARM_NEON)
            add_definitions(-DGGML_ARM_NEON_AVAILABLE)
        endif()
    endif()
endif()

# Include directories
include_directories(
    .
    llama_cpp/
    ggml/include/
    ggml/src/
    ggml/src/ggml-cpu/
)

# Core GGML source files (CPU only) - Phase 3 with full tensor operations
set(GGML_SOURCES
    ggml/src/ggml.c
    ggml/src/ggml-alloc.c
    ggml/src/ggml-backend.cpp
    ggml/src/ggml-threading.cpp
    ggml/src/ggml-quants.c
    ggml/src/gguf.cpp
)

# Llama.cpp source files - Phase 3 with real tensor data and neural network
set(LLAMA_SOURCES
    llama_bridge_phase3_real.cpp
)

# Create the library
add_library(llama_cpp_flutter SHARED 
    ${GGML_SOURCES}
    ${LLAMA_SOURCES}
)

# Link libraries
target_link_libraries(llama_cpp_flutter
    android
    log
)

# Architecture-specific compiler flags
if(ANDROID_ABI STREQUAL "armeabi-v7a")
    target_compile_options(llama_cpp_flutter PRIVATE
        -O3
        -ffast-math
        -fno-finite-math-only
        -funroll-loops
        -march=armv7-a
        -mfpu=neon
        -fopenmp
        -Wno-implicit-int-float-conversion
        -Wno-conversion
    )
elseif(ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_options(llama_cpp_flutter PRIVATE
        -O3
        -ffast-math
        -fno-finite-math-only
        -funroll-loops
        -march=armv8-a
        -mtune=cortex-a76
        -fopenmp
        -Wno-implicit-int-float-conversion
        -Wno-conversion
    )
else()
    # Fallback for other architectures
    target_compile_options(llama_cpp_flutter PRIVATE
        -O3
        -ffast-math
        -fno-finite-math-only
        -funroll-loops
        -fopenmp
        -Wno-implicit-int-float-conversion
        -Wno-conversion
    )
endif()
