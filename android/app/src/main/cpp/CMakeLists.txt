cmake_minimum_required(VERSION 3.4.1)

project(llama_cpp_flutter)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# Add compile definitions for CPU-only build
add_definitions(-DGGML_USE_CALL_LOCAL)
add_definitions(-DGGML_CPU_ONLY)

# Include directories
include_directories(
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

# Compiler flags for optimization and compatibility
target_compile_options(llama_cpp_flutter PRIVATE
    -O3
    -ffast-math
    -fno-finite-math-only
    -funroll-loops
)
