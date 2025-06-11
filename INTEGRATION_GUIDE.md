# llama.cpp Integration Guide

This guide explains how to integrate the real llama.cpp library into your GPT Lite application.

## Quick Start

### Windows Users
Run the setup script:
```cmd
setup_llama.bat
```

### Linux/Mac Users
Run the setup script:
```bash
chmod +x setup_llama.sh
./setup_llama.sh
```

## Manual Integration Steps

### 1. Clone llama.cpp Repository

```bash
git clone https://github.com/ggerganov/llama.cpp.git
```

### 2. Copy Required Files

Copy these files from `llama.cpp/` to `android/app/src/main/cpp/llama_cpp/`:

- `llama.h`
- `llama.cpp`
- `ggml.h`
- `ggml.c`
- `ggml-impl.h`
- `ggml-alloc.h`
- `ggml-alloc.c` 
- `ggml-backend.h`
- `ggml-backend.c`

### 3. Update CMakeLists.txt

Replace `android/app/src/main/cpp/CMakeLists.txt` with:

```cmake
cmake_minimum_required(VERSION 3.4.1)

project(llama_cpp_flutter)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# Add compile definitions
add_definitions(-DGGML_USE_CALL_LOCAL)

# Add source files
set(LLAMA_SOURCES
    llama_cpp/llama.cpp
    llama_cpp/ggml.c
    llama_cpp/ggml-alloc.c
    llama_cpp/ggml-backend.c
    llama_bridge.cpp
)

# Create the library
add_library(llama_cpp_flutter SHARED ${LLAMA_SOURCES})

# Link libraries
target_link_libraries(llama_cpp_flutter
    android
    log
)

# Include directories
target_include_directories(llama_cpp_flutter PRIVATE
    llama_cpp/
)

# Compiler flags
target_compile_options(llama_cpp_flutter PRIVATE
    -O3
    -DNDEBUG
    -ffast-math
)
```

### 4. Update llama_bridge.cpp

Replace the stub function calls in `llama_bridge.cpp` with real llama.cpp function calls:

```cpp
#include "llama_cpp/llama.h" // Update include path
// Remove stub implementations and use real llama.cpp functions
```

### 5. Update Header Include

Update `android/app/src/main/cpp/llama.h` to include the real llama.cpp header:

```cpp
#include "llama_cpp/llama.h"
```

### 6. Remove Stub Implementation

The setup script automatically backs up `llama_stub.cpp`. You can delete it after confirming the real implementation works.

## Building and Testing

### 1. Build the Application

```bash
flutter clean
flutter pub get
flutter build apk --debug
```

### 2. Download Model Files

Download GGUF model files from:
- [Hugging Face Models](https://huggingface.co/models?filter=gguf)
- [TheBloke's Models](https://huggingface.co/TheBloke)

### Recommended Models for Mobile:

**Small Models (Good for testing):**
- `phi-2.Q4_K_M.gguf` (~1.6GB)
- `TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf` (~0.6GB)

**Medium Models (Better quality):**
- `llama-2-7b-chat.Q4_K_M.gguf` (~4GB)
- `mistral-7b-instruct-v0.1.Q4_K_M.gguf` (~4GB)

### 3. Test the Application

1. Install the APK on your Android device
2. Copy a GGUF model file to your device storage
3. Open the app and load the model file
4. Start chatting!

## Performance Optimization

### Memory Management
- Use quantized models (Q4_K_M, Q5_K_M) for better performance
- Ensure your device has sufficient RAM (8GB+ for 7B models)
- Close other apps when using large models

### CPU Optimization
- The current setup uses CPU-only inference
- For GPU acceleration, additional Metal/OpenCL integration would be needed

### Model Size vs Quality Trade-offs
- **Q2_K**: Smallest size, lowest quality
- **Q4_K_M**: Good balance of size and quality (recommended)
- **Q5_K_M**: Better quality, larger size
- **Q8_0**: Highest quality, largest size

## Troubleshooting

### Build Issues
1. **NDK not found**: Install Android NDK via Android Studio
2. **CMake errors**: Ensure CMake 3.4.1+ is installed
3. **Compilation errors**: Check that all source files are copied correctly

### Runtime Issues
1. **Model loading fails**: Ensure GGUF file is valid and accessible
2. **Out of memory**: Try a smaller quantized model
3. **Slow inference**: Normal for CPU-only inference on mobile devices

### Common Errors
- **Symbol not found**: Check that all required source files are included
- **Linker errors**: Verify CMakeLists.txt configuration
- **JNI errors**: Ensure native method signatures match Kotlin declarations

## Advanced Configuration

### Custom Model Parameters
Modify the context parameters in `llama_bridge.cpp`:

```cpp
auto params = llama_context_default_params();
params.n_ctx = 2048;      // Context length
params.n_batch = 512;     // Batch size
params.temp = 0.8f;       // Temperature
params.top_p = 0.9f;      // Top-p sampling
params.top_k = 40;        // Top-k sampling
```

### Memory Usage Optimization
Adjust memory mapping settings:

```cpp
auto model_params = llama_model_default_params();
model_params.use_mmap = true;   // Memory mapping
model_params.use_mlock = false; // Memory locking
```

## Support and Contributions

For issues related to:
- **llama.cpp**: Visit [llama.cpp repository](https://github.com/ggerganov/llama.cpp)
- **GPT Lite integration**: Create an issue in this repository
- **Flutter development**: Check [Flutter documentation](https://flutter.dev/docs)

## License

This integration guide is part of the GPT Lite project. The llama.cpp library has its own license terms.
