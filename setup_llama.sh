#!/bin/bash

# GPT Lite - llama.cpp Integration Setup Script
# This script helps integrate the real llama.cpp library

echo "GPT Lite - llama.cpp Integration Setup"
echo "======================================"

# Check if llama.cpp directory exists
LLAMA_DIR="./llama.cpp"
if [ ! -d "$LLAMA_DIR" ]; then
    echo "Step 1: Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git
    if [ $? -ne 0 ]; then
        echo "Failed to clone llama.cpp repository"
        exit 1
    fi
else
    echo "Step 1: llama.cpp directory already exists"
fi

# Create the integration directory
INTEGRATION_DIR="./android/app/src/main/cpp/llama_cpp"
echo "Step 2: Creating integration directory..."
mkdir -p "$INTEGRATION_DIR"

# Copy necessary files
echo "Step 3: Copying llama.cpp source files..."
cp "$LLAMA_DIR/llama.h" "$INTEGRATION_DIR/"
cp "$LLAMA_DIR/llama.cpp" "$INTEGRATION_DIR/"
cp "$LLAMA_DIR/ggml.h" "$INTEGRATION_DIR/"
cp "$LLAMA_DIR/ggml.c" "$INTEGRATION_DIR/"
cp "$LLAMA_DIR/ggml-impl.h" "$INTEGRATION_DIR/"
cp "$LLAMA_DIR/ggml-alloc.h" "$INTEGRATION_DIR/"
cp "$LLAMA_DIR/ggml-alloc.c" "$INTEGRATION_DIR/"
cp "$LLAMA_DIR/ggml-backend.h" "$INTEGRATION_DIR/"
cp "$LLAMA_DIR/ggml-backend.c" "$INTEGRATION_DIR/"

# Update CMakeLists.txt
echo "Step 4: Updating CMakeLists.txt..."
cat > ./android/app/src/main/cpp/CMakeLists.txt << 'EOF'
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
EOF

echo "Step 5: Backup stub implementation..."
mv ./android/app/src/main/cpp/llama_stub.cpp ./android/app/src/main/cpp/llama_stub.cpp.bak

echo ""
echo "Integration setup complete!"
echo ""
echo "Next steps:"
echo "1. Update llama_bridge.cpp to use real llama.cpp functions instead of stubs"
echo "2. Test the build: flutter build apk --debug"
echo "3. Download a GGUF model file for testing"
echo ""
echo "Recommended models for testing:"
echo "- phi-2.Q4_K_M.gguf (smaller, good for testing)"
echo "- llama-2-7b-chat.Q4_K_M.gguf (larger, better quality)"
echo ""
echo "Download from: https://huggingface.co/models?filter=gguf"
