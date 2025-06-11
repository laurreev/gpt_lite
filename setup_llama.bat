@echo off
echo GPT Lite - llama.cpp Integration Setup
echo ======================================

REM Check if llama.cpp directory exists
if not exist "llama.cpp" (
    echo Step 1: Cloning llama.cpp repository...
    git clone https://github.com/ggerganov/llama.cpp.git
    if errorlevel 1 (
        echo Failed to clone llama.cpp repository
        exit /b 1
    )
) else (
    echo Step 1: llama.cpp directory already exists
)

REM Create the integration directory
echo Step 2: Creating integration directory...
mkdir "android\app\src\main\cpp\llama_cpp" 2>nul

REM Copy necessary files
echo Step 3: Copying llama.cpp source files...
copy "llama.cpp\llama.h" "android\app\src\main\cpp\llama_cpp\"
copy "llama.cpp\llama.cpp" "android\app\src\main\cpp\llama_cpp\"
copy "llama.cpp\ggml.h" "android\app\src\main\cpp\llama_cpp\"
copy "llama.cpp\ggml.c" "android\app\src\main\cpp\llama_cpp\"
copy "llama.cpp\ggml-impl.h" "android\app\src\main\cpp\llama_cpp\"
copy "llama.cpp\ggml-alloc.h" "android\app\src\main\cpp\llama_cpp\"
copy "llama.cpp\ggml-alloc.c" "android\app\src\main\cpp\llama_cpp\"
copy "llama.cpp\ggml-backend.h" "android\app\src\main\cpp\llama_cpp\"
copy "llama.cpp\ggml-backend.c" "android\app\src\main\cpp\llama_cpp\"

REM Update CMakeLists.txt
echo Step 4: Updating CMakeLists.txt...
(
echo cmake_minimum_required^(VERSION 3.4.1^)
echo.
echo project^(llama_cpp_flutter^)
echo.
echo # Set C++ standard
echo set^(CMAKE_CXX_STANDARD 17^)
echo set^(CMAKE_C_STANDARD 11^)
echo.
echo # Add compile definitions
echo add_definitions^(-DGGML_USE_CALL_LOCAL^)
echo.
echo # Add source files
echo set^(LLAMA_SOURCES
echo     llama_cpp/llama.cpp
echo     llama_cpp/ggml.c
echo     llama_cpp/ggml-alloc.c
echo     llama_cpp/ggml-backend.c
echo     llama_bridge.cpp
echo ^)
echo.
echo # Create the library
echo add_library^(llama_cpp_flutter SHARED ${LLAMA_SOURCES}^)
echo.
echo # Link libraries
echo target_link_libraries^(llama_cpp_flutter
echo     android
echo     log
echo ^)
echo.
echo # Include directories
echo target_include_directories^(llama_cpp_flutter PRIVATE
echo     llama_cpp/
echo ^)
echo.
echo # Compiler flags
echo target_compile_options^(llama_cpp_flutter PRIVATE
echo     -O3
echo     -DNDEBUG
echo     -ffast-math
echo ^)
) > "android\app\src\main\cpp\CMakeLists.txt"

echo Step 5: Backup stub implementation...
if exist "android\app\src\main\cpp\llama_stub.cpp" (
    move "android\app\src\main\cpp\llama_stub.cpp" "android\app\src\main\cpp\llama_stub.cpp.bak"
)

echo.
echo Integration setup complete!
echo.
echo Next steps:
echo 1. Update llama_bridge.cpp to use real llama.cpp functions instead of stubs
echo 2. Test the build: flutter build apk --debug
echo 3. Download a GGUF model file for testing
echo.
echo Recommended models for testing:
echo - phi-2.Q4_K_M.gguf ^(smaller, good for testing^)
echo - llama-2-7b-chat.Q4_K_M.gguf ^(larger, better quality^)
echo.
echo Download from: https://huggingface.co/models?filter=gguf
pause
