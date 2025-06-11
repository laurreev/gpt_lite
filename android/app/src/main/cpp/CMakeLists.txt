cmake_minimum_required(VERSION 3.4.1)

project(llama_cpp_flutter)

# Create a simple library with stub implementations for now
add_library(llama_cpp_flutter SHARED
    llama_bridge_stub.cpp
)

target_link_libraries(llama_cpp_flutter
    android
    log
)
