# ESP32-P4 Automatic Speech Recognition (ASR) Project
cmake_minimum_required(VERSION 3.16)

# Use application components locally. The Moonshine component is resolved
# through its local override in main/idf_component.yml.
set(EXTRA_COMPONENT_DIRS
	"${CMAKE_CURRENT_LIST_DIR}/components"
)

# Include ESP-IDF CMake project
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# Define the project
project(esp32_p4_asr)

# Set C++ standard (required for TFLite Micro)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -ffast-math")
