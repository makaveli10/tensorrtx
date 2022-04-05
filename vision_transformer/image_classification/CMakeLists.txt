cmake_minimum_required(VERSION 3.10)

# set the project name
project(vit)

add_definitions(-std=c++11)

# get main project dir to include common files
get_filename_component(MAIN_DIR ../ ABSOLUTE)

# When enabled the static version of the 
# CUDA runtime library will be used in CUDA_LIBRARIES
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

# include
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
# include and link cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11; -g; -G;-gencode; arch=compute_75;code=sm_75)
enable_language(CUDA)

# include and link tensorrt
include_directories(/usr/include/x86_64-linux-gnu)
link_directories(/usr/lib/x86_64-linux-gnu)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

# add the executable
file(GLOB SOURCE_FILES "*.h" "*.cpp" "*.cu")
add_executable(vit ${SOURCE_FILES})

target_link_libraries(vit nvinfer)
target_link_libraries(vit cudart)
target_link_libraries(swintransformer ${OpenCV_LIBS})

add_definitions(-O2 -pthread)