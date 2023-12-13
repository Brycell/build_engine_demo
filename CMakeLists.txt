project(build_engine_demo)
cmake_minimum_required(VERSION 3.7)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(common)
include(utils)

include_directories(
    ./inc/
    ./onnx-tensorrt/
    /opt/thrid_party/include/
    /opt/cuda-11.4/targets/aarch64-linux/include
    ${OPENCV_INCLUDE_DIRS}
    ${THIRDPARTY_INCLUDE_DIRS}
)

set(SRC_LIST
    main.cpp
    src/argparse.cpp
    src/net.cpp
)

link_directories(
    /opt/cuda-11.4/targets/aarch64-linux/lib/
    ${OPENCV_LIBRARY_DIRS}

)

add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME}
    nvinfer 
    nvonnxparser
    cudart
    ${OPENCV_LIBS}
    /opt/cuda-11.4/targets/aarch64-linux/lib/libcublas.so.11
    /opt/cuda-11.4/targets/aarch64-linux/lib/libcublasLt.so.11
)

