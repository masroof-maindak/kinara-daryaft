cmake_minimum_required(VERSION 3.14)
project(KinaraDaryaft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(MOLD_BIN mold)
find_program(LLD_BIN lld)
if(MOLD_BIN)
    set(CMAKE_LINKER_TYPE MOLD)
elseif(LLD_BIN)
    set(CMAKE_LINKER_TYPE LLD)
endif()

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

find_package(OpenCV REQUIRED)
add_library(
    KinaraDaryaft
    "src/io.cpp"
    "src/gauss.cpp"
    "src/utils.cpp"
    "src/gradient.cpp"
    "src/nms.cpp"
    "src/hysteresis.cpp"
    "src/canny.cpp"
)

target_include_directories(
    KinaraDaryaft
    PUBLIC "include/" ${OpenCV_INCLUDE_DIRS}
)

add_executable(knr "src/main.cpp" "src/args.cpp" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(KinaraDaryaft PUBLIC ${OpenCV_LIBS})
target_link_libraries(knr PRIVATE KinaraDaryaft argparse ${OpenCV_LIBS})
