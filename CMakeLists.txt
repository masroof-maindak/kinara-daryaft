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
add_executable(knr "src/main.cpp" "src/args.cpp" "src/io.cpp")
target_include_directories(knr PRIVATE "include/" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(knr PRIVATE argparse ${OpenCV_LIBS})
