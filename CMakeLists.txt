cmake_minimum_required(VERSION 3.10)
project(opencv-cmake-template LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Linker
find_program(MOLD_BIN mold)
find_program(LLD_BIN lld)
if(MOLD_BIN)
    set(CMAKE_LINKER_TYPE MOLD)
elseif(LLD_BIN)
    set(CMAKE_LINKER_TYPE LLD)
endif()

find_package(OpenCV REQUIRED)
add_executable(demo "src/main.cpp")
target_include_directories(demo PRIVATE "include/" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(demo PRIVATE ${OpenCV_LIBS})
