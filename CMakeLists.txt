cmake_minimum_required(VERSION 3.18.0)
project(mechanizm VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckIncludeFiles)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( mvec src/main.cpp )
target_link_libraries( mvec ${OpenCV_LIBS} )
