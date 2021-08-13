cmake_minimum_required(VERSION 3.12)
project(admixgwas)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_SOURCE_DIR}/thirdparty/)
set(SOURCE_FILES src/FileUtils.cpp src/FileUtils.h src/Utils.cpp src/Utils.h)

add_subdirectory(thirdparty/pybind11)
pybind11_add_module(admixgwas src/main.cpp ${SOURCE_FILES})

#add_executable(admixgwas src/main.cpp ${SOURCE_FILES} )
#install(TARGETS admixgwas)