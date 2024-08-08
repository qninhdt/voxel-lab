cmake_minimum_required(VERSION 3.27)
project(voxel_lab)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    ./src/main.cpp
)

add_executable(vxlab_main ./src/main.cpp)

add_subdirectory(src)
target_link_libraries(vxlab_main PRIVATE vxlab)

# add the tests
enable_testing()
add_subdirectory(tests)


