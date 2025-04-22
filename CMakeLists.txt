cmake_minimum_required(VERSION 3.27)
project(voxel_lab)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# enable AVX512
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-march=native)
endif()

add_executable(vxlab_main ./src/main.cpp)

add_library(vxlab ./src/vxlab/core/state.cpp)

# add the include directories
target_include_directories(vxlab PUBLIC include)

# add the precompiled headers
target_precompile_headers(vxlab PUBLIC include/vxlab/common.hpp)

find_package(magic_enum CONFIG REQUIRED)
target_link_libraries(vxlab PUBLIC magic_enum::magic_enum)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(vxlab PUBLIC fmt::fmt)

find_package(benchmark CONFIG REQUIRED)
target_link_libraries(vxlab PUBLIC benchmark::benchmark benchmark::benchmark_main)

find_package(robin_hood CONFIG REQUIRED)
target_link_libraries(vxlab PUBLIC robin_hood::robin_hood)

find_package(reflectcpp CONFIG REQUIRED)
target_link_libraries(vxlab PRIVATE reflectcpp::reflectcpp)

find_package(tsl-robin-map CONFIG REQUIRED)
target_link_libraries(vxlab PRIVATE tsl::robin_map)

find_package(glm CONFIG REQUIRED)
target_link_libraries(vxlab PRIVATE glm::glm)

find_package(robin_hood CONFIG REQUIRED)
target_link_libraries(vxlab PRIVATE robin_hood::robin_hood)

target_link_libraries(vxlab_main PRIVATE vxlab)

# add the tests
enable_testing()
add_subdirectory(tests)


