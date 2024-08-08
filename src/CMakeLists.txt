set(VXLAB_SRC
  ./vxlab/core/state.cpp
)

add_library(vxlab SHARED ${VXLAB_SRC})

# find_package(benchmark CONFIG REQUIRED)
# target_link_libraries(vxlab PRIVATE benchmark::benchmark benchmark::benchmark_main)

find_package(magic_enum CONFIG REQUIRED)
target_link_libraries(vxlab PRIVATE magic_enum::magic_enum)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(vxlab PUBLIC fmt::fmt)

# add the include directories
target_include_directories(vxlab PUBLIC ../include)

# add the precompiled headers
target_precompile_headers(vxlab PUBLIC ../include/vxlab/common.hpp)