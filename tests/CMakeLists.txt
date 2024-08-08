add_library(test_lib INTERFACE test_utils.hpp)

find_package(Catch2 CONFIG REQUIRED)

target_link_libraries(test_lib INTERFACE Catch2::Catch2 Catch2::Catch2WithMain)
target_link_libraries(test_lib INTERFACE vxlab)

# add tests
set(TESTS
  test_const_string.cpp
)

foreach(TEST ${TESTS})
  get_filename_component(TEST_NAME ${TEST} NAME_WE)
  add_executable(${TEST_NAME} ${TEST})
  target_link_libraries(${TEST_NAME} PRIVATE test_lib)
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
