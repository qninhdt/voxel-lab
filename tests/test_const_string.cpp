#include <catch2/catch_test_macros.hpp>
#include <vxlab/core/meta/const_string.hpp>

#include "test_utils.hpp"

using namespace vxlab::meta;

TEST_CASE("const_string default constructor") {
  const_string<6> str;
  REQUIRE(str.size() == 5);
  REQUIRE(str.c_str() == std::string("     "));  // 5 spaces
}

TEST_CASE("const_string literal constructor") {
  const_string<6> str("Hello");
  REQUIRE(str.size() == 5);
  REQUIRE(str.c_str() == std::string("Hello"));
}

TEST_CASE("const_string equality operator same size") {
  const_string<6> str1("Hello");
  const_string<6> str2("Hello");
  const_string<6> str3("World");
  REQUIRE(str1 == str2);
  REQUIRE_FALSE(str1 == str3);
}

TEST_CASE("const_string equality operator different size") {
  const_string<6> str1("Hello");
  const_string<7> str2("Hello!");
  REQUIRE_FALSE(str1 == str2);
}

TEST_CASE("const_string concatenation") {
  const_string<6> str1("Hello");
  const_string<2> str2(" ");
  const_string<6> str3("World");
  auto concatenated = str1 + str2 + str3;
  REQUIRE(concatenated.size() == 11);
  REQUIRE(concatenated == "Hello World");
}

TEST_CASE("const_string indexing") {
  const_string<6> str("Hello");
  REQUIRE(str[0] == 'H');
  REQUIRE(str[4] == 'o');
}

TEST_CASE("const_string size and c_str") {
  const_string<6> str("Hello");
  REQUIRE(str.size() == 5);
  REQUIRE(std::strcmp(str.c_str(), "Hello") == 0);
}

TEST_CASE("const_string with c-string equality") {
  const_string<6> str1("Hello");
  REQUIRE(str1 == "Hello");
  REQUIRE("Hello" == str1);
  REQUIRE_FALSE(str1 == "World");
  REQUIRE_FALSE("World" == str1);
}

TEST_CASE("const_string concatenation with c-string") {
  const_string str1("Hello");
  char str2[] = " World";
  auto concatenated = str1 + str2;
  REQUIRE(concatenated.size() == 11);
  REQUIRE(concatenated == "Hello World");

  auto concatenated2 = str2 + str1;
  REQUIRE(concatenated2.size() == 11);
  REQUIRE(concatenated2 == " WorldHello");
}