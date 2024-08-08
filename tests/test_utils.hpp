#pragma once

template <typename T, T t>
struct A {};

template <typename T>
constexpr bool is_template_friendly() {
  return requires { A<T, T{}>{}; };
}

template <typename T>
constexpr bool can_be_constructed_without_arguments() {
  return requires { T{}; };
}