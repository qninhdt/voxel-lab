#pragma once

#include <fmt/core.h>
#include <fmt/format.h>

#include <functional>
#include <magic_enum/magic_enum.hpp>
#include <rfl.hpp>
#include <string>
#include <type_traits>
#include <vector>

namespace vxlab {

// define some common types

using usize = typeof(sizeof(0));

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

using char8 = char;
using str = std::string;

template <typename T>
using vector = std::vector<T>;

static const std::nullptr_t null = nullptr;
}  // namespace vxlab