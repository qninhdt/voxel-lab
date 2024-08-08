#include <iostream>
#include <sstream>
#include <vxlab/core/meta/const_string.hpp>
#include <vxlab/core/meta/state_machine.hpp>

using namespace vxlab;
using namespace vxlab::meta;

enum class Color {
  WHITE = 0,
  ORANGE = 1,
  MAGENTA = 2,
  LIGHT_BLUE = 3,
  YELLOW = 4,
  LIME = 5,
  PINK = 6,
  GRAY = 7,
  LIGHT_GRAY = 8,
  CYAN = 9,
  PURPLE = 10,
  BLUE = 11,
  BROWN = 12,
  GREEN = 13,
  RED = 14,
  BLACK = 15
};

enum class Direction {
  NORTH = 0,
  EAST = 1,
  SOUTH = 2,
  WEST = 3,
  UP = 4,
  DOWN = 5
};

class Block {
 public:
  Block() {}

  bool isSolid() const { return true; }

  virtual ~Block() {}
};

using BlockStateMachine = state_machine<Block>;

using direction_attribute_compound = BlockStateMachine::attribute_compound<
    BlockStateMachine::enum_attribute<"direction", Direction>>;
using color_attribute = BlockStateMachine::enum_attribute<"color", Color>;
using is_wet_attribute = BlockStateMachine::bool_attribute<"is_wet">;

class WoolBlock
    : public BlockStateMachine::attribute_compound<
          direction_attribute_compound, color_attribute, is_wet_attribute> {
 public:
  WoolBlock() {}
};

class GlassBlock
    : public BlockStateMachine::attribute_compound<color_attribute> {
 public:
  GlassBlock() {}
};

template <typename T>
  requires std::is_enum_v<T>
class fmt::formatter<T> {
 public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template <typename Context>
  constexpr auto format(T const& foo, Context& ctx) const {
    return fmt::format_to(ctx.out(), "{}", magic_enum::enum_name(foo));
  }
};

template <typename T>
void print(T const& t) {
  const WoolBlock& wool = *dynamic_cast<const WoolBlock*>(&t);
  fmt::println("{} {} {}", wool.get<"is_wet">(), wool.get<"color">(),
               wool.get<"direction">());
}

int main() {
  BlockStateMachine state_machine;

  const auto WOOL_BLOCK = state_machine.register_state_group<WoolBlock>();
  const auto GLASS_BLOCK = state_machine.register_state_group<GlassBlock>();

  auto& base_wool = state_machine.get_state<WoolBlock>(WOOL_BLOCK, 0);

  // normal type
  print(base_wool.set<"color">(Color::MAGENTA)
            .set<"direction">(Direction::WEST)
            .set<"is_wet">(true));

  // is_wet_attribute
  const is_wet_attribute& is_wet = base_wool;
  print(is_wet.set(true));

  // color_attribute
  const color_attribute& color = base_wool;
  print(color.set(Color::LIGHT_BLUE));

  // direction_attribute_compound
  const direction_attribute_compound& direction = base_wool;
  print(direction.set<"direction">(Direction::UP));

  // cast pipeline: wool -> null -> color (RED) -> null -> direction (DOWN) ->
  // null -> is_wet(true) -> null -> wool
  const Block* block1 = &base_wool;

  fmt::print("wool: {}\n", (void*)&base_wool);

  fmt::print("block1: {}\n", (void*)block1);

  const auto& color_ptr = dynamic_cast<const color_attribute*>(block1);
  const Block* block2 = &color_ptr->set(Color::RED);

  fmt::print("block2: {}\n", (void*)block2);

  auto direction_ptr =
      dynamic_cast<const direction_attribute_compound*>(block2);
  const Block* block3 = &direction_ptr->set<"direction">(Direction::DOWN);

  fmt::print("block3: {}\n", (void*)block3);

  auto is_wet_ptr = dynamic_cast<const is_wet_attribute*>(block3);
  const Block* block4 = &is_wet_ptr->set(true);

  fmt::print("block4: {}\n", (void*)block4);

  const WoolBlock& wool = *reinterpret_cast<const WoolBlock*>(block4);
  fmt::print("wool: {}\n", (void*)&wool);
  print(wool);

  return 0;
}