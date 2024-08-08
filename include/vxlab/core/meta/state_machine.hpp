#pragma once

#include <vxlab/core/meta/const_string.hpp>

namespace vxlab::meta {

template <typename T>
struct converter {
  usize to_index(const T& value) const {
    static_assert(false, "Converter not implemented for this type");
    return 0;
  }

  T to_value(usize index) const {
    static_assert(false, "Converter not implemented for this type");
  }
};

template <typename T>
  requires std::is_integral_v<T>
struct converter<T> {
  usize to_index(const T& value) const { return static_cast<usize>(value); }

  T to_value(usize index) const { return static_cast<T>(index); }
};

template <typename T>
  requires std::is_enum_v<T>
struct converter<T> {
  usize to_index(const T& value) const {
    return magic_enum::enum_integer(value);
  }

  T to_value(usize index) const {
    return magic_enum::enum_cast<T>(index).value();
  }
};

template <typename ext_state_base>
class state_machine {
 private:
  class state_group_base;

 public:
  class state_base {
    template <typename T>
    friend class state_machine;

   public:
    usize global_id() const { return m_global_id; }

    usize local_id() const { return m_local_id; }

    usize group_id() const { return m_group_id; }

    virtual ~state_base() {}

   private:
    usize m_global_id;
    usize m_local_id;
    usize m_group_id;
  };

  class attribute_base {
   public:
    attribute_base(usize size) : m_size(size) {}

    inline const usize& get_index() const { return m_index; }

   protected:
    // init the index and weight, this function is called by
    // `attribute_compound` class.
    void initialize(usize index, usize weight,
                    state_group_base* state_group_base_) {
      m_weight = weight;
      m_state_group_base = state_group_base_;
      m_index = (index / m_weight) % m_size;
    }

    // index of the attribute in value range [0, N)
    usize m_index;

    state_group_base* m_state_group_base = nullptr;

    // weight of the attribute, if attr1, attr2, ..., arrtN-1 are the previous
    // attributes of the current one in a attribute compound, then the
    // weight of the current attribute is product of thier sizes: weight =
    // compound.weight * attr1.size * attr2.size * ... * attrN-1.size
    usize m_weight;

    // maximum number of values that the attribute can have
    usize m_size;
  };

  template <typename... S>
  class attribute_compound;

  /**
   * @brief A struct that represents a single attribute.
   *
   * @tparam t_name The name of the attribute
   * @tparam t_type The type of the attribute
   * @tparam t_size The number of attributes
   * @tparam t_to_index A function pointer that converts a attribute to an
   */
  template <const_string t_name, typename t_type, usize t_size>
  class attribute : public attribute_base,
                    public virtual state_base,
                    public virtual ext_state_base {
    template <typename... S>
    friend class attribute_compound;

   public:
    using type = t_type;
    static constexpr auto name = t_name;
    static constexpr auto size = t_size;

    // use as an identifier for the attribute
    using ID = std::integral_constant<const_string<name.size() + 1>, name>;

    attribute() : attribute_base(size) {}

    /**
     * @brief Get the value of the attribute.
     *
     * @return The value of the attribute
     */
    inline const type& get() const { return m_value; }

    template <typename Self>
    inline const auto& set(this const Self& self, const type& value);

    template <typename T>
    inline const auto& to() const {
      using attr_type = std::remove_reference_t<decltype(*this)>;

      if constexpr (std::is_base_of_v<T, attr_type>) {
        // upcasting
        return *static_cast<const T*>(this);
      } else {
        // downcasting
        return *dynamic_cast<const T*>(this);
      }
    }

    virtual ~attribute() {}

   private:
    void initialize(usize index, usize weight, state_group_base* state_group_) {
      attribute_base::initialize(index, weight, state_group_);
      m_value = converter<type>().to_value(this->m_index);
    }

    // only called when ID parameter is matched
    inline const auto& get_impl(ID) const { return *this; }

    type m_value;  // actual value of the attribute
  };
  // some common attributes

  template <const_string name>
  using bool_attribute = attribute<name, bool, 2>;

  template <const_string name, usize size>
  using u8_attribute = attribute<name, u8, size>;

  template <const_string name, usize size>
  using u16_attribute = attribute<name, u16, size>;

  template <const_string name, usize size>
  using u32_attribute = attribute<name, u32, size>;

  template <const_string name, usize size>
  using u64_attribute = attribute<name, u64, size>;

  template <const_string name, usize size>
  using i8_attribute = attribute<name, i8, size>;

  template <const_string name, usize size>
  using i16_attribute = attribute<name, i16, size>;

  template <const_string name, usize size>
  using i32_attribute = attribute<name, i32, size>;

  template <const_string name, usize size>
  using i64_attribute = attribute<name, i64, size>;

  template <const_string name, typename E>
    requires std::is_enum_v<E>
  using enum_attribute = attribute<name, E, magic_enum::enum_count<E>()>;

  class attribute_compound_base {
   public:
    attribute_compound_base(usize size) : m_size(size) {}

    inline const usize& get_size() const { return m_size; }

    inline const usize& get_weight() const { return m_weight; }

   private:
    usize m_size;
    usize m_weight;
  };

  /**
   * @brief A struct that represents a collection of attributes aka state.
   *
   * @tparam A A list of attribute metadata
   */
  template <typename... S>
  class attribute_compound : public virtual S... {
    // bring the `get_value` function from its attribute class and its
    // sub-states, so `get()` can access it via function overloading.
    using S::get_impl...;

    // allow current `attribute_compound` class to access the private
    // members of other `attribute_compound` classes.
    template <typename... SS>
    friend class attribute_compound;

    template <typename T>
    friend class state_machine;

   public:
    static constexpr usize size = sizeof...(S) > 0 ? (S::size*...) : 1;

    // attribute_compound() : attribute_compound_base(size) {}

    /**
     * @brief Get the value of an attribute.
     *
     * @tparam name The name of the attribute
     *
     * @return The value of the attribute
     */
    template <const_string name>
    inline const auto& get() const {
      return get_attr<name>().get();
    }

    /**
     * @brief Get the index of an attribute.
     *
     * @tparam name The name of the attribute
     *
     * @return The index of the attribute
     */
    template <const_string name>
    inline auto get_index() const {
      return get_attr<name>().get_index();
    }

    template <const_string name, typename Self = nullptr_t>
    inline const auto& set(this const Self& self, const auto& value) {
      return self.template get_attr<name>().set(value).template to<Self>();
    }

    virtual ~attribute_compound() {}

   private:
    void initialize(usize index, usize weight,
                    state_group_base* state_group_base_) {
      static_cast<void>((
          [&]() {
            S::initialize(index, weight, state_group_base_);
            weight *= S::size;
          }(),
          ...));
    }

    // fallback function for when the attribute is not found
    template <typename T>
    inline const auto& get_impl(T) const {
      return null;
    }

    // return the attribute object
    template <const_string name>
    inline const auto& get_attr() const {
      using ID = std::integral_constant<const_string<name.size() + 1>, name>;

      // try to get the attribute value from this state
      const auto& attr = get_impl(ID{});

      // return immediately if the attribute is present in this state
      if constexpr (std::is_same_v<decltype(attr), const std::nullptr_t&>) {
        static_assert(false, "Attribute not found");
      }

      return attr;
    }
  };

  template <typename S>
  usize register_state_group() {
    // create a new state group
    state_group<S>* group = new state_group<S>();
    group->offset = m_state_id2state_group_idx.size();
    group->state_type_id = typeid(S).hash_code();

    // map the state id to the index of the state group
    usize group_id = m_state_groups.size();

    m_state_id2state_group_idx.reserve(m_state_id2state_group_idx.size() +
                                       S::size);
    for (usize i = 0; i < S::size; i++) {
      m_state_id2state_group_idx.push_back(group_id);
    }

    for (usize i = 0; i < S::size; i++) {
      auto& state = group->states[i];
      state.initialize(i, 1, group);
      state.m_global_id = group->offset + i;
      state.m_local_id = i;
      state.m_group_id = group_id;
    }

    m_state_groups.push_back(group);

    return group_id;
  }

  template <typename S>
  S& get_state(usize global_id) {
    usize idx = m_state_id2state_group_idx[global_id];
    state_group_base* group = m_state_groups[idx];

    assert(group->state_type_id == typeid(S).hash_code() &&
           "Invalid state type");

    usize local_id = global_id - group->offset;

    assert(local_id < S::size && "Invalid state id");

    return static_cast<state_group<S>*>(m_state_groups[idx])->states[local_id];
  }

  template <typename S>
  S& get_state(usize group_id, usize local_id) {
    state_group_base* group = m_state_groups[group_id];

    assert(group->state_type_id == typeid(S).hash_code() &&
           "Invalid state type");

    assert(local_id < S::size && "Invalid state id");

    return static_cast<state_group<S>*>(m_state_groups[group_id])
        ->states[local_id];
  }

  ~state_machine() {
    for (auto group : m_state_groups) {
      delete group;
    }
  }

 private:
  struct state_group_base {
    usize state_type_id;
    usize offset;  // offset of the state group in the state machine

    state_group_base(usize state_type_id) : state_type_id(state_type_id) {}

    virtual state_base* get_state(usize local_id) = 0;
  };

  template <typename S>
  struct state_group : state_group_base {
    S states[S::size];

    state_group() : state_group_base(typeid(S).hash_code()) {}

    state_base* get_state(usize local_id) override { return &states[local_id]; }
  };

  vector<usize> m_state_id2state_group_idx;
  vector<state_group_base*> m_state_groups;
};

template <typename ext_state_base>
template <const_string t_name, typename t_type, usize t_size>
template <typename Self>
inline const auto&
state_machine<ext_state_base>::attribute<t_name, t_type, t_size>::set(
    this const Self& self, const type& value) {
  usize value_id = converter<type>().to_index(value);

  if (value_id >= size) {
    assert(false && "Invalid value");
  }

  usize new_local_id =
      self.m_local_id + (value_id - self.m_index) * self.m_weight;

  return *dynamic_cast<const Self*>(
      self.m_state_group_base->get_state(new_local_id));
}
}  // namespace vxlab::meta
