#pragma once

namespace vxlab::meta {

/**
 * @brief A compile-time string.
 *
 * This is used to store strings at compile-time in a constexpr expression or as
 * a template parameter.
 *
 * @tparam N The size of the string
 *
 * @note The size of the string is N - 1 because the last character is always
 * null-terminated '\0'.
 *
 * @example
 * constexpr auto hello = const_string<6>("Hello");
 *
 * @example
 * template <const_string S>
 * struct foo {
 *   static constexpr auto value = S;
 * }
 */
template <usize N = 1>
struct const_string {
  char data[N]{};

  /**
   * @brief Construct a const_string object from a string literal.
   *
   * @tparam N The size of the string literal
   *
   * @param str The string literal
   */
  constexpr const_string(const char (&str)[N]) { std::copy_n(str, N, data); }

  /**
   * @brief Default constructor.
   *
   * @note Initialize the data member with spaces.
   */
  constexpr const_string() {
    std::fill(data, data + N - 1, ' ');
    data[N - 1] = '\0';
  }

  /**
   * @brief Compare two const_string objects.
   *
   * @param str The other const_string object
   *
   * @return true if the two const_string objects are equal, false otherwise
   *
   * @note Only check equality if the sizes of the two const_string objects are
   * the same.
   */
  constexpr bool operator==(const const_string<N> str) const {
    return std::equal(str.data, str.data + N, data);
  }

  /**
   * @brief Compare two const_string objects.
   *
   * @tparam M The size of the other const_string object
   *
   * @param s The other const_string object
   *
   * @return false
   *
   * @note Two const_string objects with different sizes are never equal.
   */
  template <usize M>
  constexpr bool operator==(const const_string<M> s) const {
    return false;
  }

  /**
   * @brief Concatenate two const_string objects.
   *
   * @tparam M The size of the other const_string object
   *
   * @param str The other const_string object
   *
   * @return A new const_string object with the concatenated data
   */
  template <usize M>
  constexpr const_string<N + M - 1> operator+(const const_string<M> str) const {
    char new_data[N + M - 1]{};

    // copy the data from the current object and the other object to the new
    std::copy_n(data, N - 1, new_data);
    std::copy_n(str.data, M, new_data + N - 1);

    return new_data;
  }

  /**
   * @brief Get the character at the specified index.
   *
   * @param n The index of the character
   *
   * @return The character at the specified index
   */
  constexpr char operator[](usize n) const { return data[n]; }

  /**
   * @brief Get the size of the string.
   *
   * @return The size of the string
   */
  constexpr usize size() const { return N - 1; }

  /**
   * @brief Get the null-terminated string.
   *
   * @return The null-terminated string
   */
  constexpr const char* c_str() const { return data; }

  /**
   * @brief Implicit conversion to a null-terminated string.
   *
   * @return The null-terminated string
   */
  constexpr operator const char*() const { return data; }
};

/**
 * @brief Concatenate a const_string object with a string literal.
 *
 * @tparam N The size of the const_string object
 *
 * @param str1 The const_string object
 * @param str2 The string literal
 *
 * @return A new const_string object with the concatenated data
 */
template <usize N>
constexpr auto operator==(const_string<N> str1, const char* str2) {
  return std::strcmp(str1.c_str(), str2) == 0;
}

/**
 * @brief Concatenate a string literal with a const_string object.
 *
 * @tparam N The size of the string literal
 *
 * @param str1 The string literal
 * @param str2 The const_string object
 *
 * @return A new const_string object with the concatenated data
 */
template <usize N>
constexpr auto operator==(const char* str1, const_string<N> str2) {
  return std::strcmp(str1, str2.c_str()) == 0;
}

/**
 * @brief Concatenate a const_string object with a string literal.
 *
 * @tparam N The size of the const_string object
 * @tparam M The size of the string literal
 *
 * @param str1 The const_string object
 * @param str2 The string literal
 *
 * @return A new const_string object with the concatenated data
 */
template <usize N, usize M>
constexpr auto operator+(const_string<N> str1, char (&str2)[M]) {
  return str1 + const_string<M>{str2};
}

/**
 * @brief Concatenate a string literal with a const_string object.
 *
 * @tparam N The size of the string literal
 * @tparam M The size of the const_string object
 *
 * @param str1 The string literal
 * @param str2 The const_string object
 *
 * @return A new const_string object with the concatenated data
 */
template <usize N, usize M>
constexpr auto operator+(char (&str1)[N], const_string<M> str2) {
  return const_string<N>{str1} + str2;
}

/**
 * @brief Compare a const_string object with a string literal.
 *
 * @tparam N The size of the const_string object
 * @tparam M The size of the string literal
 *
 * @param str1 The const_string object
 * @param str2 The string literal
 *
 * @return true if the two objects are equal, false otherwise
 */
template <usize N, usize M>
constexpr auto operator==(const_string<N> str1, char (&str2)[M]) {
  return str1 == const_string<M>{str2};
}

/**
 * @brief Compare a string literal with a const_string object.
 *
 * @tparam N The size of the string literal
 * @tparam M The size of the const_string object
 *
 * @param str1 The string literal
 * @param str2 The const_string object
 *
 * @return true if the two objects are equal, false otherwise
 */
template <usize N, usize M>
constexpr auto operator==(char (&str1)[N], const_string<M> str2) {
  return const_string<N>{str1} == str2;
}

}  // namespace vxlab::meta