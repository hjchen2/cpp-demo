#include <type_traits>

template <
    typename T,
    typename std::enable_if<std::is_same<T, int>::value>::type * = nullptr>
T test1(T value) {
  return value;
}

template <typename T>
typename std::enable_if<std::is_same<T, int>::value, bool>::type
is_equal(T lhs, T rhs) {
  return lhs == rhs;
}

int main() {
  auto val = test1(6);
  // val = test(6.1);  // float is not support

  bool status = is_equal(6, 6);
  // status = is_equal(6.1, 6.1);  // float is not support
  return 0;
}
