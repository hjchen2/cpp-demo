#include <iostream>

template<typename T>
T add(T&& val) {
    return val;
}

template<typename T, typename... Args>
T add(T&& val, Args&&... args) {
  T sum = val;
  // sum += add(std::forward<Args>(args)...);
  sum += add(args...);
  return sum;
}

int main() {
  auto a = add(1.0, 2.0);
  std::cout << a << std::endl;

  a = add(1.0, 2.0, 5.0);
  std::cout << a << std::endl;

  a = add(1, 2);
  std::cout << a << std::endl;

  a = add(1, 2, 5);
  std::cout << a << std::endl;

  a = add(1.0, 2);
  std::cout << a << std::endl;

  a = add(1, 2.0);
  std::cout << a << std::endl;
  
  a = add(1.0, 2, 5.0);
  std::cout << a << std::endl;

  return 0;
}
