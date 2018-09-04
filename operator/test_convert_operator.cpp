#include <stdio.h>
#include <iostream>

class Tony {
 public:
  Tony() {}
  Tony& operator=(int val) {
    val_ = val;
    return *this;
  }
  operator int() {
    return val_;
  }

  operator std::string() {
      return "test string";
  }

 private:
  int val_ = 0;  //NOLINT
};

int main() {
  Tony t;
  std::cout << int(t) << std::endl;
  t = 10;
  std::cout << int(t) << std::endl;
  std::cout << std::string(t) << std::endl;
  return 0;
}
