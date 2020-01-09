#include "test.h"

int add(int a, int b) {
  return a + b;
}

int call_py_func(PyCFunc f, int a, int b) {
  return f(a, b);
}
