extern "C" {

int add(int a, int b);

typedef int (*PyCFunc)(int, int);

int call_py_func(PyCFunc f, int a, int b);

}
