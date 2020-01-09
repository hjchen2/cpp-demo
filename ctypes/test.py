import ctypes

_LIB = ctypes.CDLL("./test.so", ctypes.RTLD_GLOBAL)

a = ctypes.c_int(1)
b = ctypes.c_int(2)

# Call C func in Python
print(_LIB.add(a, b))

cfunc = ctypes.CFUNCTYPE(
    ctypes.c_int, # return type
    ctypes.c_int, # arg0 type
    ctypes.c_int  # arg1 type
    )

def add(a, b):
  return a+b

f = cfunc(add)

# ctypes CFUNCTYPE is callable in Python
print(f(5, 1))

# Call Python func in C
print(_LIB.call_py_func(f, 5, 1))

