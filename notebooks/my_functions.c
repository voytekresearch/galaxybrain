// #define PY_SSIZE_T_CLEAN
// #include <Python.h>

// then call cc -fPIC -shared -o my_functions.so my_functions.c in terminal
// in python:
// from ctypes import *
// so_file = "./my_functions.so"
// my_functions = CDLL(so_file)
#include <stdio.h>


float cov(double a, double b) {
	return a * b;
}