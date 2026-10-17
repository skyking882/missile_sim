#pragma once
#include "Module.h"
class PyMath : protected Module {
public:
  PyMath() : Module() {}
  void include(py::module_ &m);
};

extern PyMath py_math;
