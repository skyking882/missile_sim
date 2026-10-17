#pragma once
#include "Module.h"

class PyUnit : protected Module {
public:
  PyUnit() : Module() {}
  void include(py::module_ &m);
};

extern PyUnit py_unit;
