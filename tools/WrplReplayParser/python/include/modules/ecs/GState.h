#pragma once
#include "Module.h"

class PyGState : protected Module {
public:
  PyGState() : Module() {}
  void include(py::module_ &m);
};

extern PyGState py_global_state;
