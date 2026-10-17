
#pragma once
#include "Module.h"

class PyMpi : protected Module {
public:
  PyMpi() : Module() {}
  void include(py::module_ &m);
};

extern PyMpi py_mpi;
