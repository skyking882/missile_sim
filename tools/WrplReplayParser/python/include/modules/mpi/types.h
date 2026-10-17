#pragma once
#include "Module.h"

class PyMpiTypes : protected Module {
public:
  PyMpiTypes() : Module() {}
  void include(py::module_ &m);
};

extern PyMpiTypes py_mpi_types;
