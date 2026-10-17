
#pragma once
#include "Module.h"

class PyCodeGenObjects : protected Module {
public:
  PyCodeGenObjects() : Module() {}
  void include(py::module_ &m);
};

extern PyCodeGenObjects py_CodeGen;
