#pragma once
#include "Module.h"

class PyEntityManager : protected Module {
public:
  PyEntityManager() : Module() {}
  void include(py::module_ &m);
};

extern PyEntityManager py_entity_manager;
