#pragma once
#include "Module.h"

class PyEntityId : protected Module {
public:
  PyEntityId() : Module() {}
  void include(py::module_ &m);
};

extern PyEntityId py_entity_id;
