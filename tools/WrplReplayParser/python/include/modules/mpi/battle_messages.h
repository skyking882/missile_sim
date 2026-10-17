
#pragma once
#include "Module.h"

class PyBattleMessages : protected Module {
public:
  PyBattleMessages() : Module() {}
  void include(py::module_ &m);
};

extern PyBattleMessages py_battle_messages;
