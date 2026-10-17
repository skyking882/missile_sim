#pragma once
#include "Module.h"
class PyReplayState : protected Module {
public:
  PyReplayState() : Module() {}
  void include(py::module_ &m);
};

extern PyReplayState py_replay_state;
