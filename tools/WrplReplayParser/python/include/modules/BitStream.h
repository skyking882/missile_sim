

#pragma once
#include "Module.h"



class PyBitStream : protected Module {
public:
  PyBitStream() : Module() {}
  void include(py::module_ &m);
};

extern PyBitStream py_bitstream;
