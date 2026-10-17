

#ifndef WTFILEUTILS_REPLAY_H
#define WTFILEUTILS_REPLAY_H

#include "Module.h"

class PyReplay : protected Module {
public:
    PyReplay() : Module() {}
    void include(py::module_ &m);
};

extern PyReplay py_replay;

#endif //WTFILEUTILS_REPLAY_H
