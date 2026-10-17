

#ifndef WTFILEUTILS_MODULE_H
#define WTFILEUTILS_MODULE_H
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

class Module {
  bool included = false;
public:
  void include(py::module_ &m) {};
protected:
  bool onModuleLoaded() {
    if(this->included)
      return false;
    this->included = true;
    return true;
  }
};

#define DO_INCLUDE()   if(!this->onModuleLoaded()) return;

#endif //WTFILEUTILS_MODULE_H
