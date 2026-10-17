#include "modules/mpi/mpi.h"
#include "mpi/reflection.h"
void PyMpi::include(py::module_ &m) {
  DO_INCLUDE()
  auto mpi = m.def_submodule("mpi");
  py::class_<mpi::IObject> obj(mpi, "IObject");
  py::class_<mpi::Message> msg(mpi, "Message");
  obj.def_readonly("mpiObjectExtUID", &mpi::IObject::mpiObjectExtUID);
  obj.def("getUID", &mpi::IObject::getUID);
  py::class_<danet::ReflectableObject, mpi::IObject, std::unique_ptr<danet::ReflectableObject, py::nodelete>> (mpi, "ReflectableObject");
  py::class_<danet::ReplicatedObject, danet::ReflectableObject, std::unique_ptr<danet::ReplicatedObject, py::nodelete>>(mpi, "ReplicatedObject");
}
PyMpi py_mpi;