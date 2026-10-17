#pragma once
#include "Module.h"
#include "ecs/entityId.h"
namespace ecs {
  class Template;
  class EntityManager;
}
struct Entity {
  ecs::EntityId eid;
  ecs::Template * templ;
  ecs::EntityManager * mgr;
  py::object getComponent(const std::string &data_component_name);
  bool hasComponent(const std::string &data_component_name) const;
  std::vector<py::tuple> getComponentNames();
};



class PyEntity : protected Module {
public:
  PyEntity() : Module() {}
  void include(py::module_ &m);
};

extern PyEntity py_entity;
