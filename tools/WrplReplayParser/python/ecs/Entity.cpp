#include "modules/ecs/Entity.h"
#include "modules/ecs/EntityId.h"
#include "modules/ecs/ECSTypes.h"
#include "ecs/EntityManager.h"

PyEntity py_entity;

void PyEntity::include(py::module_ &m) {
  DO_INCLUDE()
  py_ecs_types.include(m);
  py_entity_id.include(m);
  auto ecs = m.def_submodule("ecs");

  py::class_<Entity>(ecs, "Entity")
      .def_property_readonly("templateName", [](Entity &e) {
        return &e.templ->getName();
      })
      .def_readonly("eid", &Entity::eid)
      .def("getData", &Entity::getComponent, py::arg("data_component_name"))
      .def("hasComponent", &Entity::hasComponent, py::arg("data_component_name"))
      .def("getComponentNames", &Entity::getComponentNames)
      .def("__getitem__", &Entity::getComponent, py::arg("data_component_name"));
}

py::object Entity::getComponent(const std::string &data_component_name) {
  ecs::HashedConstString hash = ECS_HASH_SLOW(data_component_name.c_str());
  auto datacomp = ecs::g_ecs_data->getDataComponents()->getDataComponent((ecs::component_t)hash.hash);
  if(!datacomp) {
    return py::none{};
  }
  auto ptr = mgr->getNullable(eid, hash);
  if(ptr) {
    return castData(datacomp->componentHash, ptr);
  }
  return py::none{};
}

bool Entity::hasComponent(const std::string &data_component_name) const {
  ecs::HashedConstString hash = ECS_HASH_SLOW(data_component_name.c_str());
  auto idx = ecs::g_ecs_data->getDataComponents()->getIndex(hash.hash);
  if(idx == ecs::INVALID_COMPONENT_INDEX) {
    return false;
  }
  return this->mgr->entityHasComponent(eid, idx);
}
py::tuple vector_to_tuple(const std::vector<py::tuple>& v) {
  py::tuple t(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    t[i] = v[i];
  }
  return t;
}

std::vector<py::tuple> Entity::getComponentNames() {
  std::vector<std::pair<ecs::ComponentInfo*, ecs::DataComponent*>> comps{};
  this->mgr->collectComponentInfo(this->eid, comps);
  std::vector<py::tuple> strs;
  for(auto [comp, data_comp] : comps) {
    if(!comp || !data_comp) {
      strs.push_back(py::make_tuple(fmt::format("{}", fmt::ptr(comp)), fmt::format("{}", fmt::ptr(data_comp))));
    } else {
      strs.push_back(py::make_tuple(comp->name, data_comp->getName()));
    }
  }
  return strs;
}
