#include "ecs/EntityManager.h"
#include "modules/ecs/EntityManager.h"
#include "modules/ecs/EntityId.h"
#include "modules/ecs/Entity.h"

PyEntityManager py_entity_manager;
void PyEntityManager::include(py::module_ &m) {
  DO_INCLUDE()
  py_entity.include(m);
  auto ecs = m.def_submodule("ecs");
  py::class_<ecs::EntityManager>(ecs, "EntityManager")
      .def("getEntityTemplateName", &ecs::EntityManager::getEntityTemplateName, py::arg("eid"))
      .def("getEntity", [](ecs::EntityManager &mgr, ecs::EntityId &eid) -> std::optional<Entity> {
        if(mgr.doesEntityExist(eid)) {
          auto templ = mgr.getEntityTemplateId(eid);
          auto t = ecs::g_ecs_data->getTemplateDB()->getTemplate(templ);
          return Entity(eid, t, &mgr);
        }
        return std::nullopt;
      }, py::arg("eid"))
      .def("getEntity", [](ecs::EntityManager &mgr, uint32_t &eid_t) -> std::optional<Entity> {
        ecs::EntityId eid{eid_t};
        if(mgr.doesEntityExist(eid)) {
          auto templ = mgr.getEntityTemplateId(eid);
          auto t = ecs::g_ecs_data->getTemplateDB()->getTemplate(templ);
          return Entity(eid, t, &mgr);
        }
        return std::nullopt;
      }, py::arg("eid"));

}