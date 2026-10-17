"""Expose cleanup of Python callbacks before CPython finalization.

Upstream retains py::function objects in a process-static vector. Releasing them
from its C++ static destructor after CPython shuts down causes a SIGSEGV.
Clear only callable references while Python is still alive; leave ECS descriptors
in place until their ordinary destruction. This does not change replay decoding.
"""
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "python/ecs/ECSQueryES.cpp"
source = path.read_text()
marker = '  auto ecs = m.def_submodule("ecs");'
addition = '''
  ecs.def("clearPythonCallbacks", []() {
    for (auto &system : registered_systems) {
      system->cb = py::function();
    }
  }, "Release Python callback references before interpreter shutdown.");'''
if 'ecs.def("clearPythonCallbacks"' not in source:
    if source.count(marker) != 1:
        raise RuntimeError("Unexpected upstream source; cleanup patch was not applied")
    path.write_text(source.replace(marker, marker + addition))
    print("Added clearPythonCallbacks lifecycle hook")
