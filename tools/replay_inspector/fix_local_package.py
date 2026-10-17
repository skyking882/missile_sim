"""Enable explicit read-only local package mounts and CDK include resolution."""
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "python/state/State.cpp"
source = path.read_text()
if 'm.def("mountLocalPackage"' not in source:
    declaration = '''
#include "FileSystem.h"
#include "ioSys/dag_dataBlock.h"
#include <cctype>

namespace {
struct LocalPackageIncludeResolver : DataBlock::IIncludeFileResolver {
  bool resolveIncludeFile(std::string &name) override {
    constexpr std::string_view prefix = "#/develop/gameBase/";
    if (!name.starts_with(prefix))
      return false;
    name.erase(0, prefix.size());
    for (char &c : name)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return true;
  }
};
LocalPackageIncludeResolver local_package_include_resolver;
}
'''
    marker = 'void PyReplayState::include(py::module_ &m) {'
    if source.count(marker) != 1:
        raise RuntimeError("Unexpected source layout")
    source = source.replace(marker, declaration + "\n" + marker)
    binding = '''
  m.def("mountLocalPackage", [](const std::string &path) {
    if (!fs::is_directory(path))
      throw std::runtime_error("Local package directory does not exist: " + path);
    file_mgr.add_mount(path);
    DataBlock::setIncludeResolver(&local_package_include_resolver);
  }, py::arg("path"), "Mount a local resource package; resolve CDK gameBase includes.");
'''
    marker = '  m.def("g_log_flush", []() {'
    if source.count(marker) != 1:
        raise RuntimeError("Unexpected binding layout")
    source = source.replace(marker, binding + "\n" + marker)
    path.write_text(source)
    print("Added explicit local package support")
elif '    name.erase(0, prefix.size());\n    return true;' in source:
    source = source.replace('    name.erase(0, prefix.size());\n    return true;',
        '    name.erase(0, prefix.size());\n    for (char &c : name)\n      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));\n    return true;')
    source = '#include <cctype>\n' + source
    path.write_text(source)
    print("Normalized CDK include paths for the VROMFS index")
