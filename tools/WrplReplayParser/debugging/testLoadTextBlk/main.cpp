#include "DataBlock.h"
#include "reader.h"
#include <fstream>
#include "filesystem"


namespace fs = std::filesystem;
int main() {
  fs::path base_dir = BASE_DIR;
  fs::path path = base_dir / R"(debugging\testLoadTextBlk\test.blk)";
  std::ifstream file{path};
  G_CHECKF(file.is_open(), "failed to open file");
  SharedPtr<DataBlock> blk = SharedPtr<DataBlock>::make();
  std::vector<char> characters((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  G_ASSERT(blk->loadText(characters));
  blk->printBlock(0, std::cout);
}