#include "network/Connection.h"
#include "FileSystem.h"
#include "init/initialize.h"

#define STB_IMAGE_IMPLEMENTATION

#include "network/CNetwork.h"
#include "network/eid.h"

class doSerialization {
public:
  doSerialization() {
    fs::remove_all(basePath / "creation");
    fs::remove_all(basePath / "replication");
    fs::create_directory(basePath / "creation");
    fs::create_directory(basePath / "replication");
  }

  void addEntity(ecs::EntityId eid) {
    if (curr_eids >= this->eids_to_repl.size()) {
      eids_to_repl.resize(curr_eids + 1);
    }
    if (this->eids_to_repl[curr_eids].size() >= 250) {
      curr_eids++;
      if (curr_eids >= this->eids_to_repl.size()) {
        eids_to_repl.resize(curr_eids + 1);
      }
    }
    this->eids_to_repl[curr_eids].push_back(eid);
  }

  void WriteReplication(ecs::component_index_t cidx, float time)
  {
    auto ms = static_cast<uint32_t>(time * 1000.0f);
    BitStream bs;
    BitStream tmpbs;
    uint16_t localSequence = 0;

    bs.Write((char)net::ID_ENTITY_REPLICATION);
    bs.Write(ms); // write header and curr time
    bs.Write(localSequence);
    for (int z = 0; z < 24; z++)
      for (int x = 0; x < 24; x++) {
        ecs::EntityId eid = ecs::EntityId(0x400000 + x + z * 24);
        bs.Write(true); // entity exists
        net::write_server_eid(eid.get_handle(), bs);
        memset(tmpbs.GetData(), 0, tmpbs.GetNumberOfBytesUsed());
        tmpbs.ResetWritePointer();
        tmpbs.Write(true);
        auto comp = ecs::g_entity_mgr->getComponentRefCidx(eid, cidx);
        conn.serializeComponentReplication(eid, comp, tmpbs);
        tmpbs.Write(false);

        uint32_t blockSizeBytes = tmpbs.GetNumberOfBytesUsed();
        if (blockSizeBytes)
        {
          bs.WriteCompressed(blockSizeBytes);
          bs.Write((const char *)tmpbs.GetData(), blockSizeBytes);
        }
      }

    bs.Write(false);
    std::ofstream outputFile(basePath / std::format("replication/{}", ms), std::ios::out | std::ios::binary);
    outputFile.write((const char *)bs.GetData(), BITS_TO_BYTES(bs.GetWriteOffset()));
  }

  void compile() {
    int ind = 0;
    for (auto &eids: this->eids_to_repl) {

      BitStream bs;
      BitStream tmpbs;
      uint8_t construct_header = 0x24;
      bs.Write(construct_header);
      conn.writeConstructionPacket(bs, tmpbs, eids);
      std::ofstream outputFile(basePath / std::format("creation/packet{}.bin", ind), std::ios::out | std::ios::binary);
      outputFile.write((const char *)bs.GetData(), BITS_TO_BYTES(bs.GetWriteOffset()));
      ind++;
    }
  }

private:
  fs::path basePath = fs::path(BASE_DIR) / R"(create_entitiy)";
  std::vector<std::vector<ecs::EntityId>> eids_to_repl{};
  int curr_eids=0;
  net::Connection conn{};
};
struct RGB {
  uint8_t r, g, b;
};

class ReadVideo {
public:
  ReadVideo(std::string &path)
  {
    file_data.resize(fs::file_size(path));

    std::ifstream source_file { path, std::ios::binary };
    source_file.read( reinterpret_cast<char*>(file_data.data()), static_cast<long>(file_data.size()));
    frame_count = file_data.size() / (24*24*3);
  }

  RGB * getNextFrame()
  {
    if(frame_ptr>=frame_count)
      return nullptr;
    auto buff = file_data.data()+frame_ptr*24*24*3;
    frame_ptr++;
    return (RGB*)buff;
  }


  std::vector<char> file_data;
  uint32_t frame_count;
  uint32_t frame_ptr = 0;
};

int main() {
  std::string basePath = R"(D:\SteamLibrary\steamapps\common\War Thunder\cache\binary.2.49.0)";
  std::string path = R"()";

  initialize(basePath);
  int width=24, height=24;//, channels;
  ReadVideo video{path};



  G_ASSERT(width == 24 && height == 24);

  //E3DCOLOR color[24][24];
  //memcpy(color, data, 24 * 24 * 4);


  doSerialization serializer{};

  ecs::template_t tid = ecs::g_entity_mgr->getTemplateDB()->getTemplateIdByName("omni_light");
  ecs::g_entity_mgr->instantiateTemplate(tid);

  std::cout.flush();
  for (int z = 0; z < height; z++)
    for (int x = 0; x < width; x++) {
      //E3DCOLOR curr_color = color[x][z];


      TMatrix tm;
      tm.m[0][0] = 1.0;
      tm.m[0][1] = 0.0;
      tm.m[0][2] = 0.0;

      tm.m[1][0] = 0.0;
      tm.m[1][1] = 1.0;
      tm.m[1][2] = 0.0;

      tm.m[2][0] = 0.0;
      tm.m[2][1] = 1.0;
      tm.m[2][2] = 0.0;

      tm.m[3][0] = 1706.0 - x * 3;
      tm.m[3][1] = 630.0 + 3.5;
      tm.m[3][2] = 2888.0 - z * 3;
      ecs::ComponentsInitializer init{};


      init[ECS_HASH("transform")] = std::move(ecs::Component(tm));
      init[ECS_HASH("light__nightly")] = std::move(ecs::Component(false));
      init[ECS_HASH("light__brightness")] = std::move(ecs::Component((float) 1.0));
      init[ECS_HASH("light__radius_scale")] = std::move(ecs::Component((float) 3));
      init[ECS_HASH("light__max_radius")] = std::move(ecs::Component((float) 700.0));
      init[ECS_HASH("light__color")] = std::move(ecs::Component(E3DCOLOR(255, 255, 255, 255)));
      ecs::EntityId eid = ecs::EntityId(0x400000 + x + z * height);
      init[ECS_HASH("eid")] = std::move(ecs::Component(eid));


      ecs::g_entity_mgr->createEntity(eid, tid, std::move(init));


      serializer.addEntity(eid);
    }
  serializer.compile();
  auto hsh = ECS_HASH("light__color");
  auto cidx = ecs::g_entity_mgr->getDataComponents()->getIndex(hsh.hash);
  float curr_time = 40.0;
  while(auto colors_ = video.getNextFrame())
  {
    for (int z = 0; z < height; z++)
      for (int x = 0; x < width; x++) {
        RGB color = colors_[(width - 1 - x)+z*height];
        ecs::EntityId eid = ecs::EntityId(0x400000 + x + z * height);
        E3DCOLOR *col = ecs::g_entity_mgr->getNullable<E3DCOLOR>(eid, hsh);
        col->r = color.r;
        col->g = color.g;
        col->b = color.b;

      }
    serializer.WriteReplication(cidx, curr_time);
    curr_time+=1.0/30;
  }
}