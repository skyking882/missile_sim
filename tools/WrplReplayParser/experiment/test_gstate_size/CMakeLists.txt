add_executable(test_gstate_size
        test_gstate_size.cpp
)
target_link_libraries(test_gstate_size
        LOGGER::LOGGER
        fmt::fmt
        BS::BS
        dagBlk::dagBlk
        dagIoSys::dagIoSys
        FS::FS
        VROMFS::VROMFS
        FieldSerializer::FieldSerializer
        NET::NET
        dagECS::dagECS
        INIT::INIT
        Replay::Replay
        ZLIB::ZLIB
        # mimalloc

)

target_include_directories(test_gstate_size
        PUBLIC
        "${BASE_DIR}/replay/include"
        "${BASE_DIR}/src/include"
        # "${BASE_DIR}/3rdParty"
)

attach_signal_tracer(test_gstate_size)

# SetGenFlags(dagor_rpl_test)