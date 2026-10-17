add_executable(DumpMission
        main.cpp
)

# SET_TARGET_PROPERTIES(VromfsTest PROPERTIES LINK_FLAGS "/PROFILE")
target_link_libraries(DumpMission
        # mimalloc-static
        BS::BS
        dagBlk::dagBlk
        dagIoSys::dagIoSys
        FS::FS
        VROMFS::VROMFS
        ZLIB::ZLIB
)


# target_compile_definitions(VromfsTest PRIVATE MI_MALLOC_OVERRIDE)
# target_include_directories(VromfsTest PRIVATE mimalloc/include)
