add_library(FS
        STATIC
        FSObject.cpp
        Directory.cpp
        File.cpp
        FileManager.cpp
)


target_link_libraries(FS
        ZSTD::ZSTD
        ZLIB::ZLIB
        DAG_MATH::DAG_MATH
        dagBlk::dagBlk
        VROMFS::VROMFS

)

add_library(FS::FS ALIAS FS)

set(INCLUDES
        "${BASE_DIR}/src/include"
)

target_include_directories(FS
        PUBLIC
        ${INCLUDES}
)


SetGenFlags(FS)