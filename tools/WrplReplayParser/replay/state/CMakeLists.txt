add_library(ParserState STATIC
        ParserState.cpp
)


target_link_libraries(ParserState
        FieldSerializer::FieldSerializer
        dagBlk::dagBlk
        FS::FS
        VROMFS::VROMFS
        # REGEX::REGEX
        XXH3
        NET::NET
        ZSTD::ZSTD
        DAG_MATH::DAG_MATH
        NET_UTILS::NET_UTILS
        DELTA::DELTA
)
add_library(ParserState::ParserState ALIAS ParserState)

set(INCLUDES
        "${BASE_DIR}/replay/include"
        "${BASE_DIR}/src/include"
)

target_include_directories(ParserState
        PUBLIC
        ${INCLUDES}
)


SetGenFlags(ParserState)