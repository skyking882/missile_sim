add_library(NET
        STATIC
        CNetwork.cpp
        eid.cpp
        Connection.cpp
        Object.cpp
        # Message.cpp
)

target_link_libraries(NET
        ZSTD
        LZ4::LZ4
        dagECS::dagECS
        ENET::ENET
)
add_library(NET::NET ALIAS NET)

set(INCLUDES
        "${BASE_DIR}/src/include"
        # "${BASE_DIR}/3rdParty/arc/lz4"
        # "${BASE_DIR}/3rdParty/arc/zstd-1.4.5"
        # "${BASE_DIR}/3rdParty/arc"
        # "${BASE_DIR}/3rdParty/arc/lz4"
        # "${BASE_DIR}/3rdParty/arc/zlib-ng"
        "${BASE_DIR}/replay/include"
)

target_include_directories(NET
        PUBLIC
        ${INCLUDES}
)


SetGenFlags(NET)

get_target_property(flags NET COMPILE_OPTIONS)
message(STATUS "NET COMPILE_OPTIONS: ${flags}")

get_target_property(defs NET COMPILE_DEFINITIONS)
message(STATUS "NET COMPILE_DEFINITIONS: ${defs}")

get_target_property(props NET PROPERTIES)
message(STATUS "NET PROPERTIES: ${props}")

# Print full compile command (when using Ninja generator)
get_target_property(dummy NET TYPE) # Forces evaluation