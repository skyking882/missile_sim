add_library(NET_UTILS
        netUtils.cpp
)


add_library(NET_UTILS::NET_UTILS ALIAS NET_UTILS)

target_link_libraries(NET_UTILS
        #mimalloc
        LOGGER::LOGGER
        DAG_MATH::DAG_MATH
        ZSTD::ZSTD
)

set(INCLUDES

        "${BASE_DIR}/3rdParty/Dagor_Engine/include"
        "${BASE_DIR}/src/include"
        "${BASE_DIR}/replay/include"
        # "${BASE_DIR}/src/3rdParty/fmt-12.1.0/include"
)

target_include_directories(NET_UTILS
        SYSTEM PUBLIC
        ${INCLUDES}
)

target_compile_options(NET_UTILS PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/w>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:--no-warnings>
)