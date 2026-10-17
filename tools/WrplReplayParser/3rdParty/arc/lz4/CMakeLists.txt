

add_library(LZ4 STATIC
        lz4.c
        lz4hc.c
)

target_include_directories(LZ4
        SYSTEM PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}"
)


add_library(LZ4::LZ4 ALIAS LZ4)

target_compile_options(LZ4 PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/w>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:--no-warnings>
)