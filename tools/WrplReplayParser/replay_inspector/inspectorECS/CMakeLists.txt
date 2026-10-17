add_library(InspectorECS
        STATIC
        query_draw_units.gen.es.cpp
)

target_link_libraries(InspectorECS
        ParserState::ParserState
        imgui
)
add_library(InspectorECS::InspectorECS ALIAS InspectorECS)

set(INCLUDES
        "${BASE_DIR}/replay_inspector/include"
)

target_include_directories(InspectorECS
        PUBLIC
        ${INCLUDES}
)


SetGenFlags(InspectorECS)

