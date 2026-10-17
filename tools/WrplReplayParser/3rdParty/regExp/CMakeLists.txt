add_library(REGEX STATIC
    regExp.cpp
)
target_link_libraries(REGEX
        local::local
        # OPENSSL::OPENSSL
)

target_include_directories(REGEX
        PUBLIC
        "${BASE_DIR}/3rdParty"
        "${BASE_DIR}/replay/include"
        "${BASE_DIR}/src/include"
)


add_library(REGEX::REGEX ALIAS REGEX)