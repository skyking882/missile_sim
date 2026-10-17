
add_executable(CLogDecrypt
        DecryptFile.cpp
)

target_link_libraries(CLogDecrypt
    DECRYPT::DECRYPT
)

target_include_directories(CLogDecrypt
        PUBLIC
        "${BASE_DIR}/ClogDecryptor/include"
        "${BASE_DIR}/src/include"
)
