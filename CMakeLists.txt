cmake_minimum_required (VERSION 3.8)
project ("AutoDiff")

set(CMAKE_CXX_FLAGS, "-Wall")

include_directories("include/")
file(GLOB SOURCES "src/*.cpp")

add_executable (${PROJECT_NAME} ${SOURCES})

add_custom_command(TARGET ${PROJECT_NAME}  POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/mnist
                ${CMAKE_CURRENT_BINARY_DIR}/mnist)
