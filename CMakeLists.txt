cmake_minimum_required(VERSION 3.10)
project(whisper_service)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp-httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/json/include
)

# Add Whisper.cpp source files
add_subdirectory(whisper.cpp)

# Common source files (shared functionality)
set(COMMON_SOURCES
    # Add shared utility functions here if needed
)

# Add main web service executable
add_executable(whisper_service main.cpp ${COMMON_SOURCES})

# Add CLI executable
add_executable(whisper_cli cli.cpp ${COMMON_SOURCES})

# Link libraries for main service
target_link_libraries(whisper_service
    PRIVATE
    whisper
    Threads::Threads
)

# Link libraries for CLI tool
target_link_libraries(whisper_cli
    PRIVATE
    whisper
    Threads::Threads
)

# Some platforms need the filesystem library explicitly linked
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(whisper_service PRIVATE stdc++fs)
    target_link_libraries(whisper_cli PRIVATE stdc++fs)
endif()

# Copy public directory to build folder
add_custom_command(TARGET whisper_service POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/public
    ${CMAKE_CURRENT_BINARY_DIR}/public
)

# Install targets
install(TARGETS whisper_service whisper_cli
    RUNTIME DESTINATION bin
)
