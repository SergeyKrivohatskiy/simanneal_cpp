project(tests)


include(ExternalProject)
ExternalProject_Add(catch
    GIT_REPOSITORY https://github.com/philsquared/Catch.git
    GIT_TAG "v1.5.6"
    TIMEOUT 10
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
    DOWNLOAD_DIR 
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )
ExternalProject_Get_Property (catch source_dir)
set(CATCH_INCLUDE_DIR ${source_dir}/include)


file (GLOB SOURCE_FILES "*.cpp" ".hpp")

add_executable (tests ${SOURCE_FILES})
include_directories (${simanneal_cpp_SOURCE_DIR} ${CATCH_INCLUDE_DIR})
target_link_libraries (tests simanneal_cpp)
add_dependencies (tests catch)