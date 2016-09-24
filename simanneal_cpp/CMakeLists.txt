project(simanneal_cpp)

set (HEADER_FILES "Annealer.hpp")
set (SOURCE_FILES "Annealer.cpp")

add_library (simanneal_cpp ${SOURCE_FILES} ${HEADER_FILES})

install (TARGETS simanneal_cpp DESTINATION bin)
install (FILES "${HEADER_FILES}" DESTINATION include)