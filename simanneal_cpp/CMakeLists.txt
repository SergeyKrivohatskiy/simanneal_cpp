project(simanneal_cpp)

set (HEADER_FILES "Annealer.hpp Annealer.inl")

add_library (simanneal_cpp INTERFACE)

install (FILES "${HEADER_FILES}" DESTINATION include)