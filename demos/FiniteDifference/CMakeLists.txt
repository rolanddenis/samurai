include(FindPkgConfig)

add_executable(finite-difference-transport-1d-uniform transport_1d_uniform.cpp)
target_link_libraries(finite-difference-transport-1d-uniform samurai CLI11::CLI11)

# add_subdirectory(BZ)
