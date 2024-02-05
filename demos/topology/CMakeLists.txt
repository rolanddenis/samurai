include(FindPkgConfig)

add_executable(topology-fv-advection-1d advection_1d.cpp)
target_link_libraries(topology-fv-advection-1d samurai CLI11::CLI11)
