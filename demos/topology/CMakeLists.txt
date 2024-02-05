include(FindPkgConfig)

add_executable(topology-fv-advection-1d advection_1d.cpp)
target_link_libraries(topology-fv-advection-1d samurai CLI11::CLI11)

add_executable(topology-fv-advection-nd advection_nd.cpp)
target_link_libraries(topology-fv-advection-nd samurai CLI11::CLI11)
