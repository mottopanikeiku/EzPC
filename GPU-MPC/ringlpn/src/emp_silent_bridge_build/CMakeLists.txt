cmake_minimum_required(VERSION 3.21)
project(ringlpn_emp_silent_bridge LANGUAGES CXX)

find_package(emp-tool 1.0 CONFIG REQUIRED)
find_package(emp-ot 1.0 CONFIG REQUIRED)
find_package(OpenSSL 3.0 REQUIRED)

add_library(ringlpn_emp_silent_bridge SHARED ../emp_silent_bridge.cpp)
target_compile_features(ringlpn_emp_silent_bridge PRIVATE cxx_std_20)
target_include_directories(ringlpn_emp_silent_bridge PRIVATE ..)
target_link_libraries(ringlpn_emp_silent_bridge PRIVATE
  emp-ot::emp-ot emp-tool::emp-tool OpenSSL::Crypto)
set_target_properties(ringlpn_emp_silent_bridge PROPERTIES
  CXX_EXTENSIONS OFF
  OUTPUT_NAME ringlpn_emp_silent_bridge
  POSITION_INDEPENDENT_CODE ON)

install(TARGETS ringlpn_emp_silent_bridge
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
