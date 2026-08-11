cmake_minimum_required(VERSION 3.17)
project(ringlpn_graph_libraries LANGUAGES CXX)

if(NOT DEFINED RINGLPN_REPO_ROOT)
  message(FATAL_ERROR "RINGLPN_REPO_ROOT is required")
endif()
get_filename_component(RINGLPN_REPO_ROOT "${RINGLPN_REPO_ROOT}" ABSOLUTE)
set(SYTORCH_ROOT "${RINGLPN_REPO_ROOT}/GPU-MPC/ext/sytorch")
set(CRYPTO_TOOLS_ROOT "${SYTORCH_ROOT}/ext/cryptoTools")
set(BITPACK_ROOT "${SYTORCH_ROOT}/ext/bitpack")
set(LLAMA_ROOT "${SYTORCH_ROOT}/ext/llama")
set(EIGEN_ROOT "${RINGLPN_REPO_ROOT}/SCI/extern/eigen")

foreach(required_path
    "${CRYPTO_TOOLS_ROOT}/cryptoTools/Crypto/PRNG.cpp"
    "${BITPACK_ROOT}/src/bitpack/bitpack.cpp"
    "${LLAMA_ROOT}/src/llama/config.cpp"
    "${EIGEN_ROOT}/Eigen/Dense"
    "${LLAMA_ROOT}/api.cpp")
  if(NOT EXISTS "${required_path}")
    message(FATAL_ERROR
      "Missing graph dependency source: ${required_path}. Initialize recursive submodules before building.")
  endif()
endforeach()

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

function(ringlpn_library_defaults target seed)
  target_compile_options(${target} PRIVATE
    -maes -msse4.1 -mpclmul -mavx2 -mrdseed
    -Wno-ignored-attributes
    "-ffile-prefix-map=${RINGLPN_REPO_ROOT}=/ringlpn/source"
    "-fdebug-prefix-map=${RINGLPN_REPO_ROOT}=/ringlpn/source"
    "-frandom-seed=${seed}")
endfunction()

add_library(cryptoTools STATIC
  "${CRYPTO_TOOLS_ROOT}/cryptoTools/Common/Defines.cpp"
  "${CRYPTO_TOOLS_ROOT}/cryptoTools/Common/Log.cpp"
  "${CRYPTO_TOOLS_ROOT}/cryptoTools/Crypto/AES.cpp"
  "${CRYPTO_TOOLS_ROOT}/cryptoTools/Crypto/PRNG.cpp")
target_include_directories(cryptoTools PUBLIC "${CRYPTO_TOOLS_ROOT}")
ringlpn_library_defaults(cryptoTools ringlpn-cryptotools-20260810)

add_library(bitpack STATIC "${BITPACK_ROOT}/src/bitpack/bitpack.cpp")
target_include_directories(bitpack PUBLIC "${BITPACK_ROOT}/include")
ringlpn_library_defaults(bitpack ringlpn-bitpack-20260810)

add_library(LLAMA STATIC
  "${LLAMA_ROOT}/src/llama/config.cpp"
  "${LLAMA_ROOT}/src/llama/comms.cpp"
  "${LLAMA_ROOT}/src/llama/input_prng.cpp"
  "${LLAMA_ROOT}/src/llama/prng.cpp"
  "${LLAMA_ROOT}/src/llama/stats.cpp"
  "${LLAMA_ROOT}/src/llama/utils.cpp"
  "${LLAMA_ROOT}/and.cpp"
  "${LLAMA_ROOT}/api.cpp"
  "${LLAMA_ROOT}/conv.cpp"
  "${LLAMA_ROOT}/dcf.cpp"
  "${LLAMA_ROOT}/mult.cpp"
  "${LLAMA_ROOT}/pubdiv.cpp"
  "${LLAMA_ROOT}/relu.cpp"
  "${LLAMA_ROOT}/signextend.cpp"
  "${LLAMA_ROOT}/pubcmp.cpp"
  "${LLAMA_ROOT}/clip.cpp"
  "${LLAMA_ROOT}/dpf.cpp"
  "${LLAMA_ROOT}/lut.cpp"
  "${LLAMA_ROOT}/select.cpp"
  "${LLAMA_ROOT}/fixtobfloat16.cpp"
  "${LLAMA_ROOT}/wrap.cpp"
  "${LLAMA_ROOT}/float.cpp"
  "${LLAMA_ROOT}/taylor.cpp"
  "${LLAMA_ROOT}/mic.cpp"
  "${LLAMA_ROOT}/msnzb.cpp"
  "${LLAMA_ROOT}/truncate.cpp")
target_compile_options(LLAMA PRIVATE
  -Wno-write-strings -Wno-unused-result -Wno-deprecated-declarations -fopenmp)
target_include_directories(LLAMA PUBLIC
  "${LLAMA_ROOT}/include" "${LLAMA_ROOT}" "${EIGEN_ROOT}")
target_link_libraries(LLAMA PUBLIC Threads::Threads cryptoTools bitpack)
ringlpn_library_defaults(LLAMA ringlpn-llama-20260810)
