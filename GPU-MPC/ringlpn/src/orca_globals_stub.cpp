// Standalone ringlpn benches link this instead of utils/sigma_comms.cpp
// (which would drag in the full SigmaPeer comms stack). The value must match
// the definition in GPU-MPC/utils/sigma_comms.cpp.
#include <cstddef>

size_t OneGB = 1024 * 1024 * 1024;
