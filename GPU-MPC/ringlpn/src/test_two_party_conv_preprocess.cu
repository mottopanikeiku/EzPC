// Canonical Orca forward-Conv2D specialization of the live two-process
// Ring-LPN preprocessing composition. The shared implementation preserves the
// FC path when this macro is absent.
#define RINGLPN_LIVE_CONV 1
#include "test_two_party_fc_preprocess.cu"
