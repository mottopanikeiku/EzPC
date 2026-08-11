// Canonical Orca forward-FC command-line adapter over the reusable library.
#include "linear_preprocess.h"

int main(int argc, char **argv) {
    return ringlpn_linear::FcPreprocessor::run_cli(argc, argv);
}
