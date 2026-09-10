// Canonical Orca forward-Conv2D command-line adapter over the reusable library.
#include "linear_preprocess.h"

int main(int argc, char **argv) {
    return ringlpn_linear::Conv2dPreprocessor::run_cli(argc, argv);
}
