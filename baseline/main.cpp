#include <iostream>

#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"
#include "dynet/rnn.h"


#include "src/myutils.h"
#include "src/race.h"
#include "src/mlp.h"

using namespace std;
using namespace dynet;
using namespace myutil;

int main(int argc, char **argv) {
    PO::variables_map conf = myutil::InitCommandLine(argc, argv);
    if (conf.count("heuristic_choose")) {
      cout << "RUN heuristic choice" << endl;
      run_heuristic_choice(conf);
    }
    if (conf.count("mlp")) {
      cout << "RUN Multi Layer Perceptron" << endl;
      run_mlp(argc, argv, conf);
    }
}
