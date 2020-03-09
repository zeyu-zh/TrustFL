//#include <fdeep/fdeep.hpp>

#include <stdarg.h>
#include <stdio.h>      /* vsnprintf */

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */

#include "sgx_trts.h"
#include "sgx_tcrypto.h"

//#include "sgxdnn_main.hpp"
#include "tensor_types.h"
//#include "Crypto.h"

#include <Eigen/Dense>
#include <algorithm>
#include "layer.h"
#include "layer/conv.h"
#include "layer/fully_connected.h"
#include "layer/ave_pooling.h"
#include "layer/max_pooling.h"
#include "layer/relu.h"
#include "layer/sigmoid.h"
#include "layer/softmax.h"
#include "loss.h"
#include "loss/mse_loss.h"
#include "loss/cross_entropy_loss.h"
#include "mnist.h"
#include "network.h"
#include "optimizer.h"
#include "optimizer/sgd.h"


/*
 * Invokes OCALL to display the enclave buffer to the terminal.
 */
void printf(const char *fmt, ...){
	char buf[BUFSIZ] = {'\0'};
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, BUFSIZ, fmt, ap);
	va_end(ap);
	ocall_print_string(buf);
}

/*
 * Invokes OCALL to start the clock.
 */
void start_clock() {
	ocall_start_clock();
}

/*
 * Invokes OCALL to end the clock.
 */
void end_clock(const char* str) {
	ocall_end_clock(str);
}


typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Vector;
typedef Eigen::Array<float, 1, Eigen::Dynamic> RowVector;

class RandData {
public:
    Matrix train_data;
    Matrix train_labels;
    Matrix test_data;
    Matrix test_labels;

    RandData(int size) {
        train_data = Matrix::Random(28 * 28, size);
        train_labels = Matrix::Ones(1, size);
        test_data = Matrix::Random(28 * 28, size);
        test_labels = Matrix::Ones(1, size);
    }
};
