#ifndef SMALL
#define SMALL

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "MiniDnn/layer.h"
#include "MiniDnn/layer/conv.h"
#include "MiniDnn/layer/fully_connected.h"
#include "MiniDnn/layer/ave_pooling.h"
#include "MiniDnn/layer/max_pooling.h"
#include "MiniDnn/layer/relu.h"
#include "MiniDnn/layer/sigmoid.h"
#include "MiniDnn/layer/softmax.h"
#include "MiniDnn/loss.h"
#include "MiniDnn/loss/mse_loss.h"
#include "MiniDnn/loss/cross_entropy_loss.h"
#include "MiniDnn/mnist.h"
#include "MiniDnn/network.h"
#include "MiniDnn/optimizer.h"
#include "MiniDnn/optimizer/sgd.h"
#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */


using namespace std;

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

void ecall_ml_small() {
    // data
//    MNIST dataset("/Users/rc/Study/Projects/mini-dnn/data/mnist/");
//    dataset.read();
//    int n_train = dataset.train_data.cols();
//    int dim_in = dataset.train_data.rows();
//    std::cout << "mnist train number: " << n_train << std::endl;
//    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
    RandData dataset(1000);
//    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    // dnn
    Network dnn;


    //------------------------------------------------------------
    Layer *conv1 = new Conv(1, 28, 28, 128, 3, 3, 1, 1, 1);
    Layer *relu1 = new ReLU;
    Layer *conv2 = new Conv(128, 28, 28, 128, 3, 3, 1, 1, 1);
    Layer *relu2 = new ReLU;
    dnn.add_layer(conv1);
    dnn.add_layer(relu1);
    dnn.add_layer(conv2);
    dnn.add_layer(relu2);

    Layer *b1_pool = new MaxPooling(128, 28, 28, 2, 2, 2);
    dnn.add_layer(b1_pool);


    Layer *fc_fc1 = new FullyConnected(b1_pool->output_dim(), 10);
    Layer *fc_relu1 = new Softmax;

    dnn.add_layer(fc_fc1);
    dnn.add_layer(fc_relu1);

    // loss
    Loss *loss = new CrossEntropy;
    dnn.add_loss(loss);
    // train & test
    SGD opt(0.001, 5e-4, 0.9, true);
    // SGD opt(0.001);
    const int n_epoch = 5;
    const int batch_size = 14;
    for (int epoch = 0; epoch < n_epoch; epoch++) {
        shuffle_data(dataset.train_data, dataset.train_labels);
        for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
            int ith_batch = start_idx / batch_size;
            Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                                      std::min(batch_size, n_train - start_idx));
            Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                                            std::min(batch_size, n_train - start_idx));
            Matrix target_batch = one_hot_encode(label_batch, 10);
            if (false && ith_batch % 10 == 1) {
                // std::cout << ith_batch << "-th grad: " << std::endl;
                printf("ith_batch: %d-th grad: \n");
                dnn.check_gradient(x_batch, target_batch, 10);
            }

            ocall_start_clock();
            dnn.forward(x_batch);
            ocall_end_clock("Forward: %f\n");


            dnn.backward(x_batch, target_batch);
            ocall_end_clock("Backward: %f\n");


            // display
            if (ith_batch % 2 == 0) {
                //std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss() << std::endl;
                printf("%d-th, loss: %f\n", ith_batch, dnn.get_loss());
            }
            // optimize
            dnn.update(opt);
        }


//        // test
        dnn.forward(dataset.test_data);
        float acc = compute_accuracy(dnn.output(), dataset.test_labels);
        // std::cout << std::endl;
        // std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
        // std::cout << std::endl;
    }
    return;
}

#endif
