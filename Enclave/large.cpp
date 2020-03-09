#ifndef LARGE
#define LARGE

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

# define SIZE_CONTROLL 72
# define CATEGORY 2

class RandData {
public:
    Matrix train_data;
    Matrix train_labels;

    RandData(int size) {
        train_data = Matrix::Random(SIZE_CONTROLL * SIZE_CONTROLL, size);
        train_labels = Matrix::Ones(1, size);
    }
};

void ecall_ml_large() {
    // data
//    MNIST dataset("/Users/rc/Study/Projects/mini-dnn/data/mnist/");
//    dataset.read();
//    int n_train = dataset.train_data.cols();
//    int dim_in = dataset.train_data.rows();
//    std::cout << "mnist train number: " << n_train << std::endl;
//    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
    RandData dataset(10240);
//    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    // dnn
    Network dnn;


    // std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    // std::chrono::duration<double> elapsed;


    //------------------------------------------------------------

    Layer *fc_fc1 = new FullyConnected(SIZE_CONTROLL * SIZE_CONTROLL, SIZE_CONTROLL * SIZE_CONTROLL);
    Layer *fc_relu1 = new ReLU;

    dnn.add_layer(fc_fc1);
    dnn.add_layer(fc_relu1);

    Layer *fc_fc2 = new FullyConnected(SIZE_CONTROLL * SIZE_CONTROLL, CATEGORY);
    Layer *fc_relu2 = new Softmax;

    dnn.add_layer(fc_fc2);
    dnn.add_layer(fc_relu2);

    // loss
    Loss *loss = new CrossEntropy;
    dnn.add_loss(loss);
    // train & test
    SGD opt(0.001, 5e-4, 0.9, true);
    // SGD opt(0.001);
    const int n_epoch = 3;
    // const int batch_size = 3072;
    for (int batch_size=8192; batch_size<=10240; batch_size+=1024){
        printf("batch_size=%d\n", batch_size);
        for (int epoch = 0; epoch < n_epoch; epoch++) {
            shuffle_data(dataset.train_data, dataset.train_labels);
            for (int start_idx = 0, rounds=0; start_idx < n_train && rounds < 1; start_idx += batch_size) {
                int ith_batch = start_idx / batch_size;
                Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                                          std::min(batch_size, n_train - start_idx));
                Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                                                std::min(batch_size, n_train - start_idx));
                Matrix target_batch = one_hot_encode(label_batch, CATEGORY);


                // start = std::chrono::high_resolution_clock::now();
                ocall_start_clock();
                dnn.forward(x_batch);
                ocall_end_clock("Forward: %f  ");
                // end = std::chrono::high_resolution_clock::now();
                // elapsed = end - start;
                // std::cout << "Forward: " << elapsed.count() << std::endl;

                // start = std::chrono::high_resolution_clock::now();
                ocall_start_clock();
                dnn.backward(x_batch, target_batch);
                // end = std::chrono::high_resolution_clock::now();
                // elapsed = end - start;
                // std::cout << "Backward: " << elapsed.count() << std::endl;

                // optimize
                // dnn.update(opt);
                ocall_end_clock("Backward: %f\n");
                rounds++;
            }

        }
    }
}

#endif
