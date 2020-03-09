#ifndef DEMO
#define DEMO

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "MiniDnn/layer.h"
#include "MiniDnn/layer/conv.h"
#include "MiniDnn/layer/identity_block.h"
#include "MiniDnn/layer/conv_block.h"
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
//#include <time.h>


using namespace std;

class RandData {
public:
    Matrix train_data;
    Matrix train_labels;
    Matrix test_data;
    Matrix test_labels;

    RandData(int size) {
        train_data = Matrix::Ones(224 * 224 * 3, size);
        train_labels = Matrix::Ones(1, size);
        test_data = Matrix::Ones(224 * 224 * 3, size);
        test_labels = Matrix::Ones(1, size);
    }
};

void ecall_ml_resnet50() {
    // data
//    MNIST dataset("/Users/rc/Study/Projects/mini-dnn/data/mnist/");
//    dataset.read();
//    int n_train = dataset.train_data.cols();
//    int dim_in = dataset.train_data.rows();
//    std::cout << "mnist train number: " << n_train << std::endl;
//    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
    RandData dataset(100);
//    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    // dnn
    Network dnn;

//    // Conv
//    // int channel_in, int height_in, int width_in, int channel_out, int height_kernel,
//    // int width_kernel, int stride = 1, int pad_w = 0, int pad_h = 0


    Layer *input_conv1 = new Conv(3, 224, 224, 64, 7, 7, 2, 3, 3);
    Layer *input_relu1 = new ReLU;


    Layer *b1_pool = new MaxPooling(64, 112, 112, 2, 2, 2);

    dnn.add_layer(input_conv1);
    dnn.add_layer(input_relu1);
    dnn.add_layer(b1_pool);

    //------------------------------------------------------------
    Layer *s1_conv_block = new ConvBlock(64, 56, 56, 64,
                                         64, 256, 3, 3, 1);
    Layer *s1_id_block1 = new IdentityBlock(256, 56, 56, 64, 64, 3, 3);
    Layer *s1_id_block2 = new IdentityBlock(256, 56, 56, 64, 64, 3, 3);

    dnn.add_layer(s1_conv_block);
    dnn.add_layer(s1_id_block1);
    dnn.add_layer(s1_id_block2);

    //------------------------------------------------------------

    Layer *s2_conv_block = new ConvBlock(256, 56, 56, 128,
                                         128, 512, 3, 3, 2);
    Layer *s2_id_block1 = new IdentityBlock(512, 28, 28, 128, 128, 3, 3);
    Layer *s2_id_block2 = new IdentityBlock(512, 28, 28, 128, 128, 3, 3);

    dnn.add_layer(s2_conv_block);
    dnn.add_layer(s2_id_block1);
    dnn.add_layer(s2_id_block2);

    //------------------------------------------------------------

    Layer *s3_conv_block = new ConvBlock(512, 28, 28, 256,
                                         256, 1024, 3, 3, 2);
    Layer *s3_id_block1 = new IdentityBlock(1024, 14, 14, 256, 256, 3, 3);
    Layer *s3_id_block2 = new IdentityBlock(1024, 14, 14, 256, 256, 3, 3);

    dnn.add_layer(s3_conv_block);
    dnn.add_layer(s3_id_block1);
    dnn.add_layer(s3_id_block2);
    //------------------------------------------------------------

    Layer *s4_conv_block = new ConvBlock(1024, 14, 14, 512,
                                         512, 2048, 3, 3, 2);
    Layer *s4_id_block1 = new IdentityBlock(2048, 7, 7, 512, 512, 3, 3);
    Layer *s4_id_block2 = new IdentityBlock(2048, 7, 7, 512, 512, 3, 3);


    dnn.add_layer(s4_conv_block);
    dnn.add_layer(s4_id_block1);
    dnn.add_layer(s4_id_block2);
    //------------------------------------------------------------


    Layer *out_pool = new MaxPooling(2048, 7, 7, 7, 7, 1);

//    Layer *b1_pool = new MaxPooling(64, 28, 28, 2, 2, 2);
    Layer *fc_fc1 = new FullyConnected(out_pool->output_dim(), 1000);
    Layer *softmax = new Softmax;

    dnn.add_layer(out_pool);
    dnn.add_layer(fc_fc1);
    dnn.add_layer(softmax);
    

    // loss
    Loss *loss = new CrossEntropy;
    dnn.add_loss(loss);
    // train & test
    SGD opt(0.001, 5e-4, 0.9, true);
    // SGD opt(0.001);
    // const int n_epoch = 5;
    const int total_rounds[4] = {1, 2, 3, 4};
    const int batch_size = 20;
    for (int idx = 0; idx < 4; idx++) {
        int n_epoch = total_rounds[idx];
        printf("total_rounds: %d\n", 5 * n_epoch);
        ocall_start_clock();
        for (int epoch = 0; epoch < n_epoch; epoch++) {
            shuffle_data(dataset.train_data, dataset.train_labels);
            for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
            // for(int start_idx=0, rounds=0; rounds < total_rounds; start_idx += batch_size){
                int ith_batch = start_idx / batch_size;
                Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                                          std::min(batch_size, n_train - start_idx));
                Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                                                std::min(batch_size, n_train - start_idx));
                Matrix target_batch = one_hot_encode(label_batch, 1000);
                // if (false && ith_batch % 10 == 1) {
                //     std::cout << ith_batch << "-th grad: " << std::endl;
                //     dnn.check_gradient(x_batch, target_batch, 10);
                // }
                
                dnn.forward(x_batch);
                // ocall_end_clock("Forward: %f   ");
                
                // ocall_start_clock();
                dnn.backward(x_batch, target_batch);

                
    //            // display
    //            if (ith_batch % 2 == 0) {
    //                //std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss() << std::endl;
    //                printf("%d-th batch, loss: %f\n", ith_batch, dnn.get_loss());
    //            }
                // optimize
                dnn.update(opt);
                // rounds++;
            }
        }
        ocall_end_clock("total: %f\n");

// //        // test
//         dnn.forward(dataset.test_data);
//         float acc = compute_accuracy(dnn.output(), dataset.test_labels);
        // std::cout << std::endl;
        // std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
        // std::cout << std::endl;
    }
}



#endif