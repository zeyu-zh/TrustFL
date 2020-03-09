#ifndef DEMO
#define DEMO

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include "verify.h"
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

#include <vector>
#include <string>
//#include <sstream>
//using namespace std;

extern uint32_t seed;
unsigned int prf(unsigned int next)
{
    next = next * 1103515245 + 12345;
    return (unsigned int) (next / 65536) % 60000;
}
float minus_abs(float a, float b)
{
    float temp = a - b;
    if (temp < 0)
        temp = temp * -1.0;
    return temp;
}
int judge(std::vector<std::vector<float> > vnet, int ith_batch)
{
    int error_number = 0;
    float *p_conv1, *p_conv2, *p_fc1, *p_fc2, *p_fc3;
    sgx_get_parameters(ith_batch, &p_conv1, &p_conv2, &p_fc1, &p_fc2, &p_fc3);
    std::vector<float> v1,v2,v3,v4,v5;
    for(int i = 0; i < CONV1_SIZE; i++)
        v1.push_back(p_conv1[i]);
    for(int i = 0; i < CONV2_SIZE; i++)
        v2.push_back(p_conv2[i]);
    for(int i = 0; i < FC1_SIZE; i++)
        v3.push_back(p_fc1[i]);
    for(int i = 0; i < FC2_SIZE; i++)
        v4.push_back(p_fc2[i]);
    for(int i = 0; i < FC3_SIZE; i++)
        v5.push_back(p_fc3[i]);
    sgx_free_parameters(p_conv1, p_conv2, p_fc1, p_fc2, p_fc3);
    std::vector<float> all;
    std::vector<float> all2;
    all.insert(all.end(), v1.begin(), v1.end());
    all.insert(all.end(), v2.begin(), v2.end());
    all.insert(all.end(), v3.begin(), v3.end());
    all.insert(all.end(), v4.begin(), v4.end());
    all.insert(all.end(), v5.begin(), v5.end());
    int index = 0;
    for (int i = 0; i < vnet.size(); i++)
        for (int j = 0; j < vnet[i].size(); j++)
            all2.push_back(vnet[i][j]);
    if (all.size() != all2.size())
        return 0;
    for (int i = 0; i < all.size(); i++)
    {
        if (minus_abs(all[i], all2[i])>0.00001)
            error_number++;
        if (error_number > 0)
            return 0;
    }
    return 1;
}


#define FILE_NAME "README.md"
void ecall_ml_vgg16() {
    //const int ith_batch = 120;
    const int proof_number = 5;
    const int batch_size = 128;
    const int total_rounds = 200;
    // read data
    printf("[Enclave]: Start loading data... \n");
    MNIST dataset("./Training/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    printf("[Enclave]: mnist train number:%d\n", n_train);

    std::vector<int> result;
    std::vector<int> proof;
    printf("[Enclave]: Round number:");
    for(int i = 0; i < proof_number; i++){
        unsigned int temp = 0;
        sgx_read_rand((uint8_t*)&temp, 32);
        temp = temp % total_rounds;
        while(std::find(proof.begin(), proof.end(), temp) != proof.end()){
            sgx_read_rand((uint8_t*)&temp, 32);
            temp = temp % total_rounds;
        }
        proof.push_back((int)temp);
        printf("%d, ", temp);
    }
    printf(" are going to be verified...\n");
    for (int number = 0; number < proof.size(); number++){
        int ith_batch = proof[number];
        Network dnn;
        Layer *conv1 = new Conv(1, 28, 28, 6, 5, 5, 1, 2, 2);
        Layer *pool1 = new MaxPooling(6, 28, 28, 2, 2, 2);
        Layer *conv2 = new Conv(6, 14, 14, 16, 5, 5, 1, 0, 0);
        Layer *pool2 = new MaxPooling(16, 10, 10, 2, 2, 2);
        Layer *fc3 = new FullyConnected(pool2->output_dim(), 120);
        Layer *fc4 = new FullyConnected(120, 84);
        Layer *fc5 = new FullyConnected(84, 10);
        Layer *relu1 = new ReLU;
        Layer *relu2 = new ReLU;
        Layer *relu3 = new ReLU;
        Layer *relu4 = new ReLU;
        Layer *softmax = new Softmax;

        //loading data
        float *p_conv1, *p_conv2, *p_fc1, *p_fc2, *p_fc3;
        sgx_get_parameters(ith_batch, &p_conv1, &p_conv2, &p_fc1, &p_fc2, &p_fc3);
        std::vector<float> v1,v2,v3,v4,v5;
        for(int i = 0; i < CONV1_SIZE; i++) v1.push_back(p_conv1[i]);
        for(int i = 0; i < CONV2_SIZE; i++) v2.push_back(p_conv2[i]);
        for(int i = 0; i < FC1_SIZE; i++) v3.push_back(p_fc1[i]);
        for(int i = 0; i < FC2_SIZE; i++) v4.push_back(p_fc2[i]);
        for(int i = 0; i < FC3_SIZE; i++) v5.push_back(p_fc3[i]);

        if(0 == sgx_check_parameter(ith_batch, p_conv1, p_conv2, p_fc1, p_fc2, p_fc3))
            printf("[Enclave]: Round %d parameter checking succeeded!\n", ith_batch);
        else{
            printf("[Enclave]: Round %d parameter checking failed!\n", ith_batch);
            return;
        }
        sgx_free_parameters(p_conv1, p_conv2, p_fc1, p_fc2, p_fc3);
        
        conv1->set_parameters(v1);
        conv2->set_parameters(v2);
        fc3->set_parameters(v3);
        fc4->set_parameters(v4);
        fc5->set_parameters(v5);
        
        //apply
        dnn.add_layer(conv1);
        dnn.add_layer(relu1);
        dnn.add_layer(pool1);
        dnn.add_layer(conv2);
        dnn.add_layer(relu2);
        dnn.add_layer(pool2);
        dnn.add_layer(fc3);
        dnn.add_layer(relu3);
        dnn.add_layer(fc4);
        dnn.add_layer(relu4);
        dnn.add_layer(fc5);
        dnn.add_layer(softmax);

        // loss
        Loss *loss = new CrossEntropy;
        dnn.add_loss(loss);

        // train & test
        SGD opt(0.001);
        Matrix x_batch;
        Matrix label_batch;
        x_batch.resize(dim_in, batch_size);
        label_batch.resize(1, batch_size);
        for(int j = 0; j < batch_size; j++){ //confirm training data
            unsigned int index = prf(seed * ith_batch + j);
            uint8_t* p_data = (uint8_t*)malloc(784), label;
            int ret_val;
            ocall_get_data(&ret_val, index, p_data, &label);
            if(0 != sgx_check_data(index, p_data, label)){
                printf("[Enclave]: Round %d data check failed!\n", ith_batch);
                return;
            }
            for(int i = 0; i < dim_in; i++)
                x_batch(i, j) = dataset.train_data(i, index);  //opt. std::copy (cols first)
            label_batch(0, j) = dataset.train_labels(0, index);
        }
        printf("[Enclave]: Round %d data check succeeded!\n", ith_batch);
        //start training
        printf("[Enclave]: Verify round %d training...\n", ith_batch);
        Matrix target_batch = one_hot_encode(label_batch, 10);
        dnn.forward(x_batch);
        dnn.backward(x_batch, target_batch);
        // optimize
        dnn.update(opt);

        //jduge
        // printf("[Enclave] Judging...\n");
        std::vector<std::vector<float> > vnet = dnn.get_parameters();
        if (judge(vnet, ith_batch + 1)){
            result.push_back(ith_batch);
            printf("[Enclave]: Round %d matched\n", ith_batch);
        }
        else{
            printf("[Enclave]: Round %d not matched\n", ith_batch);
        }
    }
    if(proof.size()==result.size())
    	printf("\n[Enclave]: Congratulations! Your training has been verified successfully\n\n");
    else
    	printf("[Enclave]: Verification failed\n");
}



#endif
