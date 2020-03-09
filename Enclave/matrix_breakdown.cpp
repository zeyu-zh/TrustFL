#ifndef MATRIX
#define MATRIX

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

using namespace Eigen;
using namespace std;

//#define N 4



void Multiply_directly(int N) {
    // std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    // std::chrono::duration<double> elapsed;
    MatrixXf X = MatrixXf::Random(N, N);
    MatrixXf Y = MatrixXf::Random(N, N);
    MatrixXf Z = MatrixXf::Zero(N, N);
    //start = std::chrono::high_resolution_clock::now();
    ocall_start_clock();
    for (int i = 0; i < 5; i++) {
        Z = X * Y;
    }
    //end = std::chrono::high_resolution_clock::now();
    ocall_end_clock("%f: ");
    printf("Direct[%d*%d]\n", N, N);
    // elapsed = end - start;
    // std::cout << "Direct[" << N << '*' << N << "]:" << elapsed.count() << std::endl;

}

void Multiply_breakdown(int N) {
    // std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    // std::chrono::duration<double> elapsed;
    MatrixXf Z = MatrixXf::Zero(N, N);

    MatrixXf XA = MatrixXf::Random(N / 2, N / 2);
    MatrixXf XC = MatrixXf::Random(N / 2, N / 2);
    MatrixXf XB = MatrixXf::Random(N / 2, N / 2);
    MatrixXf XD = MatrixXf::Random(N / 2, N / 2);

    MatrixXf YA = MatrixXf::Random(N / 2, N / 2);
    MatrixXf YC = MatrixXf::Random(N / 2, N / 2);
    MatrixXf YB = MatrixXf::Random(N / 2, N / 2);
    MatrixXf YD = MatrixXf::Random(N / 2, N / 2);

    //start = std::chrono::high_resolution_clock::now();
    ocall_start_clock();
    for (int i = 0; i < 5; i++) {
        Z.block(0, 0, N / 2, N / 2) = XA * YA + XB * YC;
        Z.block(0, N / 2, N / 2, N / 2) = XA * YB + XB * YD;
        Z.block(N / 2, 0, N / 2, N / 2) = XC * YA + XD * YC;
        Z.block(N / 2, N / 2, N / 2, N / 2) = XC * YB + XD * YD;
    }
    ocall_end_clock("%f: ");
    printf("Break[%d*%d]\n", N, N);
    //end = std::chrono::high_resolution_clock::now();

    // elapsed = end - start;
    // std::cout << "Break[" << N << '*' << N << "]:" << elapsed.count() << std::endl;

}

void ecall_ml_matrix_breakdown() {


    for (int i = 256; i <= 2048; i *= 2) {
        Multiply_directly(i);
        Multiply_breakdown(i);
    }

}

#endif