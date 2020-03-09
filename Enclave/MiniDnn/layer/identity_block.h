//
// Created by Fengting Li on 2019-07-30.
//

#ifndef MYPROJECT_IDENTITY_BLOCK_H
#define MYPROJECT_IDENTITY_BLOCK_H

#include <vector>
#include "../layer.h"
#include "conv.h"
#include "relu.h"

class IdentityBlock : public Layer {
private:
    const int dim_in;
    int dim_out;

    int channel_in;
    int height_in;
    int width_in;
    int channel_out1;
    int channel_out2;
    int channel_out;
    int height_kernel;
    int width_kernel;

    std::vector<Layer *> layers;  // layer pointers
//    std::vector<Layer *> shortcut;  // layer pointers
    void init();

public:
    IdentityBlock(int channel_in, int height_in, int width_in, int channel_out1, int channel_out2,
                  int height_kernel, int width_kernel) :
            dim_in(channel_in * height_in * width_in), dim_out(channel_in * height_in * width_in),
            channel_in(channel_in), height_in(height_in), width_in(width_in), channel_out1(channel_out1),
            channel_out2(channel_out2), channel_out(channel_in), height_kernel(height_kernel),
            width_kernel(width_kernel) { init(); }

    void forward(const Matrix &bottom);

    void backward(const Matrix &bottom, const Matrix &grad_top);

    void update(Optimizer &opt);

    int output_dim() { return dim_out; }

    const Matrix &output() { return layers[layers.size() - 1]->output(); }

//    const Matrix &back_gradient() {
//        Matrix temp1 = (layers[layers.size() - 1]->back_gradient()).eval();
//        Matrix temp2 = (layers[0]->back_gradient()).eval();
//        Matrix temp = temp1 + temp2;
//        return temp;
//    }


};


#endif //MYPROJECT_IDENTITY_BLOCK_H
