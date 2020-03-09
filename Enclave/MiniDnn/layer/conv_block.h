//
// Created by Fengting Li on 2019-07-30.
//

#ifndef MYPROJECT_CONV_BLOCK_H
#define MYPROJECT_CONV_BLOCK_H

#include <vector>
#include "../layer.h"
#include "conv.h"
#include "relu.h"

class ConvBlock : public Layer {
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
    int strides;


    std::vector<Layer *> layers;  // layer pointers
    std::vector<Layer *> shortcut;  // layer pointers
    void init();

public:
    ConvBlock(int channel_in, int height_in, int width_in, int channel_out1, int channel_out2, int channel_out,
              int height_kernel, int width_kernel, int strides) :
            dim_in(channel_in * height_in * width_in), dim_out(channel_out * height_in * width_in),
            channel_in(channel_in), height_in(height_in), width_in(width_in), channel_out1(channel_out1),
            channel_out2(channel_out2), channel_out(channel_out), height_kernel(height_kernel),
            width_kernel(width_kernel), strides(strides) { init(); }

    void forward(const Matrix &bottom);

    void backward(const Matrix &bottom, const Matrix &grad_top);

    void update(Optimizer &opt);

    int output_dim() { return dim_out; }

    const Matrix &output() { return layers[layers.size() - 1]->output(); }

//    const Matrix &back_gradient() { return layers[layers.size() - 1]->back_gradient() + shortcut[0]->back_gradient(); }


};

#endif //MYPROJECT_CONV_BLOCK_H
