//
// Created by Fengting Li on 2019-07-13.
//

#ifndef SRC_LAYER_LAYER_BLOCK_H
#define SRC_LAYER_LAYER_BLOCK_H

#include "../layer.h"
//#include "../network.h"

class Layer_Block : Layer {
private:
    Matrix top;  // Block output
    std::vector<Layer *> layers;  // layer pointers
//    Loss *loss;  // loss pointer

    const int dim_in;
    int dim_out;

    int channel_in;
    int height_in;
    int width_in;
    int channel_out;
    int height_kernel;
    int width_kernel;
    int stride;
    int pad_h;
    int pad_w;

    int height_out;
    int width_out;

    Matrix weight;  // weight param, size=channel_in*h_kernel*w_kernel*channel_out
    Vector bias;  // bias param, size = channel_out
    Matrix grad_weight;  // gradient w.r.t weight
    Vector grad_bias;  // gradient w.r.t bias

    std::vector<Matrix> data_cols;

    void init();

public:
    Layer_Block(int channel_in, int height_in, int width_in, int channel_out,
                int height_kernel, int width_kernel, int stride = 1, int pad_w = 0,
                int pad_h = 0) :
            dim_in(channel_in * height_in * width_in),
            channel_in(channel_in), height_in(height_in), width_in(width_in),
            channel_out(channel_out), height_kernel(height_kernel),
            width_kernel(width_kernel), stride(stride), pad_w(pad_w), pad_h(pad_h) { init(); }

    ~Layer_Block() {
        for (int i = 0; i < layers.size(); i++) {
            delete layers[i];
        }
    }

    void add_layer(Layer *layer) { layers.push_back(layer); }

    void forward(const Matrix &input);  //

    void backward(const Matrix &bottom, const Matrix &grad_top); //

    void update(Optimizer &opt);  //
};


#endif //SRC_LAYER_LAYER_BLOCK_H
