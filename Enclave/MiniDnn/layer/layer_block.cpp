//
// Created by Fengting Li on 2019-07-13.
//

#include "layer_block.h"


void Layer_Block::forward(const Matrix &input) {
    if (layers.empty())
        return;
    layers[0]->forward(input);
    for (int i = 1; i < layers.size(); i++) {
        layers[i]->forward(layers[i - 1]->output());
    }
    top = layers[layers.size() - 1]->output() + input;
}

void Layer_Block::backward(const Matrix &bottom, const Matrix &grad_top) {
    int n_layer = layers.size();
    // 0 layer
    if (n_layer <= 0)
        return;
    // 1 layer
    if (n_layer == 1) {
        layers[0]->backward(bottom, grad_top);
        return;
    }
    // >1 layers
    layers[n_layer - 1]->backward(layers[n_layer - 2]->output(),
                                  grad_top);
    for (int i = n_layer - 2; i > 0; i--) {
        layers[i]->backward(layers[i - 1]->output(), layers[i + 1]->back_gradient());
    }
    layers[0]->backward(bottom, layers[1]->back_gradient());
}

void Layer_Block::update(Optimizer &opt) {
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->update(opt);
    }
}