//
// Created by Fengting Li on 2019-07-13.
//

#ifndef SRC_NETWORK_RESNET_H
#define SRC_NETWORK_RESNET_H

#include "../network.h"
#include "../la"


class ResNet : public Network {
private:

public:
    void forward(const Matrix& input);
};


#endif //SRC_NETWORK_RESNET_H
