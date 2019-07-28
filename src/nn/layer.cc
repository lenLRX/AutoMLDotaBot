//
// Created by len on 6/6/19.
//

#include "nn/layer.h"

NS_NN_BEGIN

CMsgBotWorldState_Action Layer::get_action() {
    throw std::runtime_error("Layer::get_action not implemented");
}

std::shared_ptr<Layer> Layer::forward_expert(const LayerForwardConfig& state) {
    throw std::runtime_error("Layer::forward_expert not implemented");
}

NS_NN_END