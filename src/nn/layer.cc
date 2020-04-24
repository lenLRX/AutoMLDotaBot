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

torch::Tensor Layer::get_masked_reward(const std::vector<float>& reward) {
    int tick_num = ticks.size();
    torch::Tensor ret = torch::zeros({tick_num});
    for (int i = 0; i < tick_num; ++i) {
        ret[i] = reward.at(ticks[i]);
    }
    ret.to(torch::kCUDA);
    return ret;
}

void Layer::update_params(Layer& other) {
    std::lock_guard<std::mutex> g(other.mtx);
    for (const auto& p_other: other.networks) {
        auto& p_net = networks.at(p_other.first);
        p_net = p_other.second->clone();
        p_net->eval();
        p_net->to(torch::kCPU);
        //cloned->to(torch::kCPU);
    }
}

void Layer::reset() {
    std::cerr << std::this_thread::get_id() << " Layer " << get_name() << " reset " << std::endl;
    states.clear();
    ticks.clear();
    expert_mode = false;
    reset_custom();

    for (auto c:children) {
        c->reset();
    }
}

NS_NN_END