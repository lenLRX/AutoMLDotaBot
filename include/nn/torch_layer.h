//
// Created by len on 6/6/19.
//

#ifndef AUTOMLDOTABOT_TORCH_LAYER_H
#define AUTOMLDOTABOT_TORCH_LAYER_H

#include "util/util.h"

NS_NN_BEGIN

class TorchLayer: public torch::nn::Cloneable<TorchLayer>
{
public:
    virtual ~TorchLayer() {}
    virtual torch::Tensor forward(const torch::Tensor& x) {}
    virtual void reset() {}
};

class Dense : public TorchLayer
{
public:
    virtual ~Dense() {}
    Dense(int state_dim, int hidden_dim, int action_dim):
    state_dim(state_dim), action_dim(action_dim), hidden_dim(hidden_dim), fc1(nullptr), fc2(nullptr)
    {
        reset();
        for (auto& param : named_parameters())
        {
            int dim_ = param.value().dim();
            if (dim_ == 2) {
                torch::nn::init::xavier_normal_(param.value());
            }
            else {
                torch::nn::init::constant_(param.value(), 0);
            }
        }
    }

    torch::Tensor forward(const torch::Tensor& state) override {
        torch::Tensor o = torch::tanh(fc1->forward(state));
        return fc2->forward(o);
    }

    void reset() override {
        fc1 = torch::nn::Linear(state_dim, hidden_dim);
        fc2 = torch::nn::Linear(hidden_dim, action_dim);
        register_module("fc_1", fc1);
        register_module("fc_2", fc2);
    }

    int state_dim;
    int action_dim;
    int hidden_dim;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};


NS_NN_END

#endif //AUTOMLDOTABOT_TORCH_LAYER_H
