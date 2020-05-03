//
// Created by len on 6/6/19.
//

#ifndef AUTOMLDOTABOT_TORCH_LAYER_H
#define AUTOMLDOTABOT_TORCH_LAYER_H

#include "util/util.h"

NS_NN_BEGIN

enum LayerType {
    TYPE_NONE=0,
    TYPE_Dense,
    TYPE_LSTM,
    TYPE_LSTM_DENSE,
};

class TorchLayer
{
public:
    explicit TorchLayer(LayerType t): type_(t) {}

    virtual ~TorchLayer() {}

    virtual void to(torch::DeviceType device) = 0;
    virtual void eval() = 0;
    virtual std::shared_ptr<TorchLayer> clone() = 0;

    virtual void* get_ptr() = 0;

    template <typename T>
    T* get();

    LayerType type_;
};


class Dense :public torch::nn::Cloneable<Dense>, public TorchLayer
{
public:
    virtual ~Dense() {}
    Dense(int state_dim, int hidden_dim, int action_dim):
    TorchLayer(TYPE_Dense),
    state_dim(state_dim), action_dim(action_dim), hidden_dim(hidden_dim),
    fc1(nullptr), fc2(nullptr)
    {
        reset();
    }

    torch::Tensor forward(const torch::Tensor& state) {
        torch::Tensor o = torch::tanh(fc1->forward(state));
        return fc2->forward(o);
    }

    void reset() override {
        fc1 = torch::nn::Linear(state_dim, hidden_dim);
        fc2 = torch::nn::Linear(hidden_dim, action_dim);
        register_module("fc_1", fc1);
        register_module("fc_2", fc2);

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

    void to(torch::DeviceType device) override {
        torch::nn::Cloneable<Dense>::to(device);
    }

    void eval() override {
        torch::nn::Cloneable<Dense>::eval();
    }

    std::shared_ptr<TorchLayer> clone() override {
        auto ret = std::dynamic_pointer_cast<TorchLayer>(torch::nn::Cloneable<Dense>::clone());
        assert(ret);
        return ret;
    }

    void* get_ptr() override {
        return this;
    }

    int state_dim;
    int action_dim;
    int hidden_dim;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};


class LSTM: public torch::nn::Cloneable<LSTM>, public TorchLayer
{
public:
    virtual ~LSTM() {}
    LSTM(int state_dim, int hidden_dim, int action_dim):
        TorchLayer(TYPE_LSTM),
        state_dim(state_dim), action_dim(action_dim), hidden_dim(hidden_dim)
    {
        reset();
    }

    torch::Tensor forward(const torch::Tensor& state) {
        auto o = lstm->forward(state);
        return fc1->forward(std::get<0>(o));
    }

    void reset() override {

        lstm = torch::nn::LSTM(state_dim, hidden_dim);
        fc1 = torch::nn::Linear(hidden_dim, action_dim);

        register_module("lstm", lstm);
        register_module("fc_1", fc1);

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

    void to(torch::DeviceType device) override {
        torch::nn::Cloneable<LSTM>::to(device);
    }

    void eval() override {
        torch::nn::Cloneable<LSTM>::eval();
    }

    std::shared_ptr<TorchLayer> clone() override {
        auto ret = std::dynamic_pointer_cast<TorchLayer>(torch::nn::Cloneable<LSTM>::clone());
        assert(ret);
        return ret;
    }

    void* get_ptr() override {
        return this;
    }

    int state_dim;
    int hidden_dim;
    int action_dim;
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc1{nullptr};
};


    class LSTM_DENSE: public torch::nn::Cloneable<LSTM_DENSE>, public TorchLayer
    {
    public:
        virtual ~LSTM_DENSE() {}
        LSTM_DENSE(int lstm_state_dim, int dense_state_dim, int hidden_dim, int action_dim):
                TorchLayer(TYPE_LSTM_DENSE), lstm_state_dim(lstm_state_dim),
                dense_state_dim(dense_state_dim),action_dim(action_dim), hidden_dim(hidden_dim)
        {
            reset();
        }

        torch::Tensor forward(const torch::Tensor& lstm_input, const torch::Tensor& dense_input) {
            auto lstm_o = std::get<0>(lstm->forward(lstm_input));
            torch::Tensor dense_o = fc_state->forward(dense_input);
            torch::Tensor hidden = torch::cat({lstm_o[0], dense_o},-1);
            return fc1->forward(hidden);
        }

        void reset() override {

            lstm = torch::nn::LSTM(lstm_state_dim, hidden_dim);
            fc_state = torch::nn::Linear(dense_state_dim, hidden_dim);
            fc1 = torch::nn::Linear(hidden_dim*2, action_dim);

            register_module("lstm", lstm);
            register_module("fc_state", fc_state);
            register_module("fc_1", fc1);

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

        void to(torch::DeviceType device) override {
            torch::nn::Cloneable<LSTM_DENSE>::to(device);
        }

        void eval() override {
            torch::nn::Cloneable<LSTM_DENSE>::eval();
        }

        std::shared_ptr<TorchLayer> clone() override {
            auto ret = std::dynamic_pointer_cast<TorchLayer>(torch::nn::Cloneable<LSTM_DENSE>::clone());
            assert(ret);
            return ret;
        }

        void* get_ptr() override {
            return this;
        }

        int lstm_state_dim;
        int dense_state_dim;
        int hidden_dim;
        int action_dim;
        torch::nn::LSTM lstm{nullptr};
        torch::nn::Linear fc_state{nullptr};
        torch::nn::Linear fc1{nullptr};
    };


template<typename T>
T* TorchLayer::get() {
    switch (type_) {
        case TYPE_NONE:
            throw std::runtime_error("invalid type");
        case TYPE_Dense:
            return reinterpret_cast<T*>(get_ptr());
        case TYPE_LSTM:
            return reinterpret_cast<T*>(get_ptr());
        case TYPE_LSTM_DENSE:
            return reinterpret_cast<T*>(get_ptr());
    }
    return nullptr;
}


NS_NN_END

#endif //AUTOMLDOTABOT_TORCH_LAYER_H
