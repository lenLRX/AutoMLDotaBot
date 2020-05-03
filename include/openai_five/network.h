//
// Created by len on 12/16/19.
//

#ifndef AUTOMLDOTABOT_NETWORK_H
#define AUTOMLDOTABOT_NETWORK_H

#include "util/util.h"

using namespace dotautil;

class CreepInputLayer :public torch::nn::Cloneable<CreepInputLayer>
{
public:
    static int hidden_dim;
    virtual ~CreepInputLayer() {}
    CreepInputLayer():
            fc1(nullptr), fc2(nullptr), fc3(nullptr), fc4(nullptr)
    {
    }

    torch::Tensor forward(const torch::Tensor& state) {
        torch::Tensor o = torch::relu(fc1->forward(state));
        o = torch::relu(fc2->forward(o));
        o = torch::relu(fc3->forward(o));
        o = fc4->forward(o);
        return o;
    }

    void reset() override {
        fc1 = torch::nn::Linear(max_unit_field, hidden_dim);
        fc2 = torch::nn::Linear(hidden_dim, hidden_dim);
        fc3 = torch::nn::Linear(hidden_dim, hidden_dim);
        fc4 = torch::nn::Linear(hidden_dim, hidden_dim);
        register_module("fc_1", fc1);
        register_module("fc_2", fc2);
        register_module("fc_3", fc3);
        register_module("fc_4", fc4);

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

    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
    torch::nn::Linear fc4;
};

class MainLSTMLayer :public torch::nn::Cloneable<MainLSTMLayer>
{
public:
    static int input_dim;
    static int hidden_dim;
    virtual ~MainLSTMLayer() {}
    MainLSTMLayer():
            fc1(nullptr), lstm2(nullptr)
    {
    }

    torch::Tensor forward(const torch::Tensor& hero_state,
                          const torch::Tensor& ally_creep_state,
                          const torch::Tensor& enemy_creep_state) {
        torch::Tensor input_tensor = torch::cat({hero_state,
                                                 ally_creep_state,
                                                 enemy_creep_state}, -1);
        torch::Tensor o = torch::relu(fc1->forward(input_tensor));
        o = o.reshape({1,1,-1});
        auto rnn_out = std::get<0>(lstm2->forward(o));
        return rnn_out;
    }

    void reset() override {
        fc1 = torch::nn::Linear(512 + max_unit_field, hidden_dim);
        lstm2 = torch::nn::LSTM(hidden_dim, hidden_dim);
        register_module("fc_1", fc1);
        register_module("lstm_2", lstm2);

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

    torch::nn::Linear fc1;
    torch::nn::LSTM lstm2;
};


class ActionSelectionLayer :public torch::nn::Cloneable<ActionSelectionLayer>
{
public:
    // move attack
    static int action_input_dim;
    static int lstm_input_dim;
    static int hidden_dim;
    virtual ~ActionSelectionLayer() {}
    ActionSelectionLayer():
            embedding1(nullptr), fc2(nullptr)
    {
    }

    torch::Tensor forward(const torch::Tensor& lstm_input,
                          const torch::Tensor& available_action) {
        // embedding input: [batch, n_action]
        // embedding output: [batch, n_action, hidden_dim]
        auto embedding_out = embedding1->forward(available_action);
        // fc_o dim: [batch, 1, hidden_dim]
        torch::Tensor fc_o = fc2->forward(lstm_input).view({-1,1, hidden_dim});
        // hidden_o dim: [batch, n_action, 1]
        // since there is not batch dot we use (x*y).sum(-1)
        torch::Tensor hidden_o = (embedding_out * fc_o).sum(-1).squeeze();
        torch::Tensor softmax_o = torch::softmax(hidden_o, -1);
        return softmax_o;
    }

    void reset() override {
        // Todo update pytorch to support padding idx
        embedding1 = torch::nn::Embedding(action_input_dim, hidden_dim);
        fc2 = torch::nn::Linear(lstm_input_dim, hidden_dim);
        register_module("embedding_1", embedding1);
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

    torch::nn::Embedding embedding1;
    torch::nn::Linear fc2;
};

class FCSoftmax: public torch::nn::Cloneable<FCSoftmax> {
public:
    static int lstm_input_dim;
    virtual ~FCSoftmax() {}
    explicit FCSoftmax(int dim):
        output_dim(dim), fc(nullptr)
    {
    }

    torch::Tensor forward(const torch::Tensor& lstm_input) {
        torch::Tensor fc_o = fc->forward(lstm_input);
        torch::Tensor softmax_o = torch::softmax(fc_o, -1);
        return softmax_o;
    }

    void reset() override {
        // Todo update pytorch to support padding idx
        fc = torch::nn::Linear(lstm_input_dim, output_dim);
        register_module("fc", fc);

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
    int output_dim;
    torch::nn::Linear fc;
};

class UnitSelectionLayer: public torch::nn::Cloneable<UnitSelectionLayer>
{
public:
    // move attack
    static int action_input_dim;
    static int lstm_input_dim;
    static int hidden_dim;
    virtual ~UnitSelectionLayer() {}
    UnitSelectionLayer():
            embedding1(nullptr), fc2(nullptr)
    {
    }

    torch::Tensor forward(const torch::Tensor& lstm_input,
                          const torch::Tensor& chosen_action,
                          const torch::Tensor& unit_embedding) {
        // unit_embedding : [batch, creep_num, hidden_dim]
        // embedding input: [batch]
        // embedding output: [batch, hidden_dim]
        auto embedding_out = torch::sigmoid(embedding1->forward(chosen_action));
        // hidden : [batch, creep_num, hidden_dim]
        torch::Tensor hidden = embedding_out * unit_embedding;
        // fc_o dim: [batch, 1, hidden_dim]
        torch::Tensor fc_o = fc2->forward(lstm_input).view({-1,1, hidden_dim});
        // hidden_o dim: [batch, creep_num, 1]
        // since there is not batch dot we use (x*y).sum(-1)
        torch::Tensor hidden_o = (hidden * fc_o).sum(-1).squeeze();
        torch::Tensor softmax_o = torch::softmax(hidden_o, -1);
        return softmax_o;
    }

    void reset() override {
        // Todo update pytorch to support padding idx
        embedding1 = torch::nn::Embedding(action_input_dim, hidden_dim);
        fc2 = torch::nn::Linear(lstm_input_dim, hidden_dim);
        register_module("embedding_1", embedding1);
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

    torch::nn::Embedding embedding1;
    torch::nn::Linear fc2;
};

class Network: public torch::nn::Cloneable<Network> {
public:
    using Ptr = std::shared_ptr<Network>;
    static int move_layer_extend;
    Network(int player_id, DOTA_TEAM team_id);
    void forward(const CMsgBotWorldState& state);
    void set_player_id(int id);
    void reset() override;
    std::shared_ptr<StateManager> state_manager_;
    std::shared_ptr<CreepInputLayer> ally_creep_input_layer;
    std::shared_ptr<CreepInputLayer> enemy_creep_input_layer;
    std::shared_ptr<MainLSTMLayer> main_lstm_layer;
    std::shared_ptr<ActionSelectionLayer> action_selection_layer;
    std::shared_ptr<FCSoftmax> move_x_layer;
    std::shared_ptr<FCSoftmax> move_y_layer;

    std::shared_ptr<UnitSelectionLayer> unit_selection_layer;
    int player_id_;
    int team_id_;
};

#endif //AUTOMLDOTABOT_NETWORK_H
