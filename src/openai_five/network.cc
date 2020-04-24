//
// Created by len on 12/16/19.
//

#include "openai_five/network.h"

int CreepInputLayer::hidden_dim = 256;

int MainLSTMLayer::input_dim = 128 * 2 + 32;
int MainLSTMLayer::hidden_dim = 1024;

int ActionSelectionLayer::action_input_dim = 2;
int ActionSelectionLayer::lstm_input_dim = 1024;
int ActionSelectionLayer::hidden_dim = 128;

int FCSoftmax::lstm_input_dim = 1024;

int UnitSelectionLayer::action_input_dim = 2;
int UnitSelectionLayer::lstm_input_dim = 1024;
int UnitSelectionLayer::hidden_dim = 256;

int Network::move_layer_extend = 9;

Network::Network(int player_id, DOTA_TEAM team_id)
  : player_id_(player_id), team_id_(team_id) {
    state_manager_ = std::make_shared<StateManager>(player_id, team_id);
    ally_creep_input_layer = std::make_shared<CreepInputLayer>();
    enemy_creep_input_layer = std::make_shared<CreepInputLayer>();
    main_lstm_layer = std::make_shared<MainLSTMLayer>();
    action_selection_layer = std::make_shared<ActionSelectionLayer>();
    move_x_layer = std::make_shared<FCSoftmax>(move_layer_extend);
    move_y_layer = std::make_shared<FCSoftmax>(move_layer_extend);
    unit_selection_layer = std::make_shared<UnitSelectionLayer>();
}

void Network::forward(const CMsgBotWorldState &state) {
    state_manager_->update(state);

    torch::Tensor ally_creep_dense = ally_creep_input_layer->forward(
            state_manager_->get_ally_unit_state());
    torch::Tensor enemy_creep_dense = enemy_creep_input_layer->forward(
            state_manager_->get_enemy_unit_state());
    torch::Tensor ally_creep_dense_max = std::get<0>(torch::max(ally_creep_dense, -2));
    torch::Tensor enemy_creep_dense_max = std::get<0>(torch::max(enemy_creep_dense, -2));
    torch::Tensor main_lstm = main_lstm_layer->forward(state_manager_->get_hero_state(),
                                                      ally_creep_dense_max, enemy_creep_dense_max);
    torch::Tensor available_action = torch::zeros({2}, torch::ScalarType::Long);
    torch::Tensor action_out = action_selection_layer->forward(main_lstm, available_action);

    torch::Tensor t_action_idx = torch::argmax(action_out).view({-1});
    int action_idx = t_action_idx.item().toInt();

    std::cerr << "OPENAI: action_idx " << action_idx << std::endl;

    if (action_idx == 0) {
        // move
        torch::Tensor move_x_layer_out = move_x_layer->forward(main_lstm);
        int move_x = torch::argmax(move_x_layer_out).view({-1}).item().toInt();
        torch::Tensor move_y_layer_out = move_x_layer->forward(main_lstm);
        int move_y = torch::argmax(move_y_layer_out).view({-1}).item().toInt();
        std::cerr << "OPENAI: move_x " << move_x
            << " move_y " << move_y << std::endl;
    }
    else {
        // attack
        torch::Tensor unit_selection_out = unit_selection_layer->forward(main_lstm,
                t_action_idx,
                enemy_creep_dense_max);
        int unit_idx = torch::argmax(unit_selection_out).item().toInt();\
        auto handles = state_manager_->get_enemy_handle();
        std::cerr << "selected unit_idx: " << unit_idx
            << " handle: " << handles[unit_idx] << std::endl;
    }

}

void Network::set_player_id(int id) {
    player_id_ = id;
    state_manager_->set_player_id(id);
}

void Network::reset() {
    ally_creep_input_layer->reset();
    enemy_creep_input_layer->reset();
    main_lstm_layer->reset();
    action_selection_layer->reset();
    move_x_layer->reset();
    move_y_layer->reset();
    unit_selection_layer->reset();
}