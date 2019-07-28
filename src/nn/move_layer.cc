//
// Created by len on 7/7/19.
//

#include "nn/move_layer.h"
#include "util/util.h"

#include <exception>

NS_NN_BEGIN

const static int input_shape = 10;
const static int hidden_shape = 128;
const static int output_shape = 2;

MoveLayer::MoveLayer() {
    networks["actor"] = std::make_shared<Dense>(input_shape, hidden_shape, output_shape);
    networks["critic"] = std::make_shared<Dense>(input_shape, hidden_shape, 1);
    networks["discriminator"] = std::make_shared<Dense>(input_shape + output_shape, hidden_shape, 1);
}

static std::pair<float, float> get_move_vec(torch::Tensor x) {
    x = x * 300;
    std::pair<float, float> out_(dotautil::to_number<float>(x[0]), dotautil::to_number<float>(x[1]));
    return out_;
}

std::shared_ptr<Layer> MoveLayer::forward_impl(const LayerForwardConfig &cfg) {
    CMsgBotWorldState_Unit hero = dotautil::get_hero(cfg.state, cfg.team_id, cfg.player_id);

    const auto& location = hero.location();
    torch::Tensor x = dotautil::state_encoding(cfg.state, cfg.team_id, cfg.player_id);

    auto out = networks.at("actor")->forward(x);

    // range (-1,1)
    auto action = torch::tanh(out);

    if (states.size() == 1) {
        auto action_logger = spdlog::get("action_logger");
        action_logger->info("{} first tick\nmy action\n{}\n-------",
                            get_name(), dotautil::torch_to_string(action));
    }

    target_pos = get_move_vec(action);
    target_pos.first += hero.location().x();
    target_pos.second += hero.location().y();

    return std::shared_ptr<Layer>();
}

CMsgBotWorldState_Action MoveLayer::get_action(){
    CMsgBotWorldState_Action ret;
    ret.set_actiontype(CMsgBotWorldState_Action_Type_DOTA_UNIT_ORDER_MOVE_TO_POSITION);
    auto* target_loc = ret.mutable_movetolocation()->mutable_location();

    target_loc->set_z(0.0);

    target_loc->set_x(target_pos.first);
    target_loc->set_y(target_pos.second);

    return ret;
}


std::shared_ptr<Layer> MoveLayer::forward_expert(const LayerForwardConfig &cfg) {
    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                            cfg.team_id, cfg.player_id);
    const auto& location = hero.location();

    std::vector<torch::Tensor> states_cache;
    std::vector<torch::Tensor> expert_action_cache;
    std::vector<torch::Tensor> actor_action_cache;

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 400);

    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());

    auto enemy_creeps = dotautil::filter_units_by_team(nearby_units, opposed_team);
    enemy_creeps = dotautil::filter_attackable_units(enemy_creeps);
    if (enemy_creeps.empty()) {
        target_pos.first = -500;
        target_pos.second = -500;
    }
    else {
        if (cfg.team_id == DOTA_TEAM_DIRE) {
            target_pos.first = 300;
            target_pos.second = 300;
        }
        else {
            target_pos.first = -1300;
            target_pos.second = -1300;
        }
    }

    torch::Tensor expert = torch::zeros({2});
    expert[0] = target_pos.first - hero.location().x();
    expert[1] = target_pos.second - hero.location().y();

    expert_action.push_back(expert);

    return std::shared_ptr<Layer>();
}

std::shared_ptr<Layer> MoveLayer::forward(const LayerForwardConfig &cfg) {
    ticks.push_back(cfg.tick);
    save_state(cfg);
    if (cfg.expert_action) {
        return forward_expert(cfg);
    }
    return forward_impl(cfg);
}


void MoveLayer::save_state(const LayerForwardConfig &cfg) {
    torch::Tensor x = dotautil::state_encoding(cfg.state, cfg.team_id, cfg.player_id);
    states.push_back(x);
}

MoveLayer::PackedData MoveLayer::get_training_data() {
    PackedData ret;
    ret["state"] = torch::stack(states);
    ret["expert_action"] = torch::stack(expert_action);
    return ret;
}

void MoveLayer::train(PackedData& data){

}


NS_NN_END
