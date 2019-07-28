#include "nn/highlevel_decision.h"
#include "nn/torch_layer.h"
#include "util/util.h"

NS_NN_BEGIN


const static int input_shape = 10;
const static int hidden_shape = 128;
const static int output_shape = 2;


HighLevelDecision::HighLevelDecision() {
    networks["actor"] = std::make_shared<Dense>(input_shape, hidden_shape, output_shape);
    networks["critic"] = std::make_shared<Dense>(input_shape, hidden_shape, 1);
    networks["discriminator"] = std::make_shared<Dense>(input_shape + output_shape, hidden_shape, 1);
}

std::shared_ptr<Layer> HighLevelDecision::forward_impl(const LayerForwardConfig &cfg) {
    CMsgBotWorldState_Unit hero = dotautil::get_hero(cfg.state, cfg.team_id, cfg.player_id);

    const auto& location = hero.location();
    torch::Tensor x = dotautil::state_encoding(cfg.state, cfg.team_id, cfg.player_id);

    auto out = networks.at("actor")->forward(x);

    auto action_prob = torch::softmax(out, 0);
    float max_prob = dotautil::to_number<float>(torch::max(action_prob));

    auto action = action_prob.multinomial(1);

    int idx = dotautil::to_number<int>(action);

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 2000);
    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());

    dotautil::Units nearby_enemy = dotautil::filter_units_by_team(nearby_units, opposed_team);
    nearby_enemy = dotautil::filter_attackable_units(nearby_enemy);
    if (nearby_enemy.empty()) {
        idx = 0;
    }

    auto ret_layer = children.at(idx);
    ret_layer->forward(cfg);
    return ret_layer;
}


std::shared_ptr<Layer> HighLevelDecision::forward_expert(const LayerForwardConfig& cfg) {
    CMsgBotWorldState_Unit hero = dotautil::get_hero(cfg.state, cfg.team_id, cfg.player_id);

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 2000);
    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());

    int hero_atk = hero.attack_damage();

    int idx = 0;

    dotautil::Units nearby_enemy = dotautil::filter_units_by_team(nearby_units, opposed_team);
    nearby_enemy = dotautil::filter_attackable_units(nearby_enemy);
    for (const auto& creep:nearby_enemy) {
        if (creep.health() < hero_atk * 1.2) {
            idx = 1;
        }
    }

    torch::Tensor expert = torch::zeros({2});
    expert[idx] = 1;
    expert_action.push_back(expert);

    auto ret_layer = children.at(idx);
    ret_layer->forward(cfg);
    return ret_layer;
}

std::shared_ptr<Layer> HighLevelDecision::forward(const LayerForwardConfig &cfg) {
    ticks.push_back(cfg.tick);
    save_state(cfg);
    if (cfg.expert_action) {
        return forward_expert(cfg);
    }
    return forward_impl(cfg);
}

void HighLevelDecision::save_state(const LayerForwardConfig &cfg) {
    torch::Tensor x = dotautil::state_encoding(cfg.state, cfg.team_id, cfg.player_id);
    states.push_back(x);
}

HighLevelDecision::PackedData HighLevelDecision::get_training_data(){
    PackedData ret;
    ret["state"] = torch::stack(states);
    ret["expert_action"] = torch::stack(expert_action);
    return ret;
}

void HighLevelDecision::train(PackedData& data){

}

NS_NN_END
