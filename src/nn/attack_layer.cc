//
// Created by len on 7/15/19.
//
#include "nn/attack_layer.h"

NS_NN_BEGIN

const static int input_shape = 4;
const static int hidden_shape = 128;
const static int output_shape = 1;

AttackLayer::AttackLayer():atk_handle(-1) {
    networks["actor"] = std::make_shared<Dense>(input_shape, hidden_shape, output_shape);
    networks["critic"] = std::make_shared<Dense>(input_shape, hidden_shape, 1);
    networks["discriminator"] = std::make_shared<Dense>(input_shape + output_shape, hidden_shape, 1);
}


static torch::Tensor creep_encoding(const CMsgBotWorldState_Unit& h,
        const CMsgBotWorldState_Unit& s) {
    torch::Tensor x = torch::zeros({ 4 });
    const auto& hloc = h.location();
    const auto& sloc = s.location();
    x[0] = (sloc.x() - hloc.x()) / near_by_scale;
    x[1] = (sloc.y() - hloc.y()) / near_by_scale;
    x[2] = s.health();
    x[3] = h.attack_damage();
    return x;
}


std::shared_ptr<Layer> AttackLayer::forward_impl(const LayerForwardConfig &cfg) {
    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
            cfg.team_id, cfg.player_id);

    const auto& location = hero.location();

    std::vector<torch::Tensor> states_cache;
    std::vector<torch::Tensor> expert_action_cache;
    std::vector<torch::Tensor> actor_action_cache;

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 2000);

    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());

    auto enemy_creeps = dotautil::filter_units_by_team(nearby_units, opposed_team);
    enemy_creeps = dotautil::filter_attackable_units(enemy_creeps);

    float max_value = 0.0;
    int idx = -1;
    int i = 0;

    for (const auto& creep:enemy_creeps) {
        auto x = creep_encoding(hero, creep);
        states_cache.push_back(x);
        auto out = sigmoid(networks.at("actor")->forward(x));
        actor_action_cache.push_back(out);
        auto out_value = dotautil::to_number<float>(out);

        if (out_value > max_value) {
            max_value = out_value;
            idx = i;
        }
        i++;
    }

    if (idx < 0) {
        auto action_logger = spdlog::get("action_logger");
        action_logger->critical("{} attack layer found no creep, invalid idx {}",
                            get_name(), idx);
        throw std::exception();
    }



    atk_handle = enemy_creeps.at(idx).handle();
    return std::shared_ptr<Layer>();
}

std::shared_ptr<Layer> AttackLayer::forward_expert(const LayerForwardConfig &cfg) {
    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                            cfg.team_id, cfg.player_id);

    const auto& location = hero.location();

    std::vector<torch::Tensor> states_cache;
    std::vector<torch::Tensor> expert_action_cache;
    std::vector<torch::Tensor> actor_action_cache;

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 2000);

    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());

    auto enemy_creeps = dotautil::filter_units_by_team(nearby_units, opposed_team);
    enemy_creeps = dotautil::filter_attackable_units(enemy_creeps);


    int hero_atk = hero.attack_damage();

    atk_handle = -1;

    int i = 0;
    for (const auto& creep:enemy_creeps) {
        if (creep.health() < hero_atk * 1.2) {
            atk_handle = creep.handle();
            expert_action.push_back(1);
        }
        else {
            expert_action.push_back(0);
        }
        if (atk_handle < 0) {
            atk_handle = creep.handle();
        }
        ++i;
    }

    return std::shared_ptr<Layer>();
}

std::shared_ptr<Layer> AttackLayer::forward(const LayerForwardConfig &cfg) {
    ticks.push_back(cfg.tick);
    save_state(cfg);
    if (cfg.expert_action) {
        return forward_expert(cfg);
    }
    return forward_impl(cfg);
}


CMsgBotWorldState_Action AttackLayer::get_action() {
    CMsgBotWorldState_Action ret;
    ret.set_actiontype(CMsgBotWorldState_Action_Type_DOTA_UNIT_ORDER_ATTACK_TARGET);
    ret.mutable_attacktarget()->set_target(atk_handle);
    ret.mutable_attacktarget()->set_once(false);
    return ret;
}

void AttackLayer::save_state(const LayerForwardConfig &cfg) {
    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                            cfg.team_id, cfg.player_id);
    const auto& location = hero.location();

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 2000);

    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());

    auto enemy_creeps = dotautil::filter_units_by_team(nearby_units, opposed_team);
    enemy_creeps = dotautil::filter_attackable_units(enemy_creeps);

    state_len.push_back(enemy_creeps.size());

    for (const auto& creep:enemy_creeps) {
        states.push_back(creep_encoding(hero, creep));
    }
}


AttackLayer::PackedData AttackLayer::get_training_data() {
    PackedData ret;
    ret["state"] = torch::stack(states);
    ret["expert_action"] = dotautil::vector2tensor(expert_action);
    return ret;
}

void AttackLayer::train(PackedData& data){

}



NS_NN_END