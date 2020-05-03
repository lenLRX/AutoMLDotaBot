//
// Created by len on 4/25/20.
//

#include <cmath>
#include <torch/jit.h>
#include "nn/solo_pred_reward.h"


NS_NN_BEGIN

static const int hero_dim = 25;
static const int creep_dim = 21;
static const int creep_batch_dim = 20;
static const float cell_size = 128;

SoloPredReward::SoloPredReward(const std::string &model_path) {
    model_ = torch::jit::load(model_path);
}

void SoloPredReward::reset(const dotautil::ObserverState &cfg) {
    const int lstm_hidden_size = 512;
    auto h_0 = torch::zeros({1,1, lstm_hidden_size});
    auto c_0 = torch::zeros({1,1, lstm_hidden_size});
    lstm_states_ = torch::ivalue::Tuple::create({h_0, c_0});
    last_prob_ = forward(cfg);
}

float SoloPredReward::get_reward(const dotautil::ObserverState &cfg) {
    float curr_prob = forward(cfg);
    float reward = curr_prob - last_prob_;
    last_prob_ = curr_prob;
    return reward;
}

std::vector<torch::jit::IValue> SoloPredReward::make_state_inputs(const dotautil::ObserverState &cfg) {
    torch::jit::IValue rad_hero_state = torch::zeros({1, 1, hero_dim});
    torch::jit::IValue dire_hero_state = torch::zeros({1, 1, hero_dim});
    torch::jit::IValue rad_creep_state;
    torch::jit::IValue dire_creep_state;

    torch::Tensor rad_creep_state_tensor = torch::zeros({creep_batch_dim, creep_dim});
    torch::Tensor dire_creep_state_tensor = torch::zeros({creep_batch_dim, creep_dim});

    int rad_index = 0;
    int dire_index = 0;

    for (auto& unit: cfg.units_) {
        if (unit.unit_type() == CMsgBotWorldState_UnitType_HERO) {
            auto hero_value = make_hero_value(unit, cfg);
            if (unit.team_id() == DOTA_TEAM_RADIANT) {
                rad_hero_state.toTensor()[0] = hero_value;
            } else {
                dire_hero_state.toTensor()[0] = hero_value;
            }
        } else if (unit.unit_type() == CMsgBotWorldState_UnitType_TOWER ||
                   unit.unit_type() == CMsgBotWorldState_UnitType_LANE_CREEP) {
            auto creep_value = make_creep_value(unit, cfg);
            if (unit.team_id() == DOTA_TEAM_RADIANT) {
                if (rad_index < creep_batch_dim) {
                    rad_creep_state_tensor[rad_index] = creep_value;
                    ++rad_index;
                }
            } else {
                if (dire_index < creep_batch_dim) {
                    dire_creep_state_tensor[dire_index] = creep_value;
                    ++dire_index;
                }
            }
        }
    }

    rad_creep_state = torch::unsqueeze(rad_creep_state_tensor, 0);
    dire_creep_state = torch::unsqueeze(dire_creep_state_tensor, 0);

    std::vector<torch::jit::IValue> tmp = {rad_hero_state, dire_hero_state, rad_creep_state, dire_creep_state};
    std::vector<torch::jit::IValue> ret;
    for (const auto& t: tmp) {
        ret.emplace_back(torch::unsqueeze(t.toTensor(), 0));
    }
    return ret;
}

torch::Tensor SoloPredReward::make_hero_value(const CMsgBotWorldState_Unit& unit,
                                              const dotautil::ObserverState &cfg) {
    torch::Tensor t = torch::zeros({hero_dim});
    float x = unit.location().x();
    float y = unit.location().y();
    float z = unit.location().z();
    float cell_x = x / cell_size;
    float cell_y = y / cell_size;
    float cell_z = z / cell_size;
    float vec_x = fmod(x, cell_x);
    float vec_y = fmod(y, cell_y);
    float vec_z = fmod(z, cell_z);
    float life_state = !unit.is_alive();
    float damage_max = unit.base_damage() + unit.base_damage_variance() / 2;
    float damage_min = unit.base_damage() - unit.base_damage_variance() / 2;
    float damage_bonus = unit.bonus_damage();
    float max_mana = unit.mana_max();
    float gold_bounty_max = unit.bounty_gold_max();
    float gold_bounty_min = unit.bounty_gold_min();
    float hp_regen = unit.health_regen();
    float hp_max = unit.health_max();
    float hp = unit.health();
    float xp_bounty = unit.bounty_xp();
    float is_rad = unit.team_id() == DOTA_TEAM_RADIANT;
    float is_dire = unit.team_id() == DOTA_TEAM_DIRE;
    float visible_by_rad = cfg.rad_visible_.count(unit.handle());
    float visible_by_dire = cfg.dire_visibe_.count(unit.handle());
    float current_level = unit.level();
    float strength = unit.strength();
    float agility = unit.agility();
    float intellect = unit.intelligence();

    t[0] = cell_x;
    t[1] = cell_y;
    t[2] = cell_z;
    t[3] = vec_x;
    t[4] = vec_y;
    t[5] = vec_z;
    t[6] = life_state;
    t[7] = damage_max;
    t[8] = damage_min;
    t[9] = damage_bonus;
    t[10] = max_mana;
    t[11] = gold_bounty_max;
    t[12] = gold_bounty_min;
    t[13] = hp_regen;
    t[14] = hp_max;
    t[15] = hp;
    t[16] = xp_bounty;
    t[17] = is_rad;
    t[18] = is_dire;
    t[19] = visible_by_rad;
    t[20] = visible_by_dire;
    t[21] = current_level;
    t[22] = strength;
    t[23] = agility;
    t[24] = intellect;

    return t;
}

torch::Tensor SoloPredReward::make_creep_value(const CMsgBotWorldState_Unit& unit,
                                               const dotautil::ObserverState &cfg) {
    torch::Tensor t = torch::zeros({creep_dim});
    float x = unit.location().x();
    float y = unit.location().y();
    float z = unit.location().z();
    float cell_x = x / cell_size;
    float cell_y = y / cell_size;
    float cell_z = z / cell_size;
    float vec_x = fmod(x, cell_x);
    float vec_y = fmod(y, cell_y);
    float vec_z = fmod(z, cell_z);
    float life_state = !unit.is_alive();
    float damage_max = unit.base_damage() + unit.base_damage_variance() / 2;
    float damage_min = unit.base_damage() - unit.base_damage_variance() / 2;
    float damage_bonus = unit.bonus_damage();
    float max_mana = unit.mana_max();
    float gold_bounty_max = unit.bounty_gold_max();
    float gold_bounty_min = unit.bounty_gold_min();
    float hp_regen = unit.health_regen();
    float hp_max = unit.health_max();
    float hp = unit.health();
    float xp_bounty = unit.bounty_xp();
    float is_rad = unit.team_id() == DOTA_TEAM_RADIANT;
    float is_dire = unit.team_id() == DOTA_TEAM_DIRE;
    float visible_by_rad = cfg.rad_visible_.count(unit.handle());
    float visible_by_dire = cfg.dire_visibe_.count(unit.handle());

    t[0] = cell_x;
    t[1] = cell_y;
    t[2] = cell_z;
    t[3] = vec_x;
    t[4] = vec_y;
    t[5] = vec_z;
    t[6] = life_state;
    t[7] = damage_max;
    t[8] = damage_min;
    t[9] = damage_bonus;
    t[10] = max_mana;
    t[11] = gold_bounty_max;
    t[12] = gold_bounty_min;
    t[13] = hp_regen;
    t[14] = hp_max;
    t[15] = hp;
    t[16] = xp_bounty;
    t[17] = is_rad;
    t[18] = is_dire;
    t[19] = visible_by_rad;
    t[20] = visible_by_dire;
    return t;
}

float SoloPredReward::forward(const dotautil::ObserverState &cfg) {
    auto state = make_state_inputs(cfg);
    if (!lstm_states_.isTuple()) {
        std::stringstream ss;
        ss << "ERROR: lstm state must be tuple";
        throw std::runtime_error(ss.str().c_str());
    }
    auto old_lstm_tup = lstm_states_.toTuple();
    state.push_back(old_lstm_tup->elements()[0]);
    state.push_back(old_lstm_tup->elements()[1]);

    torch::IValue curr_state;
    try {
        curr_state = model_.forward(state);
    }
    catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        throw;
    }
    if (!curr_state.isTuple()) {
        throw std::runtime_error("LSTM state must be tuple");
    }
    auto tup = curr_state.toTuple();
    int tup_size = tup->elements().size();
    if (tup_size != 2) {
        std::stringstream ss;
        ss << "ERROR: lstm state tup size " << tup_size << " expected 2";
        throw std::runtime_error(ss.str().c_str());
    }
    auto output = tup->elements()[0];
    lstm_states_ = tup->elements()[1];
    float win_prob = output.toTensor().view({-1}).item().toFloat();
    if (std::isnan(win_prob)) {
        throw std::runtime_error("found nan win prob");
    }

    return win_prob;
}

NS_NN_END
