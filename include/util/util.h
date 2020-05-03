//
// Created by len on 6/4/19.
//

#pragma once

#include <vector>
#include <exception>

#include "dotaservice/protos/DotaService.grpc.pb.h"
#include "dotaservice/protos/DotaService.pb.h"

#include <torch/torch.h>
#include "spdlog/spdlog.h"
#include "nlohmann/json.hpp"

#define NS_DOTACLIENT_BEGIN namespace dotaservice {
#define NS_DOTACLIENT_END }

#define NS_NN_BEGIN namespace nn {
#define NS_NN_END }


#define NS_UTIL_BEGIN namespace dotautil {
#define NS_UTIL_END }


static const float lr = 1e-4;

static const float near_by_scale = 2000;
static const int hero_state_size = 16;

static int n_step = 20;

enum DOTA_TEAM {
    DOTA_TEAM_RADIANT = 2,
    DOTA_TEAM_DIRE = 3,
};

NS_UTIL_BEGIN

typedef std::vector<CMsgBotWorldState_Unit> Units;

class ObserverState {
public:
    ObserverState(const CMsgBotWorldState& rad_state,
                  const CMsgBotWorldState& dire_state);
    Units units_;
    std::unordered_set<int> rad_visible_;
    std::unordered_set<int> dire_visibe_;
};


CMsgBotWorldState_Unit get_hero(const CMsgBotWorldState& state, DOTA_TEAM team, int player_id);

bool has_hero(const CMsgBotWorldState& state, DOTA_TEAM team, int player_id);

CMsgBotWorldState_Action no_op(uint32_t player_id);

float get_unit_distance(const CMsgBotWorldState_Unit& unit1, const CMsgBotWorldState_Unit& unit2);

Units get_nearby_unit_by_type(const CMsgBotWorldState& state,
        const CMsgBotWorldState_Unit& unit, CMsgBotWorldState_UnitType type, float distance);

Units get_nearby_unit(const CMsgBotWorldState& state,
        const CMsgBotWorldState_Unit& unit, float distance);

Units filter_units_by_team(const Units& units, uint32_t team_id);

Units filter_units_by_type(const Units& units, CMsgBotWorldState_UnitType type);

CMsgBotWorldState_Unit get_nearest_unit(const Units& units, const CMsgBotWorldState_Unit& target);

Units filter_attackable_units(const Units& units);

uint32_t get_opposed_team(uint32_t team_id);

template <typename T>
std::string torch_to_string(const T& data) {
    std::stringstream ss;
    ss << data;
    return ss.str();
}


template<typename T>
T to_number(torch::Tensor x) {
    return x.item().to<T>();
}

// maybe we need a faster version with torch::from_blob
// but we have to pay attention to ownership of data
template <typename T>
torch::Tensor vector2tensor(const std::vector<T>& vec) {
    int vec_size = vec.size();
    torch::Tensor ret = torch::zeros({vec_size});
    for (int i = 0;i < vec_size; ++i) {
        ret[i] = vec[i];
    }
    return ret;
}

int get_total_ability_level(const CMsgBotWorldState_Unit& hero);

torch::Tensor state_encoding(const CMsgBotWorldState& state, DOTA_TEAM team_id, int player_id);

class AbilityManager {
public:
    static AbilityManager& get_instance();
    int get_id_by_name(const char* name);
    void set_data(const nlohmann::json& data);
private:
    AbilityManager() = default;
    nlohmann::json data;
};

const static char* reward_hp_key = "reward_hp";
const static char* reward_lasthit_key = "reward_lasthit";

static int max_unit_num = 64;
static int max_unit_field = 32;

class StateManager
{
public:
    explicit StateManager(int player_id, DOTA_TEAM team_id):
        player_id_(player_id), team_id_(team_id) {}
    void set_player_id(int id);
    void update(const CMsgBotWorldState& state);
    torch::Tensor get_ally_unit_state();
    torch::Tensor get_enemy_unit_state();
    std::vector<int> get_enemy_handle();
    torch::Tensor get_hero_state();
    int player_id_;
    DOTA_TEAM team_id_;
private:

    void update_unit(torch::Tensor& t, const CMsgBotWorldState_Unit& state,
                     const CMsgBotWorldState_Unit& self);
    torch::Tensor update_hero(const CMsgBotWorldState_Unit& self);

    void update_location(const CMsgBotWorldState_Unit& unit, torch::Tensor& t, int& i);
    void update_facing_angle(const CMsgBotWorldState_Unit& unit, torch::Tensor& t, int& i);
    void update_hp(const CMsgBotWorldState_Unit& unit, torch::Tensor& t, int& i);
    void update_armor(const CMsgBotWorldState_Unit& unit, torch::Tensor& t, int& i);
    void update_vector_to_me(const CMsgBotWorldState_Unit& unit, const CMsgBotWorldState_Unit& me,
                             torch::Tensor& t, int& i);

    void update_reward(const CMsgBotWorldState& state);

    std::unordered_map<int, std::deque<float>> hp_buffer_;
    std::vector<torch::Tensor> ally_unit_states_;
    std::vector<torch::Tensor> enemy_unit_states_;
    std::vector<int> enemy_handle_;
    std::vector<torch::Tensor> hero_states_;
    std::vector<float> rewards_;
};

NS_UTIL_END
