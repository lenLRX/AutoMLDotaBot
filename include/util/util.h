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

#define NS_DOTACLIENT_BEGIN namespace dotaservice {
#define NS_DOTACLIENT_END }

#define NS_NN_BEGIN namespace nn {
#define NS_NN_END }


#define NS_UTIL_BEGIN namespace dotautil {
#define NS_UTIL_END }


static const float lr = 1e-3;

static const float near_by_scale = 2000;

enum DOTA_TEAM {
    DOTA_TEAM_RADIANT = 2,
    DOTA_TEAM_DIRE = 3,
};

NS_UTIL_BEGIN

typedef std::vector<CMsgBotWorldState_Unit> Units;


CMsgBotWorldState_Unit get_hero(const CMsgBotWorldState& state, DOTA_TEAM team, int player_id);

bool has_hero(const CMsgBotWorldState& state, DOTA_TEAM team, int player_id);

CMsgBotWorldState_Action no_op(uint32_t player_id);

float get_unit_distance(const CMsgBotWorldState_Unit& unit1, const CMsgBotWorldState_Unit& unit2);

Units get_nearby_unit_by_type(const CMsgBotWorldState& state,
        const CMsgBotWorldState_Unit& unit, CMsgBotWorldState_UnitType type, float distance);

Units get_nearby_unit(const CMsgBotWorldState& state,
        const CMsgBotWorldState_Unit& unit, float distance);

Units filter_units_by_team(const Units& units, uint32_t team_id);

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

torch::Tensor state_encoding(const CMsgBotWorldState& state, DOTA_TEAM team_id, int player_id);

NS_UTIL_END
