//
// Created by len on 6/9/19.
//

#include "util/util.h"
#include <math.h>

NS_UTIL_BEGIN

CMsgBotWorldState_Unit get_hero(const CMsgBotWorldState& state, DOTA_TEAM team, int player_id)
{
    for (const auto& unit:state.units()) {
        if (unit.unit_type() == CMsgBotWorldState_UnitType_HERO
        && unit.player_id() == player_id && unit.team_id() == team) {
            return unit;
        }
    }
    throw std::exception();
}

bool has_hero(const CMsgBotWorldState& state, DOTA_TEAM team, int player_id) {
    for (const auto& unit:state.units()) {
        if (unit.unit_type() == CMsgBotWorldState_UnitType_HERO
            && unit.player_id() == player_id && unit.team_id() == team) {
            return true;
        }
    }
    return false;
}

CMsgBotWorldState_Action no_op(uint32_t player_id) {
    CMsgBotWorldState_Action ret;
    ret.set_actiontype(CMsgBotWorldState_Action_Type_DOTA_UNIT_ORDER_NONE);
    ret.set_player(player_id);
    return ret;
}

float get_unit_distance(const CMsgBotWorldState_Unit& unit1, const CMsgBotWorldState_Unit& unit2) {
    const auto& loc1 = unit1.location();
    const auto& loc2 = unit2.location();
    float xd = loc1.x() - loc2.x();
    float yd = loc1.y() - loc2.y();
    return sqrtf(xd * xd + yd * yd);
}

Units get_nearby_unit_by_type(const CMsgBotWorldState& state,
        const CMsgBotWorldState_Unit& base_unit, CMsgBotWorldState_UnitType type, float distance) {
    Units ret;
    for (const auto& unit:state.units()) {
        if (unit.unit_type() == type &&
            get_unit_distance(base_unit, unit) < distance) {
            ret.push_back(unit);
        }
    }
    return ret;
}

Units get_nearby_unit(const CMsgBotWorldState& state,
        const CMsgBotWorldState_Unit& base_unit, float distance) {
    Units ret;
    for (const auto& unit:state.units()) {
        if (get_unit_distance(base_unit, unit) < distance) {
            ret.push_back(unit);
        }
    }
    return ret;
}

Units filter_units_by_team(const Units& units, uint32_t team_id) {
    Units ret;
    for (const auto& unit:units) {
        if (unit.team_id() == team_id) {
            ret.push_back(unit);
        }
    }
    return ret;
}

Units filter_units_by_type(const Units& units, CMsgBotWorldState_UnitType type) {
    Units ret;
    for (const auto& unit:units) {
        if (unit.unit_type() == type) {
            ret.push_back(unit);
        }
    }
    return ret;
}

uint32_t get_opposed_team(uint32_t team_id)
{
    if (team_id == DOTA_TEAM_RADIANT) {
        return DOTA_TEAM_DIRE;
    }
    else if (team_id == DOTA_TEAM_DIRE) {
        return DOTA_TEAM_RADIANT;
    }
    throw std::exception();
}


torch::Tensor state_encoding(const CMsgBotWorldState& state,
        DOTA_TEAM team_id, int player_id) {
    CMsgBotWorldState_Unit hero = get_hero(state, team_id, player_id);

    const auto& hero_loc = hero.location();

    torch::Tensor x = torch::ones({ 12 });

    x[0] = hero_loc.x() / 7000;
    x[1] = hero_loc.y() / 7000;

    float ally_creep_dis = 0.0;
    float ally_tower_dis = 0.0;
    float enemy_creep_dis = 0.0;
    float enemy_creep_hp = 0.0;
    float enemy_tower_dis = 0.0;

    auto nearby_unit = get_nearby_unit(state, hero, 2000);

    for (const auto& s : nearby_unit) {
        int start_idx = -1;
        float dis = get_unit_distance(hero, s);
        float hp = s.health();
        if (s.unit_type() == CMsgBotWorldState_UnitType_LANE_CREEP) {
            if (s.team_id() == hero.team_id()) {
                if (ally_creep_dis == 0 || dis < ally_creep_dis) {
                    ally_creep_dis = dis;
                    start_idx = 2;
                }
            }
            else {
                /*
                if (enemy_creep_dis == 0 || dis < enemy_creep_dis) {
                    enemy_creep_dis = dis;
                    start_idx = 4;
                }
                */
                if (enemy_creep_hp == 0 || hp < enemy_creep_hp) {
                    enemy_creep_hp = hp;
                    start_idx = 4;
                    x[10] = hp;
                    x[11] = hero.attack_damage();
                }
            }
        }
        else if (s.unit_type() == CMsgBotWorldState_UnitType_TOWER) {
            if (s.team_id() == hero.team_id()) {
                if (ally_tower_dis == 0 || dis < ally_tower_dis) {
                    ally_tower_dis = dis;
                    start_idx = 6;
                }
            }
            else {
                if (enemy_tower_dis == 0 || dis < enemy_tower_dis) {
                    enemy_tower_dis = dis;
                    start_idx = 8;
                }
            }
        }
        if (start_idx > 0) {
            const auto& loc = s.location();
            x[start_idx] = (loc.x() - hero_loc.x()) / near_by_scale;
            x[start_idx + 1] = (loc.y() - hero_loc.y()) / near_by_scale;
        }
    }
    return x;
}

Units filter_attackable_units(const Units &units) {
    Units ret;
    for (const auto& unit:units) {
        if (!unit.is_invulnerable() &&
                !unit.is_attack_immune()
                && unit.is_alive()) {
            ret.push_back(unit);
        }
    }
    return ret;
}

CMsgBotWorldState_Unit get_nearest_unit(const Units& units, const CMsgBotWorldState_Unit& target) {
    if (units.empty()) {
        throw std::runtime_error("got empty units vector");
    }
    float min_dist = 10000.f;
    CMsgBotWorldState_Unit ret;
    float target_x = target.location().x();
    float target_y = target.location().y();
    for (const auto& unit:units) {
        float dx = unit.location().x() - target_x;
        float dy = unit.location().y() - target_y;
        float d = sqrtf(dx*dx + dy*dy);
        if (d < min_dist) {
            ret = unit;
            min_dist = d;
        }
    }
    return ret;
}

NS_UTIL_END