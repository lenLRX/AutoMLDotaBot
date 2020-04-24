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

int get_total_ability_level(const CMsgBotWorldState_Unit& hero) {
    int total_level = 0;
    for (const auto& ability:hero.abilities()) {
        total_level += ability.level();
    }
    return total_level;
}


torch::Tensor state_encoding(const CMsgBotWorldState& state,
        DOTA_TEAM team_id, int player_id) {
    CMsgBotWorldState_Unit hero = get_hero(state, team_id, player_id);

    const auto& hero_loc = hero.location();

    torch::Tensor x = torch::ones({ hero_state_size });

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
    int total_ability_level = dotautil::get_total_ability_level(hero);
    x[12] = (int)hero.level();
    x[13] = total_ability_level;
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


AbilityManager& AbilityManager::get_instance() {
    static AbilityManager instance;
    return instance;
}

int AbilityManager::get_id_by_name(const char* name){
    std::string id = data["DOTAAbilities"][name]["ID"];
    return atoi(id.c_str());
}


void AbilityManager::set_data(const nlohmann::json& new_data) {
    data = new_data;
}


void encode_single_unit(const CMsgBotWorldState_Unit& unit, torch::Tensor& tensor, int start_idx) {
    auto add_field = [&tensor] (float value, int& i) {
        tensor[i] = value;
        ++i;
    };

    int idx = start_idx;
    const auto& loc = unit.location();
    add_field(loc.x(), idx);
    add_field(loc.y(), idx);


    assert(start_idx - idx <= max_unit_field);
}

void StateManager::set_player_id(int id) {
    player_id_ = id;
}

void StateManager::update(const CMsgBotWorldState& state) {
    torch::Tensor ally_creep_state = torch::zeros({max_unit_num, max_unit_field});
    torch::Tensor enemy_creep_state = torch::zeros({max_unit_num, max_unit_field});
    const CMsgBotWorldState_Unit& self = get_hero(state, team_id_, player_id_);
    int ally_idx = 0;
    int enemy_idx = 0;
    for (const auto& unit:state.units()) {
        if (unit.team_id() == team_id_) {
            torch::Tensor single_unit = ally_creep_state[ally_idx];
            update_unit(single_unit, unit, self);
            ++ally_idx;
            if (ally_idx >= max_unit_num) {
                continue;
            }
        } else {
            torch::Tensor single_unit = enemy_creep_state[enemy_idx];
            update_unit(single_unit, unit, self);
            ++enemy_idx;
            enemy_handle_.push_back(unit.handle());
            if (enemy_idx >= max_unit_num) {
                continue;
            }
        }
    }
    ally_unit_states_.push_back(ally_creep_state);
    enemy_unit_states_.push_back(enemy_creep_state);
    hero_states_.push_back(update_hero(self));
}

torch::Tensor StateManager::get_ally_unit_state() {
    assert(!ally_unit_states_.empty());
    return ally_unit_states_.back();
}

torch::Tensor StateManager::get_enemy_unit_state() {
    assert(!enemy_unit_states_.empty());
    return enemy_unit_states_.back();
}

std::vector<int> StateManager::get_enemy_handle() {
    return enemy_handle_;
}

torch::Tensor StateManager::get_hero_state() {
    assert(!hero_states_.empty());
    return hero_states_.back();
}

void StateManager::update_unit(torch::Tensor& t, const CMsgBotWorldState_Unit& state,
                               const CMsgBotWorldState_Unit& self) {
    int i = 0;
    update_location(state, t, i);
    update_facing_angle(state, t, i);
    update_hp(state, t, i);

    const float attack_scale = 100;
    t[i] = state.attack_damage() / attack_scale;
    ++i;

    update_armor(state, t, i);

    const float move_speed_scale = 500;
    t[i] = state.current_movement_speed() / move_speed_scale;
    ++i;

    // is my team?
    t[i] = state.team_id() == team_id_;
    ++i;

    update_vector_to_me(state, self, t, i);

    // am i attacking unit?
    t[i] = self.attack_target_handle() == state.handle();
    ++i;

    // is unit attacking me?
    t[i] = state.attack_target_handle() == self.handle();
    ++i;

    // unit type is creep?
    t[i] = state.unit_type() == CMsgBotWorldState_UnitType_LANE_CREEP;
    ++i;

    // unit type is Tower?
    t[i] = state.unit_type() == CMsgBotWorldState_UnitType_TOWER;
    ++i;
    assert(i == max_unit_field);
}

torch::Tensor StateManager::update_hero(const CMsgBotWorldState_Unit& self) {
    torch::Tensor ret = torch::zeros({max_unit_field});
    int i = 0;
    update_location(self, ret, i);
    update_facing_angle(self, ret, i);
    update_hp(self, ret, i);

    const float attack_scale = 100;
    ret[i] = self.attack_damage() / attack_scale;
    ++i;

    update_armor(self, ret, i);

    const float move_speed_scale = 500;
    ret[i] = self.current_movement_speed() / move_speed_scale;
    ++i;

    // is my team?
    ret[i] = self.team_id() == team_id_;
    ++i;

    // padding 3 zero
    i += 7;

    assert(i == max_unit_field);
    return ret;
}

void StateManager::update_location(const CMsgBotWorldState_Unit& unit, torch::Tensor& t, int& i) {
    float scale_factor = 7000;
    const auto& loc = unit.location();
    t[i] = loc.x() / scale_factor;
    ++i;
    t[i] = loc.y() / scale_factor;
    ++i;
}

void StateManager::update_facing_angle(const CMsgBotWorldState_Unit& unit, torch::Tensor& t, int& i) {
    // unit.facing is [0,360] int
    float rad = static_cast<float>(unit.facing()) / 180.f * M_PI;
    t[i] = std::cos(rad);
    ++i;
    t[i] = std::sin(rad);
    ++i;
}

void StateManager::update_hp(const CMsgBotWorldState_Unit& unit, torch::Tensor& t, int& i) {
    const int history_len = 16;
    const float hp_scale_factor = 1000;
    auto handle = unit.handle();
    std::deque<float>& hp_vec = hp_buffer_[handle];
    if (hp_vec.empty()) {
        for (int n = 0;n < history_len;++n) {
            hp_vec.push_back(0);
        }
    }

    hp_vec.push_back(unit.health() / hp_scale_factor);
    hp_vec.pop_front();
    assert(hp_vec.size() == history_len);
    for (float hp : hp_vec) {
        t[i] = hp;
        ++i;
    }

    t[i] = unit.health_max() / hp_scale_factor;
    ++i;
}

void StateManager::update_armor(const CMsgBotWorldState_Unit& unit, torch::Tensor& t, int& i) {
    float armor = unit.armor();
    float damage_multiplier = 1 - ((0.052 * armor) / (0.9 + 0.048 * std::abs(armor)));
    t[i] = damage_multiplier;
    ++i;
}

void StateManager::update_vector_to_me(const CMsgBotWorldState_Unit& unit, const CMsgBotWorldState_Unit& me,
                         torch::Tensor& t, int& i) {
    float scale_factor = 7000;
    const auto& my_loc = me.location();
    const auto& unit_loc = unit.location();
    float dx = unit_loc.x() - my_loc.x();
    dx /= scale_factor;
    float dy = unit_loc.y() - my_loc.y();
    dy /= scale_factor;
    t[i] = dx;
    ++i;
    t[i] = dy;
    ++i;

    float distance = std::sqrt(dx * dx + dy * dy);
    t[i] = distance;
    ++i;
}


// should be called after update state
void StateManager::update_reward(const CMsgBotWorldState& state) {
    if (hero_states_.size() <= 1) {
        return;
    }
    int state_num = hero_states_.size();
    torch::Tensor current_tensor = hero_states_[state_num-1];
    torch::Tensor prev_tensor = hero_states_[state_num-2];

    float current_dis_to_mid = std::sqrt(to_number<float>(current_tensor[0] * current_tensor[0]
            + current_tensor[1] * current_tensor[1]));

    float prev_dis_to_mid = std::sqrt(to_number<float>(prev_tensor[0] * prev_tensor[0]
                                                              + prev_tensor[1] * prev_tensor[1]));
    // dis range 0 to 1
    float dist_reward = (prev_dis_to_mid - current_dis_to_mid) * 1E-3f;
}

NS_UTIL_END