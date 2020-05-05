//
// Created by len on 6/30/19.
//

#include "nn/nn.h"
#include "nn/highlevel_decision.h"
#include "nn/attack_layer.h"
#include "nn/move_layer.h"
#include "nn/ability_tree_layer.h"

NS_NN_BEGIN

Net::Net()
        :prev_last_hit(0), prev_health(0), prev_win_prob(0.5f) {
    auto root_ptr = new HighLevelDecision();
    root.reset(root_ptr);
    //TODO auto rename
    root->set_name(root->get_type() + "1");
    auto move_layer = std::make_shared<MoveLayer>();
    root_ptr->children.push_back(std::static_pointer_cast<Layer>(move_layer));
    move_layer->set_name(move_layer->get_type() + "1");
    auto atk_layer = std::make_shared<AttackLayer>();
    atk_layer->set_name(atk_layer->get_type() + "1");
    root_ptr->children.push_back(std::static_pointer_cast<Layer>(atk_layer));
    auto ability_layer = std::make_shared<AbilityTreeLayer>();
    ability_layer->set_name(ability_layer->get_type() + "1");
    root_ptr->children.push_back(std::static_pointer_cast<Layer>(ability_layer));

    // set up discounting factor
    reward_map.insert({dotautil::reward_hp_key, RewardRecord(0.999)});
    reward_map.insert({dotautil::reward_lasthit_key, RewardRecord(0.999)});

    //binding reward fn
    auto health_reward = [this](const LayerForwardConfig& cfg)->float{
        const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                                cfg.team_id, cfg.player_id);
        float hp_reward = 0;
        float hp = hero.health();
        if (hp < prev_health) {
            hp_reward = -1;
        }
        prev_health = hp;
        return hp_reward;
    };
    auto lasthit_reward = [this](const LayerForwardConfig& cfg)->float {
        const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                                cfg.team_id, cfg.player_id);
        float lasthit = hero.last_hits();
        float last_hit_inc = lasthit - prev_last_hit;
        prev_last_hit = lasthit;
        return last_hit_inc;
    };

    auto win_prob_reward = [this](const LayerForwardConfig& cfg)->float  {
        float win_prob = cfg.rad_win_prob;
        if (cfg.team_id == DOTA_TEAM_DIRE) {
            win_prob = 1 - win_prob;
        }
        float reward = win_prob - prev_win_prob;
        prev_win_prob = win_prob;
        return reward;
    };

    reward_fn_map[dotautil::reward_hp_key] = health_reward;
    reward_fn_map[dotautil::reward_lasthit_key] = lasthit_reward;
}

CMsgBotWorldState_Action Net::forward(const CMsgBotWorldState& state,
        DOTA_TEAM team_id, int player_id, int tick, float rad_win_prob, bool expert_action) {
    LayerForwardConfig cfg(state, team_id, player_id, tick, rad_win_prob, expert_action);

    auto last_layer = root->forward(cfg);

    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                            cfg.team_id, cfg.player_id);

    for (const auto&p : reward_fn_map) {
        float reward_ = p.second(cfg);
        auto& r = reward_map.at(p.first);
        r.rewards.push_back(reward_);
        r.last_reward = reward_;
    }

    prev_health = hero.health();
    if (cfg.team_id == DOTA_TEAM_RADIANT) {
        prev_win_prob = cfg.rad_win_prob;
    }
    else {
        prev_win_prob = 1 - cfg.rad_win_prob;
    }


    return last_layer->get_action();
}

void Net::padding_reward() {
    for (auto& p:reward_map){
        p.second.rewards.push_back(0);
    }
}

void Net::collect_training_data() {
    std::map<std::string, std::vector<float>> m;
    for (auto& p:reward_map) {
        auto discounted_rewards = get_discounted_reward(p.second);
        m[p.first] = discounted_rewards;
    }

    auto pack_data = root->get_training_data();
    for (const auto&p :m){
        pack_data[p.first] = root->get_masked_reward(p.second);
    }

    replay_buffer[root->get_name()] = pack_data;
    for (const auto &child : root->children) {
        auto child_pack_data = child->get_training_data();
        for (const auto&p :m) {
            child_pack_data[p.first] = child->get_masked_reward(p.second);
        }
        replay_buffer[child->get_name()] = child_pack_data;
    }

}

const ReplayBuffer& Net::get_replay_buffer() {
    return replay_buffer;
}

std::vector<float> Net::get_discounted_reward(RewardRecord& reward) {
    std::vector<float> ret;

    auto& rewards = reward.rewards;
    rewards.push_back(reward.last_reward);

    float total_reward = 0;

    float r = 0;// may be should be value
    for (auto rit = rewards.rbegin(); rit != rewards.rend(); ++rit) {
        r = *rit + reward.discount_factor * r;
        if (r < -1) {
            r = -1;
        }
        ret.push_back(r);
        total_reward += *rit;
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
}

void Net::update_param(const Net& other){
    root->update_params(*other.root);
    int children_num = root->children.size();
    for (int i = 0;i < children_num; ++i) {
        root->children[i]->update_params(*other.root->children[i]);
    }
}

void Net::train(const std::vector<ReplayBuffer>& replays) {
    std::map<std::string, std::vector<Layer::PackedData>> layer2data;
    for (const auto& replay: replays) {
        for (const auto& p : replay.buffer) {
            layer2data[p.first].push_back(p.second);
        }
    }

    if (layer2data.count(root->get_name())) {
        root->train(layer2data.at(root->get_name()));
    }
    int children_num = root->children.size();
    for (int i = 0;i < children_num; ++i) {
        auto& c = root->children[i];
        if (layer2data.count(c->get_name())) {
            c->train(layer2data.at(c->get_name()));
        }
    }
}


void Net::reset(float prob) {
    last_hit_statistic = 0;
    prev_win_prob = prob;
    for (auto& p : reward_map) {
        p.second.reset();
    }
    root->reset();
}

void Net::print_scoreboard() {
    std::cerr << "score board: last hit " << prev_last_hit << std::endl;
}

void ReplayQueue::add_buffer(const ReplayBuffer& buffer){
    std::lock_guard<std::mutex> g(mtx);
    vec_buffer.push_back(buffer);
    if (vec_buffer.size() > capacity) {
        vec_buffer.pop_front();
    }
}


void ReplayQueue::get_last_buffer(std::vector<ReplayBuffer>& ret, int num) {
    ret.clear();
    std::lock_guard<std::mutex> g(mtx);
    int current_rep_num = vec_buffer.size();
    if (current_rep_num < num) {
        ret = {vec_buffer.begin(), vec_buffer.end()};
    }
    else {
        int diff = current_rep_num - num;
        vec_buffer.erase(vec_buffer.begin(), vec_buffer.begin() + diff);
        ret = {vec_buffer.begin(), vec_buffer.end()};
    }
}

void ReplayQueue::get_all_buffer(std::vector<ReplayBuffer>& ret) {
    ret.clear();
    std::lock_guard<std::mutex> g(mtx);
    ret = {vec_buffer.begin(), vec_buffer.end()};
    vec_buffer.clear();
}

NS_NN_END