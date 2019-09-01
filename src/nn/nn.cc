//
// Created by len on 6/30/19.
//

#include "nn/nn.h"
#include "nn/highlevel_decision.h"
#include "nn/attack_layer.h"
#include "nn/move_layer.h"

NS_NN_BEGIN

Net::Net(float discount_factor)
        :d_factor(discount_factor), last_reward(0), prev_last_hit(0) {
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
}

CMsgBotWorldState_Action Net::forward(const CMsgBotWorldState& state,
        DOTA_TEAM team_id, int player_id, int tick, bool expert_action) {
    LayerForwardConfig cfg(state, team_id, player_id, tick, expert_action);
    auto last_layer = root->forward(cfg);

    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                            cfg.team_id, cfg.player_id);

    uint32_t reward = hero.last_hits() - prev_last_hit;

    rewards.push_back(0);

    return last_layer->get_action();
}

void Net::padding_reward() {
    rewards.push_back(0);
}

void Net::collect_training_data() {
    auto discounted_rewards = get_discounted_reward();
    auto pack_data = root->get_training_data();
    pack_data["reward"] = root->get_masked_reward(discounted_rewards);
    replay_buffer[root->get_name()] = pack_data;
    for (const auto& child : root->children) {
        auto child_pack_data = child->get_training_data();
        child_pack_data["reward"] = child->get_masked_reward(discounted_rewards);
        replay_buffer[child->get_name()] = child_pack_data;
    }
}

const ReplayBuffer& Net::get_replay_buffer() {
    return replay_buffer;
}

std::vector<float> Net::get_discounted_reward() {
    std::vector<float> ret;

    rewards.push_back(last_reward);

    float r = 0;// may be should be value
    for (auto rit = rewards.rbegin(); rit != rewards.rend(); ++rit) {
        r = *rit + d_factor * r;
        ret.push_back(r);
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


void Net::reset() {
    last_reward = 0;
    root->reset();
}

void ReplayQueue::add_buffer(const ReplayBuffer& buffer){
    std::lock_guard<std::mutex> g(mtx);
    vec_buffer.push_back(buffer);
}


void ReplayQueue::get_last_buffer(std::vector<ReplayBuffer>& ret, int num) {
    ret.clear();
    std::lock_guard<std::mutex> g(mtx);
    int current_rep_num = vec_buffer.size();
    if (current_rep_num < num) {
        ret = vec_buffer;
    }
    else {
        int diff = current_rep_num - num;
        vec_buffer.erase(vec_buffer.begin(), vec_buffer.begin() + diff);
        ret = vec_buffer;
    }
}

void ReplayQueue::get_all_buffer(std::vector<ReplayBuffer>& ret) {
    ret.clear();
    std::lock_guard<std::mutex> g(mtx);
    vec_buffer.swap(ret);
}

NS_NN_END