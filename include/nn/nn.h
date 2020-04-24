//
// Created by len on 6/5/19.
//

#ifndef AUTOMLDOTABOT_NN_H
#define AUTOMLDOTABOT_NN_H

#include <functional>

#include "util/util.h"
#include "layer.h"
#include "openai_five/network.h"

NS_NN_BEGIN

class ReplayBuffer {
public:
    Layer::PackedData & operator [] (const std::string& key) {
        return buffer[key];
    }
    std::map<std::string, Layer::PackedData> buffer;
};

class RewardRecord {
public:
    explicit RewardRecord(float d) :
        discount_factor(d),
        last_reward(0) {
    }
    float discount_factor;
    std::vector<float> rewards;
    float last_reward;
    void reset() {
        rewards.clear();
        last_reward = 0;
    }
};

//
// a Net class contains the tree of the layers
//

class Net {
public:

    Net();

    CMsgBotWorldState_Action forward(const CMsgBotWorldState& state,
            DOTA_TEAM team_id, int player_id, int tick, bool expert_action);

    void collect_training_data();

    const ReplayBuffer& get_replay_buffer();

    std::vector<float> get_discounted_reward(RewardRecord& reward);

    void update_param(const Net& other);

    void padding_reward();

    void train(const std::vector<ReplayBuffer>& replays);

    void reset();

private:
    uint32_t prev_last_hit;
    uint32_t prev_health;
    std::unordered_map<std::string, RewardRecord> reward_map;
    std::unordered_map<std::string, std::function<float(const LayerForwardConfig&)>> reward_fn_map;
    ReplayBuffer replay_buffer;
    Layer::Ptr root;
};

class ReplayQueue {
public:
    void add_buffer(const ReplayBuffer& buffer);
    void get_last_buffer(std::vector<ReplayBuffer>& ret, int num);
    void get_all_buffer(std::vector<ReplayBuffer>& ret);
private:
    std::vector<ReplayBuffer> vec_buffer;
    std::mutex mtx;
};

NS_NN_END

#endif //AUTOMLDOTABOT_NN_H
