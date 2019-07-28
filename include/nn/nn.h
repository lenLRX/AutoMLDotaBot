//
// Created by len on 6/5/19.
//

#ifndef AUTOMLDOTABOT_NN_H
#define AUTOMLDOTABOT_NN_H

#include "util/util.h"
#include "layer.h"

NS_NN_BEGIN

class ReplayBuffer {
public:
    Layer::PackedData & operator [] (const std::string& key) {
        return buffer[key];
    }
    std::map<std::string, Layer::PackedData> buffer;
};

//
// a Net class contains the tree of the layers
//

class Net {
public:

    Net(float discount_factor);

    CMsgBotWorldState_Action forward(const CMsgBotWorldState& state,
            DOTA_TEAM team_id, int player_id, int tick);

    void collect_training_data();

    const ReplayBuffer& get_replay_buffer();

    std::vector<float> get_discounted_reward();

private:
    float d_factor;
    uint32_t prev_last_hit;
    std::vector<float> rewards;
    ReplayBuffer replay_buffer;
    Layer::Ptr root;
};

NS_NN_END

#endif //AUTOMLDOTABOT_NN_H
