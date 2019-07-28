//
// Created by len on 6/30/19.
//

#include "nn/nn.h"
#include "nn/highlevel_decision.h"
#include "nn/attack_layer.h"
#include "nn/move_layer.h"

NS_NN_BEGIN

Net::Net(float discount_factor)
        :d_factor(discount_factor),prev_last_hit(0) {
    auto root_ptr = new HighLevelDecision();
    root.reset(root_ptr);
    auto move_layer = std::make_shared<MoveLayer>();
    root_ptr->children.push_back(std::static_pointer_cast<Layer>(move_layer));
    auto atk_layer = std::make_shared<AttackLayer>();
    root_ptr->children.push_back(std::static_pointer_cast<Layer>(atk_layer));
}

CMsgBotWorldState_Action Net::forward(const CMsgBotWorldState& state,
        DOTA_TEAM team_id, int player_id, int tick) {
    LayerForwardConfig cfg(state, team_id, player_id, tick, true);
    auto last_layer = root->forward(cfg);

    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                            cfg.team_id, cfg.player_id);

    uint32_t reward = hero.last_hits() - prev_last_hit;

    rewards.push_back(reward);

    return last_layer->get_action();
}

void Net::collect_training_data() {
    replay_buffer[root->get_name()] = root->get_training_data();
    for (const auto& child : root->children) {
        replay_buffer[child->get_name()] = child->get_training_data();
    }
}

const ReplayBuffer& Net::get_replay_buffer() {
    return replay_buffer;
}

std::vector<float> Net::get_discounted_reward() {
    std::vector<float> ret;
    if (rewards.empty()) {
        return ret;
    }

    float r = 0;// may be should be value
    for (auto rit = rewards.rbegin(); rit != rewards.rend(); ++rit) {
        r = *rit + d_factor * r;
        ret.push_back(r);
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
}


NS_NN_END