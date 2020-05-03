//
// Created by len on 4/25/20.
//

#ifndef AUTOMLDOTABOT_SOLO_PRED_REWARD_H
#define AUTOMLDOTABOT_SOLO_PRED_REWARD_H

#include <torch/script.h>
#include "layer.h"
#include "util/util.h"

NS_NN_BEGIN

class SoloPredReward {
public:
    explicit SoloPredReward(const std::string& model_path);

    void reset(const dotautil::ObserverState &cfg);

    float get_reward(const dotautil::ObserverState &cfg);
    float forward(const dotautil::ObserverState &cfg);
private:

    std::vector<torch::jit::IValue> make_state_inputs(const dotautil::ObserverState &cfg);

    torch::Tensor make_hero_value(const CMsgBotWorldState_Unit& unit,
                                  const dotautil::ObserverState &cfg);
    torch::Tensor make_creep_value(const CMsgBotWorldState_Unit& unit,
                                   const dotautil::ObserverState &cfg);

    float last_prob_;
    torch::jit::IValue lstm_states_;
    torch::jit::Module model_;
};

NS_NN_END

#endif //AUTOMLDOTABOT_SOLO_PRED_REWARD_H
