//
// Created by len on 7/14/19.
//

#ifndef AUTOMLDOTABOT_ATTACK_LAYER_H
#define AUTOMLDOTABOT_ATTACK_LAYER_H

#include "layer.h"

NS_NN_BEGIN

class AttackLayer : public Layer
{
public:
    AttackLayer();
    virtual ~AttackLayer() {}

    TYPE_NAME(AttackLayer)

    std::shared_ptr<Layer> forward_expert(const LayerForwardConfig &cfg) override ;

    std::shared_ptr<Layer> forward_impl(const LayerForwardConfig &cfg) override ;

    std::shared_ptr<Layer> forward(const LayerForwardConfig &cfg) override ;

    CMsgBotWorldState_Action get_action() final ;

    void reset_custom() override ;
    PackedData get_training_data() override ;
    void train(std::vector<PackedData>& data) override ;

private:

    void save_state(const LayerForwardConfig &cfg);

    uint32_t atk_handle;

    // length of each tick states
    std::vector<int> state_len;
    std::vector<int> expert_action;
    std::vector<int> actual_action_idx;
    std::vector<int> tick_offset;
    std::vector<torch::Tensor> actual_state;
    std::vector<int> actual_expert_act;
};

NS_NN_END
#endif //AUTOMLDOTABOT_ATTACK_LAYER_H
