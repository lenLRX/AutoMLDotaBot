//
// Created by len on 11/3/19.
//

#ifndef AUTOMLDOTABOT_ABILITY_TREE_LAYER_H
#define AUTOMLDOTABOT_ABILITY_TREE_LAYER_H

#include "layer.h"

NS_NN_BEGIN

class AbilityTreeLayer : public Layer {
public:
    AbilityTreeLayer();
    virtual ~AbilityTreeLayer() {}

    TYPE_NAME(AbilityTreeLayer)

    std::shared_ptr<Layer> forward_expert(const LayerForwardConfig &cfg) override ;

    std::shared_ptr<Layer> forward_impl(const LayerForwardConfig &cfg) override ;

    std::shared_ptr<Layer> forward(const LayerForwardConfig &cfg) override ;

    CMsgBotWorldState_Action get_action() final ;

    void reset_custom() override ;
    PackedData get_training_data() override ;
    void train(std::vector<PackedData>& data) override ;

private:
    void save_state(const LayerForwardConfig &cfg);

    int ability_idx;
    std::vector<torch::Tensor> expert_actions;
    std::vector<torch::Tensor> actions;
};

NS_NN_END

#endif //AUTOMLDOTABOT_ABILITY_TREE_LAYER_H
