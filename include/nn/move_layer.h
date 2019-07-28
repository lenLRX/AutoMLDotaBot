//
// Created by len on 7/7/19.
//

#ifndef AUTOMLDOTABOT_MOVE_LAYER_H
#define AUTOMLDOTABOT_MOVE_LAYER_H


#include "layer.h"
NS_NN_BEGIN

class MoveLayer : public Layer
{
public:
    MoveLayer();
    virtual ~MoveLayer() {}

    TYPE_NAME(MoveLayer)

    std::shared_ptr<Layer> forward_expert(const LayerForwardConfig &cfg) override ;

    std::shared_ptr<Layer> forward_impl(const LayerForwardConfig &cfg) override ;

    std::shared_ptr<Layer> forward(const LayerForwardConfig &cfg) override ;

    CMsgBotWorldState_Action get_action() final ;

    PackedData get_training_data() override ;
    void train(PackedData& data) override ;

private:
    void save_state(const LayerForwardConfig &cfg);
    std::vector<torch::Tensor> expert_action;
    std::pair<float, float> target_pos;
};

NS_NN_END


#endif //AUTOMLDOTABOT_MOVE_LAYER_H
