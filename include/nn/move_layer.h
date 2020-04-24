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
    void train(std::vector<PackedData>& data) override ;
    void reset_custom() override ;

private:
    void save_state(const LayerForwardConfig &cfg);
    std::deque<torch::Tensor> state_buffer;
    std::deque<torch::Tensor> expert_action_buffer;
    std::deque<torch::Tensor> move_action_buffer;
    std::vector<torch::Tensor> expert_action;
    std::vector<torch::Tensor> move_action;
    std::pair<float, float> target_pos;
    float dist;
    float z;
};

NS_NN_END


#endif //AUTOMLDOTABOT_MOVE_LAYER_H
