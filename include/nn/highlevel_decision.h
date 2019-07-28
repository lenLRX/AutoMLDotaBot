//
// Created by len on 6/9/19.
//

#ifndef AUTOMLDOTABOT_HIGHLEVEL_DECISION_H
#define AUTOMLDOTABOT_HIGHLEVEL_DECISION_H

#include "layer.h"
NS_NN_BEGIN

class HighLevelDecision : public Layer
{
public:
    HighLevelDecision();

    virtual ~HighLevelDecision() {}

    TYPE_NAME(HighlevelDecision)

    std::shared_ptr<Layer> forward_expert(const LayerForwardConfig& cfg) override;

    std::shared_ptr<Layer> forward_impl(const LayerForwardConfig &cfg) override ;

    std::shared_ptr<Layer> forward(const LayerForwardConfig &cfg) override ;

    PackedData get_training_data() override ;
    void train(PackedData& data) override ;

private:
    void save_state(const LayerForwardConfig &cfg);

    std::vector<torch::Tensor> expert_action;
};

NS_NN_END

#endif //AUTOMLDOTABOT_HIGHLEVEL_DECISION_H
