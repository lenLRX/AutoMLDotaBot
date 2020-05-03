//
// Created by len on 6/5/19.
//

#ifndef AUTOMLDOTABOT_LAYER_H
#define AUTOMLDOTABOT_LAYER_H

#include "util/util.h"
#include "torch_layer.h"

NS_NN_BEGIN

#define TYPE_NAME(x) const std::string& get_type() override {\
    static std::string type = #x;\
    return type;\
}

class nn;

class LayerForwardConfig {
public:
    LayerForwardConfig(const CMsgBotWorldState& state,
        DOTA_TEAM team_id,
        uint32_t player_id,
        int tick,
        float rad_win_prob,
        bool expert_action=false)
        :state(state), team_id(team_id), player_id(player_id),
        tick(tick), rad_win_prob(rad_win_prob), expert_action(expert_action)
        {}
    CMsgBotWorldState state;
    DOTA_TEAM team_id;
    uint32_t player_id;
    int tick;
    float rad_win_prob;
    bool expert_action;
};

class Layer: public std::enable_shared_from_this<Layer> {
public:

    virtual ~Layer() {}

    using PackedData = std::map<std::string, torch::Tensor>;
    using NetWork = std::shared_ptr<TorchLayer>;
    using NetWorks = std::map<std::string, NetWork>;
    using Ptr = std::shared_ptr<Layer>;

    virtual const std::string& get_type() = 0;

    // let network set the name for layer so each layer will have a unique name
    void set_name(const std::string& new_name) {name = new_name;}
    const std::string& get_name() {return name;}

    // return the last layer
    virtual std::shared_ptr<Layer> forward(const LayerForwardConfig &cfg) {
        std::cerr << get_name() << " using default layer forward" << std::endl;
        ticks.push_back(cfg.tick);
        if (cfg.expert_action) {
            return forward_expert(cfg);
        }
        return forward_impl(cfg);
    }

    virtual std::shared_ptr<Layer> forward_expert(const LayerForwardConfig& cfg);

    virtual std::shared_ptr<Layer> forward_impl(const LayerForwardConfig &cfg) = 0;

    virtual CMsgBotWorldState_Action get_action();

    torch::Tensor get_masked_reward(const std::vector<float>& reward);

    virtual void update_params(Layer& other);

    virtual PackedData get_training_data() = 0;
    virtual void train(std::vector<PackedData>& data) = 0;
    void reset();

    virtual void reset_custom() {}

    NetWorks networks;
    std::vector<Ptr> children;
    std::string name;

    std::vector<torch::Tensor> states;
    std::vector<int> ticks;
    bool expert_mode{false};
    std::mutex mtx;
};





NS_NN_END

#endif //AUTOMLDOTABOT_LAYER_H
