//
// Created by len on 6/5/19.
//

#ifndef AUTOMLDOTABOT_LAYER_H
#define AUTOMLDOTABOT_LAYER_H

#include "util.h"

NS_NN_BEGIN



class Layer: std::enable_shared_from_this<Layer> {
public:

    using PackedData = std::map<std::string, torch::Tensor>;
    using NetWork = std::shared_ptr<torch::nn::Module>;
    using NetWorks = std::map<std::string, NetWork>;
    using Ptr = std::shared_ptr<Layer>;

    virtual const std::string& get_type() = 0;

    // let network set the name for layer so each layer will have a unique name
    void set_name(const std::string& new_name) {name = new_name;}
    const std::string& get_name() {return name;}

    virtual void forward(CMsgBotWorldState state) = 0;
    virtual CMsgBotWorldState_Action get_action();

    virtual PackedData get_trainng_data() = 0;
    virtual void train(PackedData& data) = 0;

    NetWorks networks;
    std::string name;
};





NS_NN_END

#endif //AUTOMLDOTABOT_LAYER_H
