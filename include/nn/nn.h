//
// Created by len on 6/5/19.
//

#ifndef AUTOMLDOTABOT_NN_H
#define AUTOMLDOTABOT_NN_H

#include "util.h"
#include "layer.h"

NS_NN_BEGIN

class ReplayBuffer {
public:
    std::map<std::string, Layer::PackedData> buffer;
};

//
// a Net class contains the tree of the layers
//

class Net {
public:
    ReplayBuffer replay_buffer;
    Layer:: root;
};

NS_NN_END

#endif //AUTOMLDOTABOT_NN_H
