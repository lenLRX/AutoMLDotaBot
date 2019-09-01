//
// Created by len on 6/4/19.
//

#ifndef AUTOMLDOTABOT_ENV_H
#define AUTOMLDOTABOT_ENV_H

#include "util/util.h"
#include "nn/nn.h"

#include <string>

#include "dotaservice/protos/DotaService.grpc.pb.h"
#include "dotaservice/protos/DotaService.pb.h"


NS_DOTACLIENT_BEGIN

class DotaEnv {
public:
    DotaEnv(const std::string& host, short port,
            HostMode mode, int max_game_time, bool expert_action = false);

    bool game_running();

    void reset();

    std::shared_ptr<CMsgBotWorldState> get_state(Team team);
    std::shared_ptr<Actions> get_action(Team team);

    void step();

    nn::ReplayBuffer get_replay_buffer(Team team);

    void update_param(Team team, const nn::Net& net);

private:
    void init();
    std::shared_ptr<nn::Net> init_net();

    void reset_player_id(Team team);

    void send_action(Team team);
    void add_action(const CMsgBotWorldState_Action& action, Team team);
    void reset_action(Team team);

    std::string host;
    short port;
    HostMode host_mode;
    int max_game_time;
    bool expert_action;
    bool valid;
    std::shared_ptr<DotaService::Stub> env_stub;

    std::shared_ptr<CMsgBotWorldState> radiant_state;
    std::shared_ptr<CMsgBotWorldState> dire_state;

    std::shared_ptr<Actions> radiant_action;
    std::shared_ptr<Actions> dire_action;

    std::shared_ptr<nn::Net> radiant_net;
    std::shared_ptr<nn::Net> dire_net;

    std::set<uint32_t> radiant_dummy_player;
    std::set<uint32_t> dire_dummy_player;

    int tick;
    uint32_t radiant_player_id;
    uint32_t dire_player_id;

    Status game_status;
};

NS_DOTACLIENT_END

#endif //AUTOMLDOTABOT_ENV_H
