//
// Created by len on 6/4/19.
//

#ifndef AUTOMLDOTABOT_ENV_H
#define AUTOMLDOTABOT_ENV_H

#include "util.h"

#include <string>

#include "dotaservice/protos/DotaService.grpc.pb.h"
#include "dotaservice/protos/DotaService.pb.h"


NS_DOTACLIENT_BEGIN

class DotaEnv {
public:
    DotaEnv(const std::string& host, short port);

    void reset();

    std::shared_ptr<CMsgBotWorldState> get_state(Team team);
    std::shared_ptr<Actions> get_action(Team team);

    void step();

private:
    void reset_action(Team team);

    std::string host;
    short port;
    bool valid;
    std::shared_ptr<DotaService::Stub> env_stub;

    std::shared_ptr<CMsgBotWorldState> radiant_state;
    std::shared_ptr<CMsgBotWorldState> dire_state;

    std::shared_ptr<Actions> radiant_action;
    std::shared_ptr<Actions> dire_action;
};

NS_DOTACLIENT_END

#endif //AUTOMLDOTABOT_ENV_H
