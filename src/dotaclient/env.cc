//
// Created by len on 6/4/19.
//

#include "grpcpp/grpcpp.h"
#include "nlohmann/json.hpp"

#include "dotaclient/env.h"

NS_DOTACLIENT_BEGIN

DotaEnv::DotaEnv(const std::string &host, short port)
    :host(host), port(port), valid(false)
{
    auto channel = grpc::CreateChannel(host + ":" + std::to_string(port),
            grpc::InsecureChannelCredentials());
    env_stub = DotaService::NewStub(channel);
}

std::shared_ptr<CMsgBotWorldState> DotaEnv::get_state(Team team) {
    if (team == TEAM_RADIANT) {
        return radiant_state;
    }
    return dire_state;
}

std::shared_ptr<Actions> DotaEnv::get_action(Team team) {
    if (team == TEAM_RADIANT) {
        return radiant_action;
    }
    return dire_action;
}

void DotaEnv::reset() {
    auto cfg = GameConfig();

    for (int i = 0; i < 5; ++i) {
        auto hero = cfg.mutable_hero_picks()->Add();
        hero->set_team_id(TEAM_RADIANT);
        hero->set_hero_id(i == 0 ? NPC_DOTA_HERO_NEVERMORE : NPC_DOTA_HERO_SNIPER);
        hero->set_control_mode(i == 0 ? HERO_CONTROL_MODE_CONTROLLED : HERO_CONTROL_MODE_IDLE);
    }

    for (int i = 0; i < 5; ++i) {
        auto hero = cfg.mutable_hero_picks()->Add();
        hero->set_team_id(TEAM_DIRE);
        hero->set_hero_id(i == 0 ? NPC_DOTA_HERO_NEVERMORE : NPC_DOTA_HERO_SNIPER);
        hero->set_control_mode(i == 0 ? HERO_CONTROL_MODE_CONTROLLED : HERO_CONTROL_MODE_IDLE);
    }

    cfg.set_host_mode(HOST_MODE_DEDICATED);
    cfg.set_game_mode(DOTA_GAMEMODE_1V1MID);
    cfg.set_ticks_per_observation(10);
    cfg.set_host_timescale(5);

    grpc::ClientContext ctx;
    InitialObservation initialObservation;
    auto status = env_stub->reset(&ctx, cfg, &initialObservation);

    radiant_state = std::make_shared<CMsgBotWorldState>();
    *radiant_state = initialObservation.world_state_radiant();
    dire_state = std::make_shared<CMsgBotWorldState>();
    *dire_state = initialObservation.world_state_dire();

    reset_action(TEAM_RADIANT);
    reset_action(TEAM_DIRE);
}

void DotaEnv::step() {
    {
        grpc::ClientContext context;
        ObserveConfig ob_cfg;
        ob_cfg.set_team_id(TEAM_RADIANT);
        Observation ob;
        env_stub->observe(&context, ob_cfg, &ob);
        *radiant_state = ob.world_state();
    }

    {
        grpc::ClientContext context;
        ObserveConfig ob_cfg;
        ob_cfg.set_team_id(TEAM_RADIANT);
        Observation ob;
        env_stub->observe(&context, ob_cfg, &ob);
        *dire_state = ob.world_state();
    }

    {
        grpc::ClientContext action_context;
        Empty empty;
        env_stub->act(&action_context, *radiant_action, &empty);
        reset_action(TEAM_RADIANT);
    }

    {
        grpc::ClientContext action_context;
        Empty empty;
        env_stub->act(&action_context, *dire_action, &empty);
        reset_action(TEAM_DIRE);
    }
}

void DotaEnv::reset_action(Team team) {
    if (team == TEAM_RADIANT) {
        radiant_action = std::make_shared<Actions>();
        radiant_action->set_team_id(TEAM_RADIANT);
    }
    else {
        dire_action = std::make_shared<Actions>();
        dire_action->set_team_id(TEAM_DIRE);
    }
}


NS_DOTACLIENT_END