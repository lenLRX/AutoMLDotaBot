//
// Created by len on 6/4/19.
//

#include "grpcpp/grpcpp.h"
#include "nlohmann/json.hpp"

#include "dotaclient/env.h"

NS_DOTACLIENT_BEGIN

DotaEnv::DotaEnv(const std::string &host, short port, HostMode mode, int max_game_time, bool expert_action)
    :host(host), port(port), host_mode(mode), max_game_time(max_game_time), expert_action(expert_action), valid(false), tick(0),
    radiant_player_id(0), dire_player_id(0), game_status(OK)
{
    auto channel = grpc::CreateChannel(host + ":" + std::to_string(port),
            grpc::InsecureChannelCredentials());
    env_stub = DotaService::NewStub(channel);
    init();
}

void DotaEnv::init() {
    radiant_net = init_net();
    dire_net = init_net();
}


std::shared_ptr<nn::Net> DotaEnv::init_net() {
    return std::make_shared<nn::Net>(0.99);
}

bool DotaEnv::game_running() {
    return game_status == OK;
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
    std::cerr << std::this_thread::get_id() << "env::reset" << std::endl;
    radiant_net->reset();
    dire_net->reset();
    tick = 0;
    game_status = OK;
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

    cfg.set_host_mode(host_mode);
    cfg.set_game_mode(DOTA_GAMEMODE_1V1MID);
    //cfg.set_ticks_per_observation(2);
    cfg.set_ticks_per_observation(10);
    cfg.set_host_timescale(10);

    grpc::ClientContext ctx;
    InitialObservation initialObservation;
    auto status = env_stub->reset(&ctx, cfg, &initialObservation);

    radiant_state = std::make_shared<CMsgBotWorldState>();
    *radiant_state = initialObservation.world_state_radiant();
    dire_state = std::make_shared<CMsgBotWorldState>();
    *dire_state = initialObservation.world_state_dire();

    reset_action(TEAM_RADIANT);
    reset_action(TEAM_DIRE);

    auto players = initialObservation.players();
    int player_num = initialObservation.players_size();
    for (int i = 0; i < player_num; ++i) {
        const auto& player = players.at(i);
        auto noop = dotautil::no_op(player.id());
        add_action(noop, player.team_id());
        if (player.hero() == NPC_DOTA_HERO_NEVERMORE) {
            if (player.team_id() == TEAM_RADIANT) {
                radiant_player_id = player.id();
            }
            else if (player.team_id() == TEAM_DIRE) {
                dire_player_id = player.id();
            }
        }

        else {
            if (player.team_id() == TEAM_RADIANT) {
                radiant_dummy_player.insert(player.id());
            }
            else if (player.team_id() == TEAM_DIRE) {
                dire_dummy_player.insert(player.id());
            }
        }
    }

    send_action(TEAM_RADIANT);
    send_action(TEAM_DIRE);

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
        game_status = ob.status();
        if (game_status != 0) {
            std::cerr << std::this_thread::get_id() << " Status " << game_status << std::endl;
            //throw std::exception();
        }
    }

    {
        grpc::ClientContext context;
        ObserveConfig ob_cfg;
        ob_cfg.set_team_id(TEAM_DIRE);
        Observation ob;
        env_stub->observe(&context, ob_cfg, &ob);
        *dire_state = ob.world_state();
    }

    reset_player_id(TEAM_RADIANT);
    reset_player_id(TEAM_DIRE);

    if (dotautil::has_hero(*get_state(TEAM_RADIANT),
            DOTA_TEAM_RADIANT, radiant_player_id)) {
        auto rad_action = radiant_net->forward(*get_state(TEAM_RADIANT),
                                               DOTA_TEAM_RADIANT, radiant_player_id, tick, expert_action);
        rad_action.set_player(radiant_player_id);
        radiant_action->mutable_actions()->mutable_actions()->Add(std::move(rad_action));
    }
    else {
        auto rad_noop = dotautil::no_op(radiant_player_id);
        radiant_action->mutable_actions()->mutable_actions()->Add(std::move(rad_noop));
        radiant_net->padding_reward();
    }

    for (auto player_id : radiant_dummy_player) {
        auto rad_noop = dotautil::no_op(player_id);
        radiant_action->mutable_actions()->mutable_actions()->Add(std::move(rad_noop));
    }

    if (dotautil::has_hero(*get_state(TEAM_DIRE),
            DOTA_TEAM_DIRE, dire_player_id)) {
        auto d_action = dire_net->forward(*get_state(TEAM_DIRE),
                                          DOTA_TEAM_DIRE, dire_player_id, tick, expert_action);
        d_action.set_player(dire_player_id);
        dire_action->mutable_actions()->mutable_actions()->Add(std::move(d_action));
    }
    else {
        auto dire_noop = dotautil::no_op(dire_player_id);
        dire_action->mutable_actions()->mutable_actions()->Add(std::move(dire_noop));
        dire_net->padding_reward();
    }

    for (auto player_id : dire_dummy_player) {
        auto dire_noop = dotautil::no_op(player_id);
        dire_action->mutable_actions()->mutable_actions()->Add(std::move(dire_noop));
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
    ++tick;

    if (tick > max_game_time) {
        game_status = OUT_OF_RANGE;
    }
}


nn::ReplayBuffer DotaEnv::get_replay_buffer(Team team) {
    if (team == TEAM_RADIANT) {
        radiant_net->collect_training_data();
        return radiant_net->get_replay_buffer();
    }
    else if (team == TEAM_DIRE) {
        dire_net->collect_training_data();
        return dire_net->get_replay_buffer();
    }
}

void DotaEnv::update_param(Team team, const nn::Net& net) {
    if (team == TEAM_RADIANT) {
        radiant_net->update_param(net);
    }
    else if (team == TEAM_DIRE) {
        dire_net->update_param(net);
    }
}

void DotaEnv::reset_player_id(Team team) {
    std::shared_ptr<CMsgBotWorldState> state = get_state(team);

    if (team == TEAM_RADIANT) {
        radiant_dummy_player.clear();
    }
    else {
        dire_dummy_player.clear();
    }

    int num_player = state->players().size();
    for (int i = 0;i < num_player; ++i) {
        auto player = state->players().at(i);
        if (player.hero_id() == NPC_DOTA_HERO_NEVERMORE) {
            if (player.team_id() == TEAM_RADIANT) {
                radiant_player_id = player.player_id();
            }
            else {
                dire_player_id = player.player_id();
            }
        }
        else {
            if (player.team_id() == TEAM_RADIANT) {
                radiant_dummy_player.insert(player.player_id());
            }
            else {
                dire_dummy_player.insert(player.player_id());
            }
        }
    }
}

void DotaEnv::send_action(Team team) {
    if (team == TEAM_RADIANT) {
        grpc::ClientContext action_context;
        Empty empty;
        env_stub->act(&action_context, *radiant_action, &empty);
        reset_action(TEAM_RADIANT);
    }
    else {
        grpc::ClientContext action_context;
        Empty empty;
        env_stub->act(&action_context, *dire_action, &empty);
        reset_action(TEAM_DIRE);
    }
}

void DotaEnv::add_action(const CMsgBotWorldState_Action& action, Team team) {
    if (team == TEAM_RADIANT) {
        *radiant_action->mutable_actions()->mutable_actions()->Add() = action;
    }
    else {
        *dire_action->mutable_actions()->mutable_actions()->Add() = action;
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