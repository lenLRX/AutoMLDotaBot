//
// Created by len on 6/4/19.
//

#include "grpcpp/grpcpp.h"
#include "nlohmann/json.hpp"

#include "dotaclient/env.h"

#include "dotaclient/expert_item_build.h"

NS_DOTACLIENT_BEGIN

constexpr int ignore_tag = 0;
constexpr int rad_tag = 1;
constexpr int dire_tag = 2;


DotaEnv::DotaEnv(const std::string &host, short port, HostMode mode, int max_game_time,
        const std::string& win_prob_model_path, bool expert_action)
    :host(host), port(port), host_mode(mode), max_game_time(max_game_time), expert_action(expert_action), valid(false), tick(0),
    radiant_player_id(0), dire_player_id(0), pred_reward(win_prob_model_path), game_status(OK)
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
    return std::make_shared<nn::Net>();
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
    std::cerr << std::this_thread::get_id() << " env::reset" << std::endl;

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
    //cfg.set_ticks_per_observation(4);
    cfg.set_ticks_per_observation(2);
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

    rad_open_ai_net = std::make_shared<Network>(radiant_player_id, DOTA_TEAM_RADIANT);
    dire_open_ai_net = std::make_shared<Network>(dire_player_id, DOTA_TEAM_DIRE);
    rad_open_ai_net->reset();
    dire_open_ai_net->reset();

    send_action(TEAM_RADIANT);
    send_action(TEAM_DIRE);

    reset_action(TEAM_RADIANT);
    reset_action(TEAM_DIRE);

    //float prob = pred_reward.reset(ObserverState(*radiant_state, *dire_state));
    float prob = 0;
    curr_dota_time = initialObservation.world_state_dire().dota_time();
    radiant_net->reset(prob);
    dire_net->reset(1 - prob);
}

void DotaEnv::step() {
    std::shared_ptr<grpc::ClientAsyncResponseReader<Observation>> rad_ob_handle, dire_ob_handle;
    PERF_TIMER();
    Observation rad_ob;
    Observation dire_ob;
    grpc::Status rad_status;
    grpc::Status dire_status;
    {
        PERF_TIMER();
        grpc::ClientContext context;
        ObserveConfig ob_cfg;
        ob_cfg.set_team_id(TEAM_RADIANT);

        // TODO Async
        //env_stub->observe(&context, ob_cfg, &ob);
        rad_ob_handle = env_stub->Asyncobserve(&context, ob_cfg, &cq);
        rad_ob_handle->Finish(&rad_ob, &rad_status, (void*)rad_tag);
    }

    {
        grpc::ClientContext context;
        ObserveConfig ob_cfg;
        ob_cfg.set_team_id(TEAM_DIRE);
        dire_ob_handle = env_stub->Asyncobserve(&context, ob_cfg, &cq);
        dire_ob_handle->Finish(&dire_ob, &dire_status, (void*)dire_tag);
    }

    bool rad_ready = false;
    bool dire_ready = false;

    while (!(rad_ready && dire_ready)) {
        void* got_tag;
        bool ok = false;
        cq.Next(&got_tag, &ok);
        if (got_tag == (void*)rad_tag) {
            rad_ready = true;
        }
        if (got_tag == (void*)dire_tag) {
            dire_ready = true;
        }
    }

    *radiant_state = rad_ob.world_state();
    game_status = rad_ob.status();
    if (game_status != 0) {
        std::cerr << std::this_thread::get_id() << " Status " << game_status << std::endl;
        //throw std::exception();
    }
    if (radiant_state->game_time() > 0) {
        // most of tick dont have valid times ..
        curr_dota_time = radiant_state->dota_time();
    }

    *dire_state = dire_ob.world_state();

    //float rad_win_prob = pred_reward.forward(ObserverState(*radiant_state, *dire_state));
    float rad_win_prob = 0;

    reset_player_id(TEAM_RADIANT);
    reset_player_id(TEAM_DIRE);

    if (dotautil::has_hero(*get_state(TEAM_RADIANT),
            DOTA_TEAM_RADIANT, radiant_player_id)) {
        PERF_TIMER();
        CMsgBotWorldState_Action rad_action;
        rad_action = radiant_net->forward(*get_state(TEAM_RADIANT),
                                          DOTA_TEAM_RADIANT, radiant_player_id, tick, rad_win_prob, expert_action);
        // dotaservice didn't impl purchase item yet
        if (tick % 20 == 0) {
            //get_expert_item_build(rad_action, *get_state(TEAM_RADIANT), DOTA_TEAM_RADIANT, radiant_player_id);
        }
        rad_action.set_player(radiant_player_id);
        radiant_action->mutable_actions()->mutable_actions()->Add(std::move(rad_action));
        //rad_open_ai_net->set_player_id(radiant_player_id);
        //rad_open_ai_net->forward(*get_state(TEAM_RADIANT));
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
                                          DOTA_TEAM_DIRE, dire_player_id, tick, rad_win_prob, expert_action);
        d_action.set_player(dire_player_id);
        dire_action->mutable_actions()->mutable_actions()->Add(std::move(d_action));
        //dire_open_ai_net->set_player_id(dire_player_id);
        //dire_open_ai_net->forward(*get_state(TEAM_DIRE));
    }
    else {
        auto dire_noop = dotautil::no_op(dire_player_id);
        dire_action->mutable_actions()->mutable_actions()->Add(std::move(dire_noop));
        dire_net->padding_reward();
    }

    PERF_TIMER();

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

void DotaEnv::print_scoreboard() {
    std::cerr << "game end dota time " << curr_dota_time << " is expert " << expert_action << std::endl;
    radiant_net->print_scoreboard();
    dire_net->print_scoreboard();
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
    else {
        throw std::runtime_error("unknown team");
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