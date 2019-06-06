#include <iostream>
#include <thread>
#include <chrono>
#include "grpcpp/grpcpp.h"
#include "nlohmann/json.hpp"
#include "dotaservice/protos/DotaService.grpc.pb.h"
#include "dotaservice/protos/DotaService.pb.h"

using json = nlohmann::json;
using grpc::Channel;
using grpc::ClientContext;

using namespace std::chrono_literals;

int main() {
    auto channel = grpc::CreateChannel("localhost:13337", grpc::InsecureChannelCredentials());
    auto state = channel->GetState(true);
    auto env = DotaService::NewStub(channel);
    //std::this_thread::sleep_for(1s);
    state = channel->GetState(true);
    std::cout << state << std::endl;
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

    ClientContext ctx;
    InitialObservation initialObservation;
    auto status = env->reset(&ctx, cfg, &initialObservation);

    //std::cout << initialObservation.DebugString() << std::endl;

    std::vector<Team> teams{TEAM_RADIANT, TEAM_DIRE};

    while (true){
        for (auto team: teams) {
            ClientContext ctx_;
            ObserveConfig ob_cfg;
            ob_cfg.set_team_id(TEAM_RADIANT);
            Observation ob;
            env->observe(&ctx_, ob_cfg, &ob);

            Actions actions;

            actions.set_team_id(team);

            ClientContext ctx_action;

            Empty response;

            env->act(&ctx_action, actions, &response);

            std::cout << ob.DebugString() << std::endl;
        }
    }

    return 0;
}
