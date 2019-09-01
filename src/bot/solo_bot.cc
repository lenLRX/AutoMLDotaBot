#include <thread>
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
#include "dotaclient/dotaservice.h"
#include <chrono>

using namespace std::chrono_literals;

void runner_thread(const char* hostname,
                   short port,
                   int max_game_time,
                   nn::Net& rad_net,
                   nn::Net& dire_net,
                   nn::ReplayQueue* rad_queue,
                   nn::ReplayQueue* dire_queue);

void trainer_thread(nn::Net& rad_net,
                    nn::Net& dire_net,
                    nn::ReplayQueue* rad_queue,
                    nn::ReplayQueue* dire_queue);


int main(int argc, char** argv) {
    auto loss_logger = spdlog::basic_logger_mt("loss_logger", "loss.log", true);
    loss_logger->flush_on(spdlog::level::level_enum::info);
    auto action_logger = spdlog::basic_logger_mt("action_logger", "action.log", true);
    auto reward_logger = spdlog::basic_logger_mt("reward_logger", "reward.log", true);

    auto exception_logger = spdlog::basic_logger_mt("exception_logger", "exception.log", true);
    auto trace_logger = spdlog::basic_logger_mt("trace_logger", "trace.log", true);

    nn::Net rad_net(0.99);
    nn::Net dire_net(0.99);

    nn::ReplayQueue rad_queue;
    nn::ReplayQueue dire_queue;

    std::vector<std::thread> workers;

    //short worker_num = 8;
    short worker_num = 8;
    std::cerr << "start with " << worker_num << " workers" << std::endl;
    short base_port = 13337 + 1;
    int max_game_time = 1000;

    for (short i = 0; i < worker_num; ++i) {
        workers.emplace_back(std::bind(runner_thread, "127.0.0.1",base_port + i, max_game_time,
                rad_net, dire_net, &rad_queue, &dire_queue));
    }

    std::thread trainer_thread_hnd(std::bind(trainer_thread, rad_net, dire_net, &rad_queue, &dire_queue));

    while (true) {
        dotaservice::DotaEnv env("127.0.0.1", 13337, HOST_MODE_GUI, max_game_time, false);
        env.reset();
        env.update_param(TEAM_RADIANT, rad_net);
        env.update_param(TEAM_DIRE, dire_net);
        while (env.game_running()) {
            env.step();
        }
    }
}


void runner_thread(const char* hostname,
                   short port,
                   int max_game_time,
                   nn::Net& rad_net,
                   nn::Net& dire_net,
                   nn::ReplayQueue* rad_queue,
                   nn::ReplayQueue* dire_queue) {
    dotaservice::DotaEnv env(hostname, port, HOST_MODE_DEDICATED, max_game_time, true);
    while (true) {
        env.reset();
        env.update_param(TEAM_RADIANT, rad_net);
        env.update_param(TEAM_DIRE, dire_net);
        while (env.game_running()) {
            env.step();
        }

        auto rad_replay = env.get_replay_buffer(TEAM_RADIANT);
        auto dire_replay = env.get_replay_buffer(TEAM_DIRE);

        rad_queue->add_buffer(rad_replay);
        dire_queue->add_buffer(dire_replay);
    }
}

void trainer_thread(nn::Net& rad_net,
                    nn::Net& dire_net,
                    nn::ReplayQueue* rad_queue,
                    nn::ReplayQueue* dire_queue) {
    while (true) {
        std::vector<nn::ReplayBuffer> rad_replays;
        std::vector<nn::ReplayBuffer> dire_replays;

        rad_queue->get_last_buffer(rad_replays, 100);
        dire_queue->get_last_buffer(dire_replays, 100);

        std::cerr << "trainner buffer size " << rad_replays.size() << std::endl;
        if (rad_replays.empty()) {
            std::this_thread::sleep_for(1s);
        }

        rad_net.train(rad_replays);
        dire_net.train(dire_replays);
    }
}

