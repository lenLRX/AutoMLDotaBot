#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
#include "dotaclient/dotaservice.h"

int main(int argc, char** argv) {
    auto loss_logger = spdlog::basic_logger_mt("loss_logger", "loss.log", true);
    auto action_logger = spdlog::basic_logger_mt("action_logger", "action.log", true);
    auto reward_logger = spdlog::basic_logger_mt("reward_logger", "reward.log", true);
    while (true) {
        dotaservice::DotaEnv env("127.0.0.1", 13337, HOST_MODE_GUI);
        env.reset();
        while (env.game_running()) {
            env.step();
        }
    }
}
