#include <nn/torch_layer.h>
#include "nn/highlevel_decision.h"
#include "nn/torch_layer.h"
#include "util/util.h"

NS_NN_BEGIN


const static int input_shape = hero_state_size;
const static int hidden_shape = 256;
const static int output_shape = 3;


HighLevelDecision::HighLevelDecision() {
    networks["actor"] = std::dynamic_pointer_cast<TorchLayer>(std::make_shared<Dense>(input_shape, hidden_shape, output_shape));
    networks["critic"] = std::dynamic_pointer_cast<TorchLayer>(std::make_shared<Dense>(input_shape, hidden_shape, 1));
    networks["discriminator"] = std::dynamic_pointer_cast<TorchLayer>(std::make_shared<Dense>(input_shape + output_shape, hidden_shape, 1));
}

#define SAVE_EXPERT_ACTION() do {\
                            if (hero.level() > dotautil::get_total_ability_level(hero))\
                            {\
                                expert_idx = 2;\
                                break;\
                            }\
                            for (const auto& creep:nearby_attackable_enemy) {\
                                if (creep.health() < hero_atk * 1.1) {\
                                    expert_idx = 1;\
                                }\
                            }\
                            } while(0);\

std::shared_ptr<Layer> HighLevelDecision::forward_impl(const LayerForwardConfig &cfg) {
    //std::cerr << "using expert high level" << std::endl;
    CMsgBotWorldState_Unit hero = dotautil::get_hero(cfg.state, cfg.team_id, cfg.player_id);

    torch::Tensor x = dotautil::state_encoding(cfg.state, cfg.team_id, cfg.player_id);

    auto out = networks.at("actor")->get<Dense>()->forward(x);

    auto action_prob = torch::softmax(out, 0);
    auto action = torch::argmax(action_prob);

    //auto action = action_prob.multinomial(1);

    int idx = dotautil::to_number<int>(action);

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 2000);
    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());

    dotautil::Units nearby_enemy = dotautil::filter_units_by_team(nearby_units, opposed_team);
    dotautil::Units nearby_attackable_enemy = dotautil::filter_units_by_team(dotautil::get_nearby_unit(cfg.state, hero, 450), opposed_team);
    nearby_attackable_enemy = dotautil::filter_attackable_units(nearby_attackable_enemy);
    //std::cerr << "highlevel near creep size " << nearby_attackable_enemy.size() << std::endl;
    nearby_enemy = dotautil::filter_attackable_units(nearby_enemy);
    if (nearby_enemy.empty()) {
        idx = 0;
    }

    int hero_atk = hero.attack_damage();
    int expert_idx = 0;
    SAVE_EXPERT_ACTION();

    torch::Tensor expert = torch::zeros({output_shape});
    expert[expert_idx] = 1;
    expert_action.push_back(expert);

    torch::Tensor act = torch::zeros({output_shape});
    act[idx] = 1;
    one_hot_action.push_back(act);

    auto ret_layer = children.at(idx);
    ret_layer->forward(cfg);
    return ret_layer;
}


std::shared_ptr<Layer> HighLevelDecision::forward_expert(const LayerForwardConfig& cfg) {
    CMsgBotWorldState_Unit hero = dotautil::get_hero(cfg.state, cfg.team_id, cfg.player_id);

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 2000);
    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());

    int hero_atk = hero.attack_damage();
    int expert_idx = 0;

    dotautil::Units nearby_enemy = dotautil::filter_units_by_team(nearby_units, opposed_team);
    nearby_enemy = dotautil::filter_attackable_units(nearby_enemy);
    dotautil::Units nearby_attackable_enemy = dotautil::filter_units_by_team(dotautil::get_nearby_unit(cfg.state, hero, 600), opposed_team);
    nearby_attackable_enemy = dotautil::filter_attackable_units(nearby_attackable_enemy);
    //std::cerr << "expert highlevel near creep size " << nearby_attackable_enemy.size() << std::endl;
    SAVE_EXPERT_ACTION();

    torch::Tensor expert = torch::zeros({output_shape});
    expert[expert_idx] = 1;
    expert_action.push_back(expert);

    torch::Tensor act = torch::zeros({output_shape});
    act[expert_idx] = 1;
    one_hot_action.push_back(act);

    std::stringstream ss;
    ss << "expert highlevel selection " << expert_idx;
    action_logger->info(ss.str().c_str());


    auto ret_layer = children.at(expert_idx);
    ret_layer->forward(cfg);
    return ret_layer;
}

std::shared_ptr<Layer> HighLevelDecision::forward(const LayerForwardConfig &cfg) {
    PERF_TIMER();
    ticks.push_back(cfg.tick);
    save_state(cfg);
    if (cfg.expert_action) {
        return forward_expert(cfg);
    }
    return forward_impl(cfg);
}

void HighLevelDecision::save_state(const LayerForwardConfig &cfg) {
    torch::Tensor x = dotautil::state_encoding(cfg.state, cfg.team_id, cfg.player_id);
    states.push_back(x);
}

HighLevelDecision::PackedData HighLevelDecision::get_training_data(){
    PackedData ret;
    if (states.empty()) {
        return ret;
    }
    ret["state"] = torch::stack(states).to(torch::kCUDA);
    ret["expert_action"] = torch::stack(expert_action).to(torch::kCUDA);
    ret["actor_one_hot"] = torch::stack(one_hot_action).to(torch::kCUDA);
    return ret;
}

void HighLevelDecision::train(std::vector<PackedData>& data){
    std::lock_guard<std::mutex> g(mtx);
    auto& actor = *networks["actor"]->get<Dense>();
    auto& critic = *networks["critic"]->get<Dense>();
    auto& discriminator = *networks["discriminator"]->get<Dense>();
    torch::optim::SGD actor_optim(actor.parameters(),
                                  torch::optim::SGDOptions(lr));

    torch::optim::SGD critic_optim(critic.parameters(),
                                   torch::optim::SGDOptions(lr));

    torch::optim::SGD d_optim(discriminator.parameters(),
                              torch::optim::SGDOptions(lr));

    actor.train(true);
    actor.to(torch::kCUDA);
    critic.train(true);
    critic.to(torch::kCUDA);
    discriminator.train(true);
    discriminator.to(torch::kCUDA);

    torch::Device dev = torch::kCUDA;

    auto logger = spdlog::get("loss_logger");
    auto avg_total_loss = torch::zeros({1});
    int active_data_num = 0;

    for (const auto& p_data:data) {
        try {
            actor_optim.zero_grad();
            critic_optim.zero_grad();
            d_optim.zero_grad();

            if (p_data.count("state") == 0) {
                continue;
            }
            active_data_num++;

            torch::Tensor state = p_data.at("state").to(dev);

            torch::Tensor expert_act = p_data.at("expert_action").to(dev);

            torch::Tensor actor_one_hot = p_data.at("actor_one_hot").to(dev);// use real action

            actor_optim.zero_grad();
            torch::Tensor actor_out = actor.forward(state);

            torch::Tensor actor_action_prob = torch::softmax(actor_out, 1);
            //torch::Tensor total_loss = actor_d_loss2 + actor_loss + critic_loss;
            torch::Tensor total_loss = torch::binary_cross_entropy(actor_action_prob, expert_act).mean();
            avg_total_loss += total_loss;
            total_loss.backward();


            //critic_optim.step();
            actor_optim.step();
        }
        catch (const std::runtime_error& e) {
            auto exception_logger = spdlog::get("exception_logger");
            exception_logger->error(e.what());
            torch::Tensor state = p_data.at("state");
            exception_logger->info("{}: \n{} \n", "state", dotautil::torch_to_string(state));
            torch::Tensor expert_act = p_data.at("expert_action");
            exception_logger->info("{}: \n{} \n", "expert_action", dotautil::torch_to_string(expert_act));
            torch::Tensor actor_one_hot = p_data.at("actor_one_hot");// use real action
            exception_logger->info("{}: \n{} \n", "actor_one_hot", dotautil::torch_to_string(actor_one_hot));
            torch::Tensor reward = p_data.at("reward");
            exception_logger->info("{}: \n{} \n", "reward", dotautil::torch_to_string(reward));
            throw;
        }
    }

    logger->info("episode {} {}, training loss: {}", 0, get_name(),
                 dotautil::torch_to_string(avg_total_loss / active_data_num));

}

void HighLevelDecision::reset_custom() {
    expert_action.clear();
    one_hot_action.clear();
}

NS_NN_END
