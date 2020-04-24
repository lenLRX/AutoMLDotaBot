//
// Created by len on 11/3/19.
//

#include "nn/ability_tree_layer.h"

NS_NN_BEGIN

const static int input_shape = 1;
const static int hidden_shape = 128;
const static int output_shape = 4;

static const int ability_num = 4;
// TODO talent
static const int level_to_ability[25] = {1,0,1,0,0,3,0,1,1,2,2,2,3,2,2,2,2,2,2,2,2,2,2,2,3};
static const char* ability_name[ability_num] = {"nevermore_shadowraze1",
                                      "nevermore_necromastery",
                                      "nevermore_dark_lord",
                                      "nevermore_requiem"};

int get_expert_action(int level) {
    auto ability_to_learn = level_to_ability[level - 1];
    return ability_to_learn;
}

AbilityTreeLayer::AbilityTreeLayer() :ability_idx(-1) {
    networks["actor"] = std::dynamic_pointer_cast<TorchLayer>(std::make_shared<Dense>(input_shape, hidden_shape, output_shape));
    networks["critic"] = std::dynamic_pointer_cast<TorchLayer>(std::make_shared<Dense>(input_shape, hidden_shape, 1));
    networks["discriminator"] = std::dynamic_pointer_cast<TorchLayer>(std::make_shared<Dense>(input_shape + output_shape, hidden_shape, 1));
}

std::shared_ptr<Layer> AbilityTreeLayer::forward_expert(const LayerForwardConfig &cfg) {
    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                            cfg.team_id, cfg.player_id);
    int level = hero.level();
    auto ability_to_learn = get_expert_action(level);
    ability_idx = ability_to_learn;
    torch::Tensor x = torch::zeros({ability_num});
    x[ability_to_learn] = 1;
    expert_actions.push_back(x);
    actions.push_back(x);
    return std::shared_ptr<Layer>();
}

std::shared_ptr<Layer> AbilityTreeLayer::forward_impl(const LayerForwardConfig &cfg) {
    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                            cfg.team_id, cfg.player_id);
    torch::Tensor x = torch::zeros({1});
    int level = hero.level();
    x[0] = level;
    auto ability_to_learn = get_expert_action(level);
    auto out = networks.at("actor")->get<Dense>()->forward(x);

    auto action_prob = torch::softmax(out, 0);
    auto action = torch::argmax(action_prob);

    ability_idx = dotautil::to_number<int>(action);

    torch::Tensor expert_action = torch::zeros({ability_num});
    expert_action[ability_to_learn] = 1;
    expert_actions.push_back(expert_action);
    actions.push_back(action);
    return std::shared_ptr<Layer>();
}

std::shared_ptr<Layer> AbilityTreeLayer::forward(const LayerForwardConfig &cfg) {
    save_state(cfg);
    if (cfg.expert_action) {
        expert_mode = true;
        return forward_expert(cfg);
    }
    return forward_impl(cfg);
}

CMsgBotWorldState_Action AbilityTreeLayer::get_action() {
    //int id = dotautil::AbilityManager::get_instance().
    //        get_id_by_name(ability_name[ability_idx]);
    CMsgBotWorldState_Action ret;
    ret.set_actiontype(CMsgBotWorldState_Action_Type_DOTA_UNIT_ORDER_TRAIN_ABILITY);
    ret.mutable_trainability()->set_ability(ability_name[ability_idx]);
    return ret;
}

void AbilityTreeLayer::reset_custom() {
    expert_actions.clear();
    actions.clear();
}

AbilityTreeLayer::PackedData AbilityTreeLayer::get_training_data() {
    PackedData ret;
    if (states.empty()) {
        return ret;
    }
    ret["state"] = torch::stack(states).to(torch::kCUDA);
    ret["expert_action"] = torch::stack(expert_actions).to(torch::kCUDA);
    ret["action"] = torch::stack(actions).to(torch::kCUDA);
    return ret;
}

void AbilityTreeLayer::train(std::vector<AbilityTreeLayer::PackedData>& data) {
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

            torch::Tensor actor_one_hot = p_data.at("action").to(dev);// use real action
            
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
            torch::Tensor actor_one_hot = p_data.at("action");// use real action
            exception_logger->info("{}: \n{} \n", "action", dotautil::torch_to_string(actor_one_hot));
            torch::Tensor reward = p_data.at("reward");
            exception_logger->info("{}: \n{} \n", "reward", dotautil::torch_to_string(reward));
            throw;
        }
    }

    logger->info("episode {} {}, training loss: {}", 0, get_name(),
                 dotautil::torch_to_string(avg_total_loss / active_data_num));

}

void AbilityTreeLayer::save_state(const LayerForwardConfig &cfg) {
    torch::Tensor x = torch::zeros({1});
    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                            cfg.team_id, cfg.player_id);
    x[0] = (int)hero.level();
    states.push_back(x);
}

NS_NN_END