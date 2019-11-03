//
// Created by len on 7/15/19.
//
#include "nn/attack_layer.h"

NS_NN_BEGIN

const static int input_shape = 4;
const static int hidden_shape = 256;
const static int output_shape = 1;

AttackLayer::AttackLayer():atk_handle(-1) {
    networks["actor"] = std::make_shared<TorchLayer>(std::make_shared<Dense>(input_shape, hidden_shape, output_shape));
    networks["critic"] = std::make_shared<TorchLayer>(std::make_shared<Dense>(input_shape, hidden_shape, 1));
    networks["discriminator"] = std::make_shared<TorchLayer>(std::make_shared<Dense>(input_shape + output_shape, hidden_shape, 1));
}


static torch::Tensor creep_encoding(const CMsgBotWorldState_Unit& h,
        const CMsgBotWorldState_Unit& s) {
    torch::Tensor x = torch::zeros({ 4 });
    const auto& hloc = h.location();
    const auto& sloc = s.location();
    x[0] = (sloc.x() - hloc.x()) / near_by_scale;
    x[1] = (sloc.y() - hloc.y()) / near_by_scale;
    x[2] = s.health();
    x[3] = h.attack_damage();
    return x;
}


#define SAVE_EXPERT_ACTION() for (const auto& creep:enemy_creeps) { \
        if (creep.health() < hero_atk * 1.5) {\
            selected = i;\
            expert_action_cache.push_back(1);\
        }\
        else {\
            expert_action_cache.push_back(0);\
        }\
            ++i;\
    }


std::shared_ptr<Layer> AttackLayer::forward_impl(const LayerForwardConfig &cfg) {
    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
            cfg.team_id, cfg.player_id);

    const auto& location = hero.location();

    std::vector<torch::Tensor> states_cache;
    std::vector<int> expert_action_cache;
    std::vector<torch::Tensor> actor_action_cache;

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 2000);

    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());

    auto enemy_creeps = dotautil::filter_units_by_team(nearby_units, opposed_team);
    enemy_creeps = dotautil::filter_attackable_units(enemy_creeps);

    int hero_atk = hero.attack_damage();

    float max_value = 0.0;
    int idx = -1;
    int i = 0;

    for (const auto& creep:enemy_creeps) {
        auto x = creep_encoding(hero, creep);
        states_cache.push_back(x);
        auto out = sigmoid(networks.at("actor")->get<Dense>()->forward(x));
        actor_action_cache.push_back(out);
        auto out_value = dotautil::to_number<float>(out);

        if (out_value > max_value) {
            max_value = out_value;
            idx = i;
        }
        i++;
    }

    if (idx < 0) {
        auto action_logger = spdlog::get("action_logger");
        action_logger->critical("{} attack layer found no creep, invalid idx {}",
                            get_name(), idx);
        throw std::exception();
    }

    i = 0;
    int selected = -1;
    SAVE_EXPERT_ACTION();

    if (selected < 0) {
        selected = 0;
    }

    actual_action_idx.push_back(idx);
    actual_expert_act.push_back(idx == expert_action_cache[idx]);
    actual_state.push_back(creep_encoding(hero, enemy_creeps[idx]));

    atk_handle = enemy_creeps.at(idx).handle();
    return std::shared_ptr<Layer>();
}

std::shared_ptr<Layer> AttackLayer::forward_expert(const LayerForwardConfig &cfg) {
    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                            cfg.team_id, cfg.player_id);

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 2000);

    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());

    auto enemy_creeps = dotautil::filter_units_by_team(nearby_units, opposed_team);
    enemy_creeps = dotautil::filter_attackable_units(enemy_creeps);


    int hero_atk = hero.attack_damage();

    atk_handle = -1;

    int i = 0;
    int selected = -1;

    std::vector<int> expert_action_cache;

    SAVE_EXPERT_ACTION();

    if (selected < 0) {
        selected = 0;
    }

    atk_handle = enemy_creeps[selected].handle();

    actual_action_idx.push_back(selected);
    actual_expert_act.push_back(1);
    actual_state.push_back(creep_encoding(hero, enemy_creeps[selected]));
    return std::shared_ptr<Layer>();
}

std::shared_ptr<Layer> AttackLayer::forward(const LayerForwardConfig &cfg) {
    ticks.push_back(cfg.tick);
    tick_offset.push_back(states.size());
    save_state(cfg);
    if (cfg.expert_action) {
        return forward_expert(cfg);
    }
    return forward_impl(cfg);
}


CMsgBotWorldState_Action AttackLayer::get_action() {
    CMsgBotWorldState_Action ret;
    ret.set_actiontype(CMsgBotWorldState_Action_Type_DOTA_UNIT_ORDER_ATTACK_TARGET);
    ret.mutable_attacktarget()->set_target(atk_handle);
    ret.mutable_attacktarget()->set_once(false);
    return ret;
}

void AttackLayer::save_state(const LayerForwardConfig &cfg) {
    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                            cfg.team_id, cfg.player_id);
    const auto& location = hero.location();

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 2000);

    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());

    auto enemy_creeps = dotautil::filter_units_by_team(nearby_units, opposed_team);
    enemy_creeps = dotautil::filter_attackable_units(enemy_creeps);

    state_len.push_back(enemy_creeps.size());

    for (const auto& creep:enemy_creeps) {
        states.push_back(creep_encoding(hero, creep));
    }

}


AttackLayer::PackedData AttackLayer::get_training_data() {
    PackedData ret;
    if (states.empty()) {
        return ret;
    }
    ret["state"] = torch::stack(states).to(torch::kCUDA);
    //ret["expert_action"] = dotautil::vector2tensor(expert_action).to(torch::kCUDA);
    ret["expert_actual_act"] = dotautil::vector2tensor(actual_expert_act).to(torch::kCUDA);
    ret["actual_state"] = torch::stack(actual_state).to(torch::kCUDA);
    return ret;
}

void AttackLayer::train(std::vector<PackedData>& data){
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
        actor_optim.zero_grad();
        critic_optim.zero_grad();
        d_optim.zero_grad();

        if (p_data.count("state") == 0) {
            continue;
        }

        active_data_num++;

        torch::Tensor state = p_data.at("state").to(dev);
        torch::Tensor actual_state_ = p_data.at("actual_state").to(dev);
        //torch::Tensor expert_act = p_data.at("expert_action").view({-1, 1}).to(dev);
        torch::Tensor expert_actual_act_ = p_data.at("expert_actual_act").view({-1, 1}).to(dev);
        torch::Tensor reward = p_data.at("reward").to(dev);

        /*
        torch::Tensor expert_prob = torch::sigmoid(discriminator.forward(
                torch::cat({ state, expert_act }, state.dim() - 1)));
        torch::Tensor expert_label = torch::ones_like(expert_prob);
        torch::Tensor expert_d_loss = torch::binary_cross_entropy(expert_prob, expert_label).mean();

        torch::Tensor actor_action_prob = torch::sigmoid(actor.forward(state));
        torch::Tensor actor_prob = torch::sigmoid(discriminator.forward(
                torch::cat({state, actor_action_prob.detach()}, state.dim() - 1)));
        torch::Tensor actor_label = torch::zeros_like(actor_prob);
        torch::Tensor actor_d_loss = torch::binary_cross_entropy(actor_prob, actor_label).mean();

        torch::Tensor prob_diff = torch::relu(expert_prob - actor_prob).detach();

        torch::Tensor total_d_loss = expert_d_loss + actor_d_loss;
        total_d_loss.backward();

        d_optim.step();


        critic_optim.zero_grad();

        torch::Tensor actor_actual_action_prob = torch::softmax(actor.forward(actual_state_), 1);

        torch::Tensor actor_prob2 = torch::sigmoid(discriminator.forward(
                torch::cat({ actual_state_, actor_actual_action_prob * expert_actual_act_}, state.dim() - 1)));
        torch::Tensor actor_label2 = torch::ones_like(actor_prob2);

        torch::Tensor actor_d_loss2 = (torch::relu(torch::binary_cross_entropy(actor_prob2, actor_label2) - 0.1))*prob_diff.mean();

        torch::Tensor values = critic.forward(actual_state_).squeeze();
        torch::Tensor critic_loss = reward - values;
        critic_loss = critic_loss * critic_loss;
        critic_loss = critic_loss.mean();

        torch::Tensor adv = reward - values.detach();

        torch::Tensor actor_log_probs = torch::log(torch::sum(actor_actual_action_prob, 1));
        torch::Tensor actor_loss = -actor_log_probs * adv;
        actor_loss = actor_loss.mean();
        */

        actor_optim.zero_grad();
        torch::Tensor actor_actual_action_ = sigmoid(actor.forward(actual_state_));
        //torch::Tensor total_loss = actor_d_loss2;//TODO FIX actor critic loss + actor_loss + critic_loss;
        torch::Tensor total_loss = actor_actual_action_ - expert_actual_act_;
        total_loss = total_loss * total_loss;
        total_loss = total_loss.mean();
        avg_total_loss += total_loss;
        total_loss.backward();

        //critic_optim.step();
        actor_optim.step();
    }

    if (active_data_num) {
        logger->info("episode {} {}, training loss: {}", 0, get_name(), dotautil::torch_to_string(avg_total_loss / active_data_num));
    }
}

void AttackLayer::reset_custom() {
    atk_handle = -1;
    state_len.clear();
    actual_action_idx.clear();
    tick_offset.clear();
    actual_state.clear();
    actual_expert_act.clear();
}

NS_NN_END