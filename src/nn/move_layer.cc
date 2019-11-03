//
// Created by len on 7/7/19.
//

#include "nn/move_layer.h"
#include "util/util.h"

#include <exception>
#include <fstream>

NS_NN_BEGIN

const static int input_shape = 12;
const static int hidden_shape = 256;
const static int output_shape = 2;

const static float max_map_size = 8000;
const static float move_scale = 1500;
const static float move_scale_sqrt = 2121.32;

MoveLayer::MoveLayer(): dist(0), z(0) {
    networks["actor"] = std::make_shared<TorchLayer>(std::make_shared<Dense>(input_shape, hidden_shape, output_shape));
    networks["critic"] = std::make_shared<TorchLayer>(std::make_shared<Dense>(input_shape + output_shape, hidden_shape, 1));
    networks["discriminator"] = std::make_shared<TorchLayer>(std::make_shared<Dense>(input_shape + output_shape, hidden_shape, 1));
}

static std::pair<float, float> get_move_vec(torch::Tensor x) {
    x = x * move_scale;
    std::pair<float, float> out_(dotautil::to_number<float>(x[0]), dotautil::to_number<float>(x[1]));
    return out_;
}

static float atk_range = 450.0f;
static float map_bound = 8000.0f;

#define SAVE_EXPERT_ACTION()  if (enemy_creeps.empty() || dotautil::filter_units_by_type(enemy_creeps, CMsgBotWorldState_UnitType_LANE_CREEP).empty()) {\
                                    tmp_target_pos.first = -500;\
                                    tmp_target_pos.second = -500;\
                                }\
                                else {\
                                auto nearest_enemy = dotautil::get_nearest_unit(enemy_creeps, hero);\
                                float nearest_enemy_x = nearest_enemy.location().x();\
                                float nearest_enemy_y = nearest_enemy.location().y();\
                                if (cfg.team_id == DOTA_TEAM_DIRE) {\
                                    float d = (map_bound - nearest_enemy_x)*(map_bound - nearest_enemy_x) +\
                                    (map_bound - nearest_enemy_y) * (map_bound - nearest_enemy_y);\
                                    d = sqrtf(d);\
                                    float d_dot = d - atk_range;\
                                    float x_dot = d_dot / d * (map_bound - nearest_enemy_x);\
                                    float y_dot = d_dot / d * (map_bound - nearest_enemy_y);\
                                    tmp_target_pos.first = map_bound - x_dot;\
                                    tmp_target_pos.second = map_bound - y_dot;\
                                }\
                                else {\
                                    float d = (map_bound + nearest_enemy_x)*(map_bound + nearest_enemy_x) +\
                                    (nearest_enemy_y + map_bound) * (nearest_enemy_y + map_bound);\
                                    d = sqrtf(d);\
                                    float d_dot = d - atk_range;\
                                    float x_dot = d_dot / d * (map_bound + nearest_enemy_x);\
                                    float y_dot = d_dot / d * (map_bound + nearest_enemy_y);\
                                    tmp_target_pos.first = x_dot - map_bound;\
                                    tmp_target_pos.second = y_dot - map_bound;\
                                }\
                                }\
                                torch::Tensor expert = torch::zeros({2});\
                                float dx = tmp_target_pos.first - location.x();\
                                float dy = tmp_target_pos.second - location.y();\
                                float l = sqrtf(dx*dx + dy*dy);\
                                if (l > move_scale_sqrt) {\
                                dx = dx / l * 1.414;\
                                dy = dy / l * 1.414;\
                                }\
                                else {\
                                dx = dx / move_scale;\
                                dy = dy / move_scale;\
                                }\
                                expert[0] = dx;\
                                expert[1] = dy;\
                                expert = expert;

std::shared_ptr<Layer> MoveLayer::forward_impl(const LayerForwardConfig &cfg) {
    CMsgBotWorldState_Unit hero = dotautil::get_hero(cfg.state, cfg.team_id, cfg.player_id);

    const auto& location = hero.location();
    torch::Tensor x = dotautil::state_encoding(cfg.state, cfg.team_id, cfg.player_id);

    auto out = networks.at("actor")->get<Dense>()->forward(x);

    // range (-1,1)
    auto action = torch::tanh(out);

    /*
    auto action_logger = spdlog::get("action_logger");
    action_logger->info("{} first tick\nmy action\n{}\n-------",
                            get_name(), dotautil::torch_to_string(action));
    */

    target_pos = get_move_vec(action);
    dist = sqrtf(target_pos.first * target_pos.first + target_pos.second * target_pos.second);
    target_pos.first += location.x();
    target_pos.second += location.y();
    z = location.z();

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 1500);
    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());
    auto enemy_creeps = dotautil::filter_units_by_team(nearby_units, opposed_team);
    enemy_creeps = dotautil::filter_attackable_units(enemy_creeps);

    std::pair<float, float> tmp_target_pos;
    SAVE_EXPERT_ACTION();

    //action_logger->info("{} first tick\nexpert action\n{}\n-------",
    //                    get_name(), dotautil::torch_to_string(expert_action));

    expert_action.push_back(expert);
    move_action.push_back(action);

    return std::shared_ptr<Layer>();
}

CMsgBotWorldState_Action MoveLayer::get_action(){
    CMsgBotWorldState_Action ret;
    if (dist < 200) {
        ret.set_actiontype(CMsgBotWorldState_Action_Type_DOTA_UNIT_ORDER_MOVE_DIRECTLY);
        auto* target_loc = ret.mutable_movedirectly()->mutable_location();
        target_loc->set_z(0.0);
        target_loc->set_x(target_pos.first);
        target_loc->set_y(target_pos.second);
    }
    else {
        ret.set_actiontype(CMsgBotWorldState_Action_Type_DOTA_UNIT_ORDER_MOVE_TO_POSITION);

        auto* target_loc = ret.mutable_movetolocation()->mutable_location();

        target_loc->set_z(0.0);
        target_loc->set_x(target_pos.first);
        target_loc->set_y(target_pos.second);
    }

    return ret;
}


std::shared_ptr<Layer> MoveLayer::forward_expert(const LayerForwardConfig &cfg) {
    const CMsgBotWorldState_Unit& hero = dotautil::get_hero(cfg.state,
                                                            cfg.team_id, cfg.player_id);
    const auto& location = hero.location();

    std::vector<torch::Tensor> states_cache;
    std::vector<torch::Tensor> expert_action_cache;
    std::vector<torch::Tensor> actor_action_cache;

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 1500);

    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());

    auto enemy_creeps = dotautil::filter_units_by_team(nearby_units, opposed_team);
    enemy_creeps = dotautil::filter_attackable_units(enemy_creeps);

    std::pair<float, float> tmp_target_pos;
    SAVE_EXPERT_ACTION();

    target_pos = tmp_target_pos;
    dist = sqrtf(target_pos.first * target_pos.first + target_pos.second * target_pos.second);
    z = hero.location().z();

    expert_action.push_back(expert);
    move_action.push_back(expert);

    return std::shared_ptr<Layer>();
}

std::shared_ptr<Layer> MoveLayer::forward(const LayerForwardConfig &cfg) {
    ticks.push_back(cfg.tick);
    save_state(cfg);
    if (cfg.expert_action) {
        return forward_expert(cfg);
    }
    return forward_impl(cfg);
}


void MoveLayer::save_state(const LayerForwardConfig &cfg) {
    torch::Tensor x = dotautil::state_encoding(cfg.state, cfg.team_id, cfg.player_id);
    states.push_back(x);
}

MoveLayer::PackedData MoveLayer::get_training_data() {
    PackedData ret;
    if (states.empty()) {
        return ret;
    }
    ret["state"] = torch::stack(states).to(torch::kCUDA);
    ret["expert_action"] = torch::stack(expert_action).to(torch::kCUDA);
    ret["move_action"] = torch::stack(move_action).to(torch::kCUDA);
    return ret;
}

void MoveLayer::train(std::vector<PackedData>& data){
    auto& actor = *networks["actor"]->get<Dense>();
    auto& critic = *networks["critic"]->get<Dense>();
    auto& discriminator = *networks["discriminator"]->get<Dense>();
    torch::optim::SGD actor_optim(actor.parameters(),
                                  torch::optim::SGDOptions(lr));

    torch::optim::SGD critic_optim(critic.parameters(),
                                   torch::optim::SGDOptions(lr));

    torch::optim::SGD d_optim(discriminator.parameters(),
                              torch::optim::SGDOptions(lr));

    auto logger = spdlog::get("loss_logger");

    actor.train(true);
    actor.to(torch::kCUDA);
    critic.train(true);
    critic.to(torch::kCUDA);
    discriminator.train(true);
    discriminator.to(torch::kCUDA);

    torch::Device dev = torch::kCUDA;

    auto avg_total_loss = torch::zeros({1});
    auto avg_actor_loss = torch::zeros({1});
    auto avg_critic_loss = torch::zeros({1});
    auto avg_actor_d_loss = torch::zeros({1});
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
        torch::Tensor expert_act = p_data.at("expert_action").to(dev);
        torch::Tensor move_act = p_data.at("move_action").to(dev);
        torch::Tensor reward = p_data.at("reward").to(dev);

/*
        torch::Tensor expert_prob = torch::sigmoid(discriminator.forward(
                torch::cat({ state, expert_act }, state.dim() - 1)));
        torch::Tensor expert_label = torch::ones_like(expert_prob);
        torch::Tensor expert_d_loss = torch::binary_cross_entropy(expert_prob, expert_label).mean();

        torch::Tensor actor_action_ = torch::tanh(actor.forward(state));
        torch::Tensor actor_prob = torch::sigmoid(discriminator.forward(
                torch::cat({state, actor_action_.detach()}, state.dim() - 1)));
        torch::Tensor actor_label = torch::zeros_like(actor_prob);
        torch::Tensor actor_d_loss = torch::binary_cross_entropy(actor_prob, actor_label).mean();

        torch::Tensor prob_diff = torch::relu(expert_prob - actor_prob).detach();

        torch::Tensor total_d_loss = expert_d_loss + actor_d_loss;

        total_d_loss.backward();

        d_optim.step();
*/

        critic_optim.zero_grad();

        torch::Tensor actor_action_ = torch::tanh(actor.forward(state));

        torch::Tensor state_act = torch::cat({state, actor_action_.detach()}, state.dim() - 1);

        torch::Tensor actor_prob2 = torch::sigmoid(discriminator.forward(
                state_act));
        //torch::Tensor actor_label2 = torch::ones_like(actor_prob2);

        //torch::Tensor actor_d_loss2 = (torch::relu(torch::binary_cross_entropy(actor_prob2, actor_label2) - 0.1))*prob_diff.mean();

        torch::Tensor values = critic.forward(state_act).squeeze();
        torch::Tensor critic_loss = reward - values;
        critic_loss = critic_loss * critic_loss;
        critic_loss = critic_loss.mean();
        critic_loss.backward();
        critic_optim.step();

        avg_critic_loss += critic_loss;

        actor_optim.zero_grad();

        torch::Tensor actor_loss = -critic.forward(torch::cat({state, actor_action_}, state.dim() - 1)).squeeze();
        actor_loss = actor_loss.mean();

        avg_actor_loss += actor_loss;
        //avg_actor_d_loss += actor_d_loss2;


        torch::Tensor total_loss = ((actor_action_ - expert_act)*(actor_action_ - expert_act)).mean();
        avg_total_loss += total_loss;
        total_loss.backward();

        actor_optim.step();
    }

    if (active_data_num) {
        logger->info("episode {} {}, training actor d loss: {} actor loss: {} critic_loss: {} ", 0,
                get_name(), dotautil::torch_to_string(avg_actor_d_loss / active_data_num),
                     dotautil::torch_to_string(avg_actor_loss / active_data_num), dotautil::torch_to_string(avg_critic_loss / active_data_num));

        logger->info("episode {} {}, training total loss: {}", 0, get_name(),
                dotautil::torch_to_string(avg_total_loss / active_data_num));
    }

}

void MoveLayer::reset_custom() {
    expert_action.clear();
    move_action.clear();
    target_pos = std::pair<int, int>();
}


NS_NN_END
