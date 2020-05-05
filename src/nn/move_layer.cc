//
// Created by len on 7/7/19.
//

#include "nn/move_layer.h"
#include "util/util.h"

#include <exception>
#include <fstream>

NS_NN_BEGIN

const static int input_shape = hero_state_size;
const static int hidden_shape = 256;
const static int output_shape = 2;

const static float max_map_size = 8000;
const static float move_scale = 1500;
const static float move_scale_sqrt = 2121.32;

MoveLayer::MoveLayer(): dist(0), z(0) {
    networks["actor"] = std::dynamic_pointer_cast<TorchLayer>(std::make_shared<LSTM>(input_shape, hidden_shape, output_shape));
    networks["hp_critic"] = std::dynamic_pointer_cast<TorchLayer>(std::make_shared<LSTM_DENSE>(input_shape, output_shape, hidden_shape, 1));
    networks["lasthit_critic"] = std::dynamic_pointer_cast<TorchLayer>(std::make_shared<LSTM_DENSE>(input_shape, output_shape, hidden_shape, 1));
    networks["discriminator"] = std::dynamic_pointer_cast<TorchLayer>(std::make_shared<Dense>(input_shape + output_shape, hidden_shape, 1));
}

static std::pair<float, float> get_move_vec(torch::Tensor x) {
    x = x * move_scale;
    std::pair<float, float> out_(dotautil::to_number<float>(x[0]), dotautil::to_number<float>(x[1]));
    return out_;
}

static float atk_range = 300.0f;// get closer
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
                                expert[1] = dy;

std::shared_ptr<Layer> MoveLayer::forward_impl(const LayerForwardConfig &cfg) {
    CMsgBotWorldState_Unit hero = dotautil::get_hero(cfg.state, cfg.team_id, cfg.player_id);

    const auto& location = hero.location();

    auto nearby_units = dotautil::get_nearby_unit(cfg.state, hero, 1500);
    uint32_t opposed_team = dotautil::get_opposed_team(hero.team_id());
    auto enemy_creeps = dotautil::filter_units_by_team(nearby_units, opposed_team);
    enemy_creeps = dotautil::filter_attackable_units(enemy_creeps);

    std::pair<float, float> tmp_target_pos;
    SAVE_EXPERT_ACTION();


    if (states.empty()) {
        // not enough data, continue;
        target_pos.first = 0;
        target_pos.second = 0;
        expert_action_buffer.push_back(expert);
        move_action_buffer.push_back(torch::zeros({1,2}));
        return std::shared_ptr<Layer>();
    }

    torch::Tensor lstm_input = states.back();

    auto out = networks.at("actor")->get<LSTM>()->forward(lstm_input);

    // range (-1,1)
    auto action = torch::tanh(out[0][0]);

    target_pos = get_move_vec(action);
    dist = sqrtf(target_pos.first * target_pos.first + target_pos.second * target_pos.second);
    target_pos.first += location.x();
    target_pos.second += location.y();
    z = location.z();


    expert_action_buffer.push_back(expert);
    move_action_buffer.push_back(action);
    if (expert_action_buffer.size() > n_step) {
        expert_action_buffer.pop_front();
        move_action_buffer.pop_front();
    }

    torch::Tensor expert_act = torch::zeros({n_step,1, output_shape});
    torch::Tensor move_act = torch::zeros({n_step,1, output_shape});

    for (int i = 0;i < n_step; ++i) {
        expert_act[i] = expert_action_buffer[i].reshape({1, -1});
        move_act[i] = move_action_buffer[i].reshape({1, -1});
    }

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

    if (!enemy_creeps.empty()) {
        auto target_unit = dotautil::get_nearest_unit(enemy_creeps, hero);
        float distance = dotautil::get_unit_distance(target_unit, hero);
        std::stringstream ss;
        ss << "expert move dist "<< distance << " hero location x "
           << location.x() << ", " << location.y() << " creep location: "
           << target_unit.location().x() << ", " << target_unit.location().y()
           << " expert move target " << tmp_target_pos.first << ", "
           << tmp_target_pos.second << " target type " << target_unit.unit_type();
        //CMsgBotWorldState_UnitType_LANE_CREEP
        action_logger->info(ss.str().c_str());
    }

    target_pos = tmp_target_pos;
    //std::cerr << "expert target pos " << target_pos.first << "," << target_pos.second << std::endl;
    dist = sqrtf(target_pos.first * target_pos.first + target_pos.second * target_pos.second);
    z = hero.location().z();

    if (!states.empty()) {
        expert_action.push_back(expert);
        move_action.push_back(expert);
    }
    return std::shared_ptr<Layer>();
}

std::shared_ptr<Layer> MoveLayer::forward(const LayerForwardConfig &cfg) {
    ticks.push_back(cfg.tick);
    save_state(cfg);
    if (cfg.expert_action) {
        expert_mode = true;
        return forward_expert(cfg);
    }
    return forward_impl(cfg);
}


void MoveLayer::save_state(const LayerForwardConfig &cfg) {
    torch::Tensor x = dotautil::state_encoding(cfg.state, cfg.team_id, cfg.player_id);

    state_buffer.push_back(x);
    if (state_buffer.size() > n_step) {
        state_buffer.pop_front();
    }

    if (state_buffer.size() < n_step) {
        // not enough data, continue;
        return;
    }

    torch::Tensor lstm_input = torch::zeros({n_step,1, input_shape});
    for (int i = 0;i < n_step; ++i) {
        lstm_input[i] = state_buffer[i].reshape({1, -1});
    }
    states.push_back(lstm_input);
}

MoveLayer::PackedData MoveLayer::get_training_data() {
    PackedData ret;
    if (states.empty()) {
        return ret;
    }
    ret["state"] = torch::cat(states,1).to(torch::kCUDA);
    ret["expert_action"] = torch::stack(expert_action).to(torch::kCUDA);
    ret["move_action"] = torch::stack(move_action).to(torch::kCUDA);
    if (expert_mode) {
        ret["expert_mode"] = torch::Tensor();
    }
    return ret;
}

void MoveLayer::train(std::vector<PackedData>& data){
    std::lock_guard<std::mutex> g(mtx);
    auto& actor = *networks["actor"]->get<LSTM>();
    auto& hp_critic = *networks["hp_critic"]->get<LSTM_DENSE>();
    auto& lasthit_critic = *networks["lasthit_critic"]->get<LSTM_DENSE>();
    auto& discriminator = *networks["discriminator"]->get<Dense>();
    torch::optim::SGD actor_optim(actor.parameters(),
                                  torch::optim::SGDOptions(lr));

    torch::optim::SGD hp_critic_optim(hp_critic.parameters(),
                                   torch::optim::SGDOptions(lr));
    torch::optim::SGD lasthit_critic_optim(lasthit_critic.parameters(),
                                   torch::optim::SGDOptions(lr));

    torch::optim::SGD d_optim(discriminator.parameters(),
                              torch::optim::SGDOptions(lr));

    auto logger = spdlog::get("loss_logger");

    actor.train(true);
    actor.to(torch::kCUDA);
    hp_critic.train(true);
    lasthit_critic.train(true);
    hp_critic.to(torch::kCUDA);
    lasthit_critic.to(torch::kCUDA);
    discriminator.train(true);
    discriminator.to(torch::kCUDA);

    torch::Device dev = torch::kCUDA;

    auto avg_total_loss = torch::zeros({1});
    auto avg_actor_loss = torch::zeros({1});
    auto avg_critic_loss = torch::zeros({1});
    auto avg_actor_d_loss = torch::zeros({1});
    int active_data_num = 0;

    auto critic_training_fn = [](LSTM_DENSE& critic,
            torch::Tensor lstm_state,
            torch::Tensor dense_state,
            torch::Tensor reward,
            torch::optim::SGD& optim) ->torch::Tensor{
        optim.zero_grad();
        torch::Tensor values = critic.forward(lstm_state, dense_state).squeeze();
        torch::Tensor critic_loss = reward - values;
        critic_loss = critic_loss * critic_loss;
        critic_loss = critic_loss.mean();
        critic_loss.backward();
        optim.step();
        return critic_loss;
    };

    for (const auto& p_data:data) {
        actor_optim.zero_grad();
        hp_critic_optim.zero_grad();
        lasthit_critic_optim.zero_grad();
        d_optim.zero_grad();

        if (p_data.count("state") == 0) {
            continue;
        }
        active_data_num++;


        torch::Tensor state = p_data.at("state").to(dev);
        torch::Tensor expert_act = p_data.at("expert_action").to(dev);
        torch::Tensor move_act = p_data.at("move_action").to(dev);
        torch::Tensor hp_reward = p_data.at(dotautil::reward_hp_key).to(dev);
        torch::Tensor lasthit_reward = p_data.at(dotautil::reward_lasthit_key).to(dev);

        torch::Tensor actor_action_ = torch::tanh(actor.forward(state)[0]);

        int start = (int)(hp_reward.size(0) - actor_action_.size(0));

        // TODO should remove first 9 elements
        hp_reward = hp_reward.slice(0, start);
        lasthit_reward = lasthit_reward.slice(0, start);
        //expert_act = expert_act.slice(0, start);
        
        /*
        std::cerr << "state shape " << state.sizes() << std::endl;
        std::cerr << "move act shape " << move_act.sizes() << std::endl;
        std::cerr << "action shape " << actor_action_.sizes() << std::endl;
        std::cerr << "lasthit reward shape " << lasthit_reward.sizes() << std::endl;
        std::cerr << "hp_reward shape " << hp_reward.sizes() << std::endl;

        //expert_act = expert_act.slice(1, -1).reshape({-1,2});
        std::cerr << "expert_act shape " << expert_act.sizes() << std::endl;
        */
         
        //torch::Tensor state_act = torch::cat({state, actor_action_.detach()}, state.dim() - 1);
        //torch::Tensor state_act = torch::cat({state, move_act}, state.dim() - 1);


        avg_critic_loss += critic_training_fn(hp_critic, state, actor_action_.detach(), hp_reward, hp_critic_optim);
        avg_critic_loss += critic_training_fn(lasthit_critic, state, actor_action_.detach(), lasthit_reward, lasthit_critic_optim);

        actor_optim.zero_grad();

        torch::Tensor actor_loss = -hp_critic.forward(state, actor_action_).squeeze().mean();
        actor_loss += -lasthit_critic.forward(state, actor_action_).squeeze().mean();

        avg_actor_loss += actor_loss;
        //avg_actor_d_loss += actor_d_loss2;


        //torch::Tensor total_loss = ((actor_action_ - expert_act)*(actor_action_ - expert_act)).mean();
        torch::Tensor total_loss = actor_loss;
        total_loss += actor_loss;
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
