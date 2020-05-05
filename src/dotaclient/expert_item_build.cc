//
// Created by len on 5/3/20.
//

#include "dotaclient/expert_item_build.h"

NS_DOTACLIENT_BEGIN

bool get_expert_item_build(CMsgBotWorldState_Action& action,
                           const CMsgBotWorldState& state,
                           DOTA_TEAM team,
                           int player_id) {
    auto hero = dotautil::get_hero(state, team, player_id);
    for (const auto& item : hero.items()) {
        std::cerr << item.DebugString() << std::endl;
    }
    std::cerr << "remain gold " << hero.unreliable_gold() << std::endl;
    //std::cerr << hero.DebugString() << std::endl;
    action.set_actiontype(CMsgBotWorldState_Action_Type_DOTA_UNIT_ORDER_PURCHASE_ITEM);
    action.mutable_purchaseitem()->set_item(20);
    action.mutable_purchaseitem()->set_item_name("item_circlet");
    return true;
}

NS_DOTACLIENT_END
