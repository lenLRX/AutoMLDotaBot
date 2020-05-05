//
// Created by len on 5/3/20.
//

#ifndef AUTOMLDOTABOT_EXPERT_ITEM_BUILD_H
#define AUTOMLDOTABOT_EXPERT_ITEM_BUILD_H

#include "util/util.h"

NS_DOTACLIENT_BEGIN

bool get_expert_item_build(CMsgBotWorldState_Action& action,
                           const CMsgBotWorldState& state,
                           DOTA_TEAM team,
                           int player_id);

NS_DOTACLIENT_END

#endif //AUTOMLDOTABOT_EXPERT_ITEM_BUILD_H
