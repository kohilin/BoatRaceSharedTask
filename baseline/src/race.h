//
// Created by ryosuke-k on 17/05/13.
//

#ifndef BOATRACERANKPREDICTION_RACE_H
#define BOATRACERANKPREDICTION_RACE_H

#include <vector>
#include <string>
#include <algorithm>


#include "player.h"
const static unsigned START_RENTAN3 = 90;
const static unsigned END_RENTAN3 = 210;
const static unsigned START_RENPUKU3 = 210;
const static unsigned END_RENPUKU3 = 231;
const static unsigned START_RENTAN2 = 231;
const static unsigned END_RENTAN2 = 261;
const static unsigned START_RENPUKU2 = 261;
const static unsigned END_RENPUKU2 = 275;
const static unsigned STADIUM_ID_COLS = 275;

const static unsigned TAN3_ID = 276;
const static unsigned PUKU3_ID = 277;
const static unsigned TAN2_ID = 278;
const static unsigned PUKU2_ID = 279;


class Race {
public:
    Race(std::vector<std::string> race_vec) :

            players_(
                    {
                            Player(race_vec, 0),
                            Player(race_vec, 1),
                            Player(race_vec, 2),
                            Player(race_vec, 3),
                            Player(race_vec, 4),
                            Player(race_vec, 5),

                    }
            ),

            player_ids_(
                    {
                            players_[0].id_,
                            players_[1].id_,
                            players_[2].id_,
                            players_[3].id_,
                            players_[4].id_,
                            players_[5].id_
                    }
            ),

            stadium_(stoi(race_vec[STADIUM_ID_COLS])),

            odds_tan3_(Odds2Float(race_vec, START_RENTAN3, END_RENTAN3)),
            odds_puku3_(Odds2Float(race_vec, START_RENPUKU3, END_RENPUKU3)),
            odds_tan2_(Odds2Float(race_vec, START_RENTAN2, END_RENTAN2)),
            odds_puku2_(Odds2Float(race_vec, START_RENPUKU2, END_RENPUKU2)),

            res_tan3_(stoi(race_vec[TAN3_ID])),
            res_puku3_(stoi(race_vec[PUKU2_ID])),
            res_tan2_(stoi(race_vec[TAN2_ID])),
            res_puku2_(stoi(race_vec[PUKU2_ID]))

    {   }


    std::vector<Player> players_;
    std::vector<unsigned> player_ids_;
    std::vector<float> odds_tan3_;
    std::vector<float> odds_puku3_;
    std::vector<float> odds_tan2_;
    std::vector<float> odds_puku2_;
    unsigned stadium_;

    unsigned res_tan3_;
    unsigned res_puku3_;
    unsigned res_tan2_;
    unsigned res_puku2_;

    std::vector<float> toVector() const {
        std::vector<float> rtn;
        for(const auto& p:players_){ std::for_each(p.info_vec_.begin(), p.info_vec_.end(), [&](float x){rtn.emplace_back(x); }); }

        const std::vector<float>& odds = odds_tan3_;
        float threshold = 100;
        std::for_each(odds_tan3_.begin(), odds_tan3_.end(), [&](float x){ x < threshold ? rtn.emplace_back(x) : rtn.emplace_back(0); });
        std::for_each(odds_tan2_.begin(), odds_tan2_.end(), [&](float x){ x < threshold ? rtn.emplace_back(x) : rtn.emplace_back(0); });
        std::for_each(odds_puku3_.begin(), odds_puku3_.end(), [&](float x){ x < threshold ? rtn.emplace_back(x) : rtn.emplace_back(0); });
        std::for_each(odds_puku2_.begin(), odds_puku2_.end(), [&](float x){ x < threshold ? rtn.emplace_back(x) : rtn.emplace_back(0); });

        return rtn;
    }


private:
    std::vector<float> Odds2Float(std::vector<std::string> v, unsigned start, unsigned end){
        std::vector<float> odds_v;
        for(size_t i = start; i < end; ++i){odds_v.emplace_back(stof(v[i])); }
        return odds_v;
    }

};

#endif //BOATRACERANKPREDICTION_RACE_H
