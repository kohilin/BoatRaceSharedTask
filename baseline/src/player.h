//
// Created by ryosuke-k on 17/05/13.
//

#ifndef BOATRACERANKPREDICTION_PLAYER_H
#define BOATRACERANKPREDICTION_PLAYER_H

#include <vector>
#include <string>

const static std::vector<unsigned> PLAYER_ID_COLS{0, 15, 30, 45, 60, 75};
const static std::vector<unsigned> GRADE_COLS = {1, 16, 31, 46, 61, 76};
const static std::vector<unsigned> AFFILIATION_COLS = {2, 17, 32, 47, 62, 77};
const static std::vector<unsigned> BIRTHPLACE_COLS{3, 18, 33, 48, 63, 78};
const static std::vector<unsigned> AGE_COLS{4, 19, 34, 49, 64, 79};
const static std::vector<unsigned> SEX_COLS{5, 20, 35, 50, 65, 80};
const static std::vector<unsigned> ST_COLS{6, 21, 36, 51, 66, 81};
const static std::vector<unsigned> TOP_ALL_COLS{7, 22, 37, 52, 67, 82};
const static std::vector<unsigned> RENTAI_ALL_COLS{8, 23, 38, 53, 68, 83};
const static std::vector<unsigned> TOP_HERE_COLS{9, 24, 39, 54, 69, 84};
const static std::vector<unsigned> RENTAI_HERE_COLS{10, 25, 40, 55, 70, 85};
const static std::vector<unsigned> MOTOR_ID_COLS{11, 26, 41, 56, 71, 86};
const static std::vector<unsigned> MOTOR_RENTAI_COLS{12, 27, 42, 57, 72, 87};
const static std::vector<unsigned> BOAT_ID_COLS{13, 28, 43, 58, 73, 88};
const static std::vector<unsigned> BOAT_RENTAI_COLS{14, 29, 44, 57, 74, 89};


class Player {
public:
    inline Player(const std::vector<std::string> &race_vec, unsigned boat_num) :
            id_(stoi(race_vec[PLAYER_ID_COLS[boat_num]])),
            grade_(stoi(race_vec[GRADE_COLS[boat_num]])),
            affiliation_(stoi(race_vec[AFFILIATION_COLS[boat_num]])),
            birth_(stoi(race_vec[BIRTHPLACE_COLS[boat_num]])),
            age_(stoi(race_vec[AGE_COLS[boat_num]])),
            sex_(stoi(race_vec[SEX_COLS[boat_num]])),
            avg_st_(stof(race_vec[ST_COLS[boat_num]])),
            top_ratio_all_(stof(race_vec[TOP_ALL_COLS[boat_num]])),
            rentai_ratio_all_(stof(race_vec[RENTAI_ALL_COLS[boat_num]])),
            top_ratio_here_(stof(race_vec[TOP_HERE_COLS[boat_num]])),
            rentai_ratio_here_(stof(race_vec[RENTAI_HERE_COLS[boat_num]])),
            motor_rentai_ratio_(stof(race_vec[MOTOR_RENTAI_COLS[boat_num]])),
            boat_rentai_ratio_(stof(race_vec[BOAT_RENTAI_COLS[boat_num]])) {

        info_vec_ = {
                static_cast<float>(age_),
                avg_st_,
                top_ratio_all_,
                rentai_ratio_all_,
                top_ratio_here_,
                rentai_ratio_here_,
                motor_rentai_ratio_,
                boat_rentai_ratio_
        };

    };

    unsigned id_;
    unsigned grade_;
    unsigned affiliation_;
    unsigned birth_;
    unsigned age_;
    unsigned sex_;
    float avg_st_;
    float top_ratio_all_;
    float rentai_ratio_all_;
    float top_ratio_here_;
    float rentai_ratio_here_;
    float motor_rentai_ratio_;
    float boat_rentai_ratio_;

    std::vector<float> info_vec_;

private:

};


#endif //BOATRACERANKPREDICTION_PLAYER_H
