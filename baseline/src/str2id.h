//
// Created by ryosuke-k on 17/01/30.
//

#include <unordered_map>
#include <algorithm>
#include <string>
#include <iostream>

#ifndef NONPROJECTIVETRANSITIONBASEDPARSER_STR2ID_H
#define NONPROJECTIVETRANSITIONBASEDPARSER_STR2ID_H


class Str2Id {
public:
    inline unsigned set(std::string str) {
        auto itr = str2id_.find(str);
        if (itr != str2id_.end()){
            return itr->second;
        }else{
            int new_id = str2id_.size();
            str2id_[str] = new_id;
            id2str_[new_id] = str;
            return new_id;
        }
    }

    inline std::string getString(int n) { return id2str_[n]; }

    inline unsigned getId(std::string str) {
        auto itr = str2id_.find(str);
        return itr == str2id_.end() ?  0 : itr->second;
    }

    inline unsigned size(){ return str2id_.size();}

private:
    std::unordered_map<std::string, unsigned> str2id_;
    std::unordered_map<unsigned, std::string> id2str_;
};



#endif //NONPROJECTIVETRANSITIONBASEDPARSER_STR2ID_H
