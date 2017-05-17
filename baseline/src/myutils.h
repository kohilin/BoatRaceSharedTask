//
// Created by ryosuke-k on 17/05/13.
//

#ifndef BOATRACERANKPREDICTION_MYUTILS_H
#define BOATRACERANKPREDICTION_MYUTILS_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include "str2id.h"
#include "race.h"
#include "mlp.h"


#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/unordered_set.hpp>


Str2Id pIDs;
Str2Id rentan3;
Str2Id renpuku3;
Str2Id rentan2;
Str2Id renpuku2;

std::unordered_map<std::string, unsigned> tan3_map{
        {"1-2-3", 0},
        {"1-2-4", 1},
        {"1-2-5", 2},
        {"1-2-6", 3},
        {"1-3-2", 4},
        {"1-3-4", 5},
        {"1-3-5", 6},
        {"1-3-6", 7},
        {"1-4-2", 8},
        {"1-4-3", 9},
        {"1-4-5", 10},
        {"1-4-6", 11},
        {"1-5-2", 12},
        {"1-5-3", 13},
        {"1-5-4", 14},
        {"1-5-6", 15},
        {"1-6-2", 16},
        {"1-6-3", 17},
        {"1-6-4", 18},
        {"1-6-5", 19},
        {"2-1-3", 20},
        {"2-1-4", 21},
        {"2-1-5", 22},
        {"2-1-6", 23},
        {"2-3-1", 24},
        {"2-3-4", 25},
        {"2-3-5", 26},
        {"2-3-6", 27},
        {"2-4-1", 28},
        {"2-4-3", 29},
        {"2-4-5", 30},
        {"2-4-6", 31},
        {"2-5-1", 32},
        {"2-5-3", 33},
        {"2-5-4", 34},
        {"2-5-6", 35},
        {"2-6-1", 36},
        {"2-6-3", 37},
        {"2-6-4", 38},
        {"2-6-5", 39},
        {"3-1-2", 40},
        {"3-1-4", 41},
        {"3-1-5", 42},
        {"3-1-6", 43},
        {"3-2-1", 44},
        {"3-2-4", 45},
        {"3-2-5", 46},
        {"3-2-6", 47},
        {"3-4-1", 48},
        {"3-4-2", 49},
        {"3-4-5", 50},
        {"3-4-6", 51},
        {"3-5-1", 52},
        {"3-5-2", 53},
        {"3-5-4", 54},
        {"3-5-6", 55},
        {"3-6-1", 56},
        {"3-6-2", 57},
        {"3-6-4", 58},
        {"3-6-5", 59},
        {"4-1-2", 60},
        {"4-1-3", 61},
        {"4-1-5", 62},
        {"4-1-6", 63},
        {"4-2-1", 64},
        {"4-2-3", 65},
        {"4-2-5", 66},
        {"4-2-6", 67},
        {"4-3-1", 68},
        {"4-3-2", 69},
        {"4-3-5", 70},
        {"4-3-6", 71},
        {"4-5-1", 72},
        {"4-5-2", 73},
        {"4-5-3", 74},
        {"4-5-6", 75},
        {"4-6-1", 76},
        {"4-6-2", 77},
        {"4-6-3", 78},
        {"4-6-5", 79},
        {"5-1-2", 80},
        {"5-1-3", 81},
        {"5-1-4", 82},
        {"5-1-6", 83},
        {"5-2-1", 84},
        {"5-2-3", 85},
        {"5-2-4", 86},
        {"5-2-6", 87},
        {"5-3-1", 88},
        {"5-3-2", 89},
        {"5-3-4", 90},
        {"5-3-6", 91},
        {"5-4-1", 92},
        {"5-4-2", 93},
        {"5-4-3", 94},
        {"5-4-6", 95},
        {"5-6-1", 96},
        {"5-6-2", 97},
        {"5-6-3", 98},
        {"5-6-4", 99},
        {"6-1-2", 100},
        {"6-1-3", 101},
        {"6-1-4", 102},
        {"6-1-5", 103},
        {"6-2-1", 104},
        {"6-2-3", 105},
        {"6-2-4", 106},
        {"6-2-5", 107},
        {"6-3-1", 108},
        {"6-3-2", 109},
        {"6-3-4", 110},
        {"6-3-5", 111},
        {"6-4-1", 112},
        {"6-4-2", 113},
        {"6-4-3", 114},
        {"6-4-5", 115},
        {"6-5-1", 116},
        {"6-5-2", 117},
        {"6-5-3", 118},
        {"6-5-4", 119},
};

std::unordered_map<std::string, unsigned> tan2_map{
        {"1-2", 0},
        {"1-3", 1},
        {"1-4", 2},
        {"1-5", 3},
        {"1-6", 4},
        {"2-1", 5},
        {"2-3", 6},
        {"2-4", 7},
        {"2-5", 8},
        {"2-6", 9},
        {"3-1", 10},
        {"3-2", 11},
        {"3-4", 12},
        {"3-5", 13},
        {"3-6", 14},
        {"4-1", 15},
        {"4-2", 16},
        {"4-3", 17},
        {"4-5", 18},
        {"4-6", 19},
        {"5-1", 20},
        {"5-2", 21},
        {"5-3", 22},
        {"5-4", 23},
        {"5-6", 24},
        {"6-1", 25},
        {"6-2", 26},
        {"6-3", 27},
        {"6-4", 28},
        {"6-5", 29},
};


namespace PO = boost::program_options;


namespace myutil {

    inline PO::variables_map InitCommandLine(int argc, char **argv) {

        PO::variables_map conf;
        PO::options_description opts("Configuration options");
        opts.add_options()
                ("train_data,T", PO::value<std::string>(),
                 "Training data, this would be *.dataAll-train-clean.tsv")
                ("dev_data,D", PO::value<std::string>(),
                 "Development data, this would be *.dataAll-dev-clean.tsv")
                ("eval_data,E", PO::value<std::string>(),
                 "Evaluation data, this would be *.dataAll-test-clean.tsv")
                 ("path,P", PO::value<std::string>(),
                 "Path to a directory contains train, dev, enaluation data")
                ("train,t",
                 "Should training be run?")
                ("evaluation,e",
                 "Should evaluation be run?")
                ("model,m", PO::value<std::string>(),
                 "Model")
                ("tan3,s",
                 "Predict 3-Rentan? (default 2-Rentan)")
                ("heuristic_choose,H",
                 "Should heuristic choose be run?")
                ("mlp,M",
                 "Should mlp be run?")
                ("help,h",
                 "show command options");

        PO::options_description dcmdline_options;
        dcmdline_options.add(opts);
        PO::store(parse_command_line(argc, argv, dcmdline_options), conf);

        if (conf.count("help") || argc < 1) {
            std::cerr << dcmdline_options << std::endl;
            exit(1);
        }

        if (conf.count("heuristic_choose") && !conf.count("eval_data") && !conf.count("path")) {
            cerr << "You need to set a data as evaluation data for heuristic_choose option" << endl;
            exit(1);
        }

        if (!conf.count("train_data") && !conf.count("path")){
            cerr << "You need the training data to set player ID in the same, even though you do not train" << endl;
            exit(1);
        }

        if (conf.count("train") && conf.count("evaluation")){
            cerr << "You cannot do training and evaluation on the same time" << endl;
            exit(1);
        }

        return conf;
    }


    std::vector<Race> load_data(std::string filename) {
        std::ifstream ifs(filename);
        if (ifs.fail()) {
            std::cerr << "Failed to read file: " << filename << std::endl;
            exit(1);
        }

        //// set psuedo tokens
        //// 0 : unknown player
        //// 1 : start sequence (i.e. left of 1-goutei)
        //// 2 : end sequence (i.e. right of 6-goutei)
        pIDs.set("UNK");
        pIDs.set("SOS");
        pIDs.set("EOS");
        std::string line;
        std::vector<Race> races;
        while (std::getline(ifs, line)) {
            std::vector<std::string> v_string;
            boost::split(v_string, line, boost::is_any_of("\t"));
            for (auto &p_id:PLAYER_ID_COLS) { v_string[p_id] = std::to_string(pIDs.set(v_string[p_id])); }
//            v_string[TAN3_ID] = std::to_string(rentan3.set(v_string[TAN3_ID]));
            v_string[TAN3_ID] = std::to_string(tan3_map[v_string[TAN3_ID]]);
            v_string[PUKU3_ID] = std::to_string(renpuku3.set(v_string[PUKU3_ID]));
//            v_string[TAN2_ID] = std::to_string(rentan2.set(v_string[TAN2_ID]));
            v_string[TAN2_ID] = std::to_string(tan2_map[v_string[TAN2_ID]]);

            v_string[PUKU2_ID] = std::to_string(renpuku2.set(v_string[PUKU2_ID]));
            races.emplace_back(v_string);
        }
        return races;
    }

    std::pair<std::vector<std::vector<float>>, std::vector<unsigned>>
    get_representations(const std::vector<Race> &races, bool tan3 = false) {
        std::vector<std::vector<float>> x;
        std::vector<unsigned> y;
        for (const auto &race:races) {
            x.emplace_back(race.toVector());
            tan3 ? y.emplace_back(race.res_tan3_) : y.emplace_back(race.res_tan2_);
        }
        return std::make_pair(x, y);
    };

    void run_most_popular(const PO::variables_map &conf) {
        std::vector<Race> dev_races;
        if (conf.count("path")){
                dev_races = myutil::load_data(conf["path"].as<string>() + "/2017-5-17-BR-test-clean.tsv");
        }else{
                dev_races = myutil::load_data(conf["eval_data"].as<string>());
    }
        cerr << "### most_popular_choice : " << endl;

        unsigned tan2_correct(0);
        unsigned tan3_correct(0);
        for (const auto &race:dev_races) {
            auto min_odds_tan2_itr = std::min_element(race.odds_tan2_.begin(), race.odds_tan2_.end());
            auto min_odds_tan3_itr = std::min_element(race.odds_tan3_.begin(), race.odds_tan3_.end());

            unsigned min_odds_tan2_idx = min_odds_tan2_itr - race.odds_tan2_.begin();
            unsigned min_odds_tan3_idx = min_odds_tan3_itr - race.odds_tan3_.begin();
            if (race.res_tan2_ == min_odds_tan2_idx) tan2_correct++;
            if (race.res_tan3_ == min_odds_tan3_idx) tan3_correct++;
        }
        cerr << "# Result :" << "\n"
             << "2-Rentan " << (double) tan2_correct / dev_races.size() << "\n"
             << "3-Rentan " << (double) tan3_correct / dev_races.size() << "\n\n";
    }

    void run_random_choice_from_populars(const PO::variables_map &conf) {
        std::vector<Race> dev_races;
        if (conf.count("path")){
            dev_races = myutil::load_data(conf["path"].as<string>() + "/2017-5-17-BR-test-clean.tsv");
        }else{

             dev_races = myutil::load_data(conf["eval_data"].as<string>());
         }

        cerr << "### random_choice_from_populars : " << endl;

        unsigned UPPER = 10;
        std::vector<double> tan2_correct_avg(UPPER - 2, 0.0);
        std::vector<double> tan3_correct_avg(UPPER - 2, 0.0);
        unsigned ITER = 10;
        for (size_t r = 0; r < ITER; ++r) {
            cerr << "-*- Iteration " << r + 1 << " -*-" << endl;
            for (size_t th = 2; th < UPPER; ++th) {
                unsigned tan2_correct(0);
                unsigned tan3_correct(0);

                for (const auto &race:dev_races) {
                    std::map<unsigned, unsigned> tan2_m;
                    std::map<unsigned, unsigned> tan3_m;

                    for (size_t i = 0; i < race.odds_tan2_.size(); ++i) {
                        tan2_m[static_cast<unsigned>(race.odds_tan2_[i])] = i;
                    }
                    for (size_t i = 0; i < race.odds_tan3_.size(); ++i) {
                        tan3_m[static_cast<unsigned>(race.odds_tan3_[i])] = i;
                    }

                    vector<unsigned> tan2_pops;
                    vector<unsigned> tan3_pops;

                    for (const auto &o:tan2_m) { tan2_pops.push_back(o.second); }
                    for (const auto &o:tan3_m) { tan3_pops.push_back(o.second); }

                    unsigned rand_idx = rand() % th;

                    unsigned tan2_pred = tan2_pops[rand_idx];
                    unsigned tan3_pred = tan3_pops[rand_idx];

                    if (race.res_tan2_ == tan2_pred) tan2_correct++;
                    if (race.res_tan3_ == tan3_pred) tan3_correct++;
                }
                tan2_correct_avg[th - 2] += (double) tan2_correct / dev_races.size();
                tan3_correct_avg[th - 2] += (double) tan3_correct / dev_races.size();
            }
        }
        cerr << "# Result average in 100 iteration :" << "\n";
        for (size_t i = 0; i < UPPER - 2; ++i) {
            cerr << "# from top " << i + 2 << "\n"
                 << "2-Rentan " << tan2_correct_avg[i] / ITER << "\n"
                 << "3-Rentan " << tan3_correct_avg[i] / ITER << "\n\n";
        }
    }

    void run_heuristic_choice(const PO::variables_map &conf) {
        myutil::run_most_popular(conf);
        myutil::run_random_choice_from_populars(conf);
    }


    void run_mlp(int argc, char **argv, const PO::variables_map &conf) {
        unsigned BATCH_SIZE = 128;
        unsigned NUM_EPOCH = 20;


        dynet::initialize(argc, argv);

        std::vector<Race> train_races, dev_races, eval_races;

        if (conf.count("path")){
            string PATH = conf["path"].as<string>();
            train_races = myutil::load_data(PATH + "/2017-5-17-BR-train-clean.tsv");
            dev_races = myutil::load_data(PATH + "/2017-5-17-BR-dev-clean.tsv");
            eval_races = myutil::load_data(PATH + "/2017-5-17-BR-test-clean.tsv");

        }else{
            std::vector<Race> train_races = myutil::load_data(conf["train_data"].as<string>());
            std::vector<Race> dev_races = myutil::load_data(conf["dev_data"].as<string>());
            std::vector<Race> eval_races = myutil::load_data(conf["eval_data"].as<string>());
        }
        std::vector<std::vector<float>> br_train_reps;
        std::vector<unsigned> br_train_ress;
        std::tie(br_train_reps, br_train_ress) = myutil::get_representations(train_races, conf.count("tan3"));

        std::vector<std::vector<float>> br_dev_reps;
        std::vector<unsigned> br_dev_ress;
        std::tie(br_dev_reps, br_dev_ress) = myutil::get_representations(dev_races, conf.count("tan3"));


        std::vector<std::vector<float>> br_eval_reps;
        std::vector<unsigned> br_eval_ress;
        std::tie(br_eval_reps, br_eval_ress) = myutil::get_representations(eval_races, conf.count("tan3"));

        if (conf.count("train")) {

            dynet::Model model;



            // Use Adam optimizer
            dynet::AdamTrainer adam(model);
            adam.clips = 5;
            adam.clip_threshold *= BATCH_SIZE;
            adam.eta_decay = 0.5;

            unsigned P_EMBED_DIM = 100;
            unsigned S_EMBED_DIM = 100;
            unsigned A_EMBED_DIM = 100;
            unsigned G_EMBED_DIM = 100;
            unsigned Se_EMBED_DIM = 100;
            unsigned FIX_VEC_DIM = br_train_reps[0].size();
            unsigned INPUT_DIM = FIX_VEC_DIM + P_EMBED_DIM * 6 + S_EMBED_DIM + A_EMBED_DIM * 6 + G_EMBED_DIM * 6 + Se_EMBED_DIM * 6;
            unsigned HIDDEN_DIM = 216;
            unsigned OUTPUT_DIM = conf.count("tan3") ? 120 : 30;

            // Create model
            MLP nn(model, std::vector<Layer>({
                                                     Layer(/* input_dim */ INPUT_DIM, /* output_dim */ HIDDEN_DIM, /* activation */
                                                                           RELU, /* dropout_rate */ 0.2),
                                                     Layer(/* input_dim */ HIDDEN_DIM, /* output_dim */ HIDDEN_DIM, /* activation */
                                                                           RELU, /* dropout_rate */ 0.2),
                                                     Layer(/* input_dim */ HIDDEN_DIM, /* output_dim */ HIDDEN_DIM, /* activation */
                                                                           RELU, /* dropout_rate */ 0.2),
                                                     Layer(/* input_dim */ HIDDEN_DIM, /* output_dim */ OUTPUT_DIM, /* activation */
                                                                           LINEAR, /* dropout_rate */ 0.0)
                                             }));
            // add player embeddings
            nn.add_embeddings(model, pIDs.size(), P_EMBED_DIM, S_EMBED_DIM, A_EMBED_DIM, G_EMBED_DIM, Se_EMBED_DIM);


            if (conf.count("model")) {
                ifstream in(conf["model"].as<string>());
                boost::archive::text_iarchive ia(in);
                ia >> model >> nn;
            }


            // Initialize variables for training -------------------------------------------------------------
            // Worst accuracy
            double worst = 0;

            // Number of batches in training set
            unsigned num_batches = br_train_reps.size() / BATCH_SIZE - 1;

            // Random indexing
            unsigned si;
            std::vector<unsigned> order(num_batches);
            for (unsigned i = 0; i < num_batches; ++i) order[i] = i;

            bool first = true;
            unsigned epoch = 0;
            std::vector<Expression> cur_batch;
            std::vector<unsigned> cur_labels;
            std::vector<std::vector<unsigned>> cur_players;

            // model file name
            std::string fname = conf.count("tan3") ? "tan3_mlp.params" : "tan2_mlp.params";

            // Run for the given number of epochs (or indefinitely if params.NUM_EPOCHS is negative)
            while (epoch < NUM_EPOCH || NUM_EPOCH < 0) {
                // Update the optimizer
                if (first) { first = false; } else { adam.update_epoch(); }
                // Reshuffle the dataset
                cerr << "**SHUFFLE\n";
                random_shuffle(order.begin(), order.end());
                // Initialize loss and number of samples processed (to average loss)
                double loss = 0;
                double num_samples = 0;

                // Start timer
                Timer *iteration = new Timer("completed in");

                // Activate dropout
                nn.enable_dropout();


                for (si = 0; si < num_batches; ++si) {
                    // build graph for this instance
                    ComputationGraph cg;
                    // Compute batch start id and size
                    int id = order[si] * BATCH_SIZE;
                    unsigned bsize = std::min((unsigned) br_train_reps.size() - id, BATCH_SIZE);
                    // Get input batch
                    cur_batch = std::vector<Expression>(bsize);
                    cur_labels = std::vector<unsigned>(bsize);
                    cur_players = std::vector<std::vector<unsigned>>(bsize);
                    for (unsigned idx = 0; idx < bsize; ++idx) {
                        std::vector<Player> &ps = train_races[id + idx].players_;
                        Expression p_embs = concatenate({
                                                                lookup(cg, nn.p_player, ps[0].id_),
                                                                lookup(cg, nn.p_player, ps[1].id_),
                                                                lookup(cg, nn.p_player, ps[2].id_),
                                                                lookup(cg, nn.p_player, ps[3].id_),
                                                                lookup(cg, nn.p_player, ps[4].id_),
                                                                lookup(cg, nn.p_player, ps[5].id_),
                                                                lookup(cg, nn.p_grade, ps[0].grade_),
                                                                lookup(cg, nn.p_grade, ps[1].grade_),
                                                                lookup(cg, nn.p_grade, ps[2].grade_),
                                                                lookup(cg, nn.p_grade, ps[3].grade_),
                                                                lookup(cg, nn.p_grade, ps[4].grade_),
                                                                lookup(cg, nn.p_grade, ps[5].grade_),
                                                                lookup(cg, nn.p_stadium, train_races[id + idx].stadium_),
                                                                lookup(cg, nn.p_affiliation, ps[0].affiliation_),
                                                                lookup(cg, nn.p_affiliation, ps[1].affiliation_),
                                                                lookup(cg, nn.p_affiliation, ps[2].affiliation_),
                                                                lookup(cg, nn.p_affiliation, ps[3].affiliation_),
                                                                lookup(cg, nn.p_affiliation, ps[4].affiliation_),
                                                                lookup(cg, nn.p_affiliation, ps[5].affiliation_),
                                                                lookup(cg, nn.p_sex, ps[0].sex_),
                                                                lookup(cg, nn.p_sex, ps[1].sex_),
                                                                lookup(cg, nn.p_sex, ps[2].sex_),
                                                                lookup(cg, nn.p_sex, ps[3].sex_),
                                                                lookup(cg, nn.p_sex, ps[4].sex_),
                                                                lookup(cg, nn.p_sex, ps[5].sex_)
                                                        });
                        Expression fix_vec = input(cg, {FIX_VEC_DIM}, br_train_reps[id + idx]);
                        cur_batch[idx] = concatenate({fix_vec, p_embs});
                        cur_labels[idx] = br_train_ress[id + idx];
                    }
                    // Reshape as batch (not very intuitive yet)
                    Expression x_batch = reshape(concatenate_cols(cur_batch), Dim({INPUT_DIM}, bsize));
                    // Get negative log likelihood on batch
                    Expression loss_expr = nn.get_nll(x_batch, cur_labels, cg);
                    // Get scalar error for monitoring
                    loss += as_scalar(cg.forward(loss_expr));
                    // Increment number of samples processed
                    num_samples += bsize;
                    // Compute gradient with backward pass
                    cg.backward(loss_expr);
                    // Update parameters
                    adam.update(1.0);
                    // Print progress every tenth of the dataset
                    if ((si + 1) % (num_batches / 10) == 0 || si == num_batches - 1) {
                        // Print informations
                        adam.status();
                        cerr << " E = " << (loss / num_samples) << ' ';
                        // Reinitialize timer
                        delete iteration;
                        iteration = new Timer("completed in");
                        // Reinitialize loss
                        loss = 0;
                        num_samples = 0;
                    }
                }

                // Disable dropout for dev testing
                nn.disable_dropout();

                // Show score on dev data
                if (si == num_batches) {
                    double dpos = 0;
                    for (unsigned i = 0; i < br_dev_reps.size(); ++i) {
                        // build graph for this instance
                        ComputationGraph cg;
                        // Get input expression
                        std::vector<unsigned> p_ids = dev_races[i].player_ids_;
                        std::vector<Player> &ps = dev_races[i].players_;
                        Expression p_embs = concatenate({
                                                                lookup(cg, nn.p_player, ps[0].id_),
                                                                lookup(cg, nn.p_player, ps[1].id_),
                                                                lookup(cg, nn.p_player, ps[2].id_),
                                                                lookup(cg, nn.p_player, ps[3].id_),
                                                                lookup(cg, nn.p_player, ps[4].id_),
                                                                lookup(cg, nn.p_player, ps[5].id_),
                                                                lookup(cg, nn.p_grade, ps[0].grade_),
                                                                lookup(cg, nn.p_grade, ps[1].grade_),
                                                                lookup(cg, nn.p_grade, ps[2].grade_),
                                                                lookup(cg, nn.p_grade, ps[3].grade_),
                                                                lookup(cg, nn.p_grade, ps[4].grade_),
                                                                lookup(cg, nn.p_grade, ps[5].grade_),
                                                                lookup(cg, nn.p_stadium, dev_races[i].stadium_),
                                                                lookup(cg, nn.p_affiliation, ps[0].affiliation_),
                                                                lookup(cg, nn.p_affiliation, ps[1].affiliation_),
                                                                lookup(cg, nn.p_affiliation, ps[2].affiliation_),
                                                                lookup(cg, nn.p_affiliation, ps[3].affiliation_),
                                                                lookup(cg, nn.p_affiliation, ps[4].affiliation_),
                                                                lookup(cg, nn.p_affiliation, ps[5].affiliation_),
                                                                lookup(cg, nn.p_sex, ps[0].sex_),
                                                                lookup(cg, nn.p_sex, ps[1].sex_),
                                                                lookup(cg, nn.p_sex, ps[2].sex_),
                                                                lookup(cg, nn.p_sex, ps[3].sex_),
                                                                lookup(cg, nn.p_sex, ps[4].sex_),
                                                                lookup(cg, nn.p_sex, ps[5].sex_)
                                                        });
                        Expression fix_vec = input(cg, {FIX_VEC_DIM}, br_dev_reps[i]);
                        Expression x = concatenate({fix_vec, p_embs});
                        // Get negative log likelihood on batch
                        unsigned predicted_idx = nn.predict(x, cg);
                        // Increment count of positive classification
                        if (predicted_idx == br_dev_ress[i]) dpos++;
                    }
                    // Print informations
                    cerr << "\n***DEV [epoch=" << (epoch)
                         << "] Accuracy = " << (dpos / (double) br_dev_reps.size()) << ' ';
                    // Reinitialize timer
                    delete iteration;
                    iteration = new Timer("completed in");

                    //                 If the dev loss is lower than the previous ones, save the ,odel
                    if (dpos > worst) {
                        cerr << "Update model" << endl;
                        worst = dpos;
                        ofstream out(fname);
                        boost::archive::text_oarchive oa(out);
                        oa << model << nn;
                    }

                }

                // Increment epoch
                ++epoch;

            }
        }

        if (conf.count("evaluation")) {

            dynet::Model model;

            // Use Adam optimizer
            dynet::AdamTrainer adam(model);
            adam.clips = 5;
            adam.clip_threshold *= BATCH_SIZE;
            adam.eta_decay = 0.5;

            unsigned P_EMBED_DIM = 100;
            unsigned S_EMBED_DIM = 100;
            unsigned A_EMBED_DIM = 100;
            unsigned G_EMBED_DIM = 100;
            unsigned Se_EMBED_DIM = 100;
            unsigned FIX_VEC_DIM = br_eval_reps[0].size();
            unsigned INPUT_DIM = FIX_VEC_DIM + P_EMBED_DIM * 6 + S_EMBED_DIM + A_EMBED_DIM * 6 + G_EMBED_DIM * 6 + Se_EMBED_DIM * 6;
            unsigned HIDDEN_DIM = 216;
            unsigned OUTPUT_DIM = conf.count("tan3") ? 120 : 30;

            // Create model
            MLP nn(model, std::vector<Layer>({
                                                     Layer(/* input_dim */ INPUT_DIM, /* output_dim */ HIDDEN_DIM, /* activation */
                                                                           RELU, /* dropout_rate */ 0.2),
                                                     Layer(/* input_dim */ HIDDEN_DIM, /* output_dim */ HIDDEN_DIM, /* activation */
                                                                           RELU, /* dropout_rate */ 0.2),
                                                     Layer(/* input_dim */ HIDDEN_DIM, /* output_dim */ HIDDEN_DIM, /* activation */
                                                                           RELU, /* dropout_rate */ 0.2),
                                                     Layer(/* input_dim */ HIDDEN_DIM, /* output_dim */ OUTPUT_DIM, /* activation */
                                                                           LINEAR, /* dropout_rate */ 0.0)
                                             }));
            // add player embeddings
            nn.add_embeddings(model, pIDs.size(), P_EMBED_DIM, S_EMBED_DIM, A_EMBED_DIM, G_EMBED_DIM, Se_EMBED_DIM);

            if (conf.count("model")) {
                ifstream in(conf["model"].as<string>());
                boost::archive::text_iarchive ia(in);
                ia >> model >> nn;
            }



            // Show score on dev data
            double dpos = 0;
            for (unsigned i = 0; i < br_eval_reps.size(); ++i) {
                // build graph for this instance
                ComputationGraph cg;
                // Get input expression
                std::vector<unsigned> p_ids = eval_races[i].player_ids_;
                std::vector<Player> &ps = eval_races[i].players_;
                Expression p_embs = concatenate({
                                                        lookup(cg, nn.p_player, ps[0].id_),
                                                        lookup(cg, nn.p_player, ps[1].id_),
                                                        lookup(cg, nn.p_player, ps[2].id_),
                                                        lookup(cg, nn.p_player, ps[3].id_),
                                                        lookup(cg, nn.p_player, ps[4].id_),
                                                        lookup(cg, nn.p_player, ps[5].id_),
                                                        lookup(cg, nn.p_grade, ps[0].grade_),
                                                        lookup(cg, nn.p_grade, ps[1].grade_),
                                                        lookup(cg, nn.p_grade, ps[2].grade_),
                                                        lookup(cg, nn.p_grade, ps[3].grade_),
                                                        lookup(cg, nn.p_grade, ps[4].grade_),
                                                        lookup(cg, nn.p_grade, ps[5].grade_),
                                                        lookup(cg, nn.p_stadium, eval_races[i].stadium_),
                                                        lookup(cg, nn.p_affiliation, ps[0].affiliation_),
                                                        lookup(cg, nn.p_affiliation, ps[1].affiliation_),
                                                        lookup(cg, nn.p_affiliation, ps[2].affiliation_),
                                                        lookup(cg, nn.p_affiliation, ps[3].affiliation_),
                                                        lookup(cg, nn.p_affiliation, ps[4].affiliation_),
                                                        lookup(cg, nn.p_affiliation, ps[5].affiliation_),
                                                        lookup(cg, nn.p_sex, ps[0].sex_),
                                                        lookup(cg, nn.p_sex, ps[1].sex_),
                                                        lookup(cg, nn.p_sex, ps[2].sex_),
                                                        lookup(cg, nn.p_sex, ps[3].sex_),
                                                        lookup(cg, nn.p_sex, ps[4].sex_),
                                                        lookup(cg, nn.p_sex, ps[5].sex_)
                                                });
                Expression fix_vec = input(cg, {FIX_VEC_DIM}, br_eval_reps[i]);
                Expression x = concatenate({fix_vec, p_embs});
                unsigned predicted_idx = nn.predict(x, cg);
                if (predicted_idx == br_eval_ress[i]) dpos++;
            }
            cerr << "\n***EVALUATION Accuracy = " << (dpos / (double) br_eval_reps.size()) << endl;
        }
    }

}

#endif //BOATRACERANKPREDICTION_MYUTILS_H
