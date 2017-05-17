# Baseline Boat Race Predictor
## Requirements
- gcc (version 5.4.0 or higher)
- cmake (version 3.5 or higher)
- boost (version 1.58.0 or higher)
- dynet (https://github.com/clab/dynet)
- eigen
> Please build dynet beforehand (http://dynet.readthedocs.io/en/latest/install.html)

## Build
- cd baseline
- mkdir build && cd build
- cmake .. -DEIGEN3_INCLUDE_DIR=/your/path/to/eigen -DDYNET_INCLUDE_DIR=/your/path/to/dynet
- make -j 2

> If you want to use a specific boost library, you can set it with -DBOOST_ROOT=/your/path/to/boost


## How to run
### Run the baseline models
- ./BaselineBoatRacePredictor -H -P /your/path/to/data (for heuristic choose)
- ./BaselineBoatRacePredictor -M -e -P /your/path/to/data --model ../model/tan2_mlp.params (for predict 2-Rentan)
- ./BaselineBoatRacePredictor -M -e -P /your/path/to/data --model ../model/tan3_mlp.params -s (for predict 3-Rentan)
> A directory given with -P option has to include train, dev, test files.

> You can also set data respectively with -T -D -E options, but the system needs all files regardless of training or evaluation

### Train multi layer perceptron
- ./BaselineBoatRacePredictor -M -t -P /your/path/to/data -D dev.tsv
> Sorry still not support to adjust hyper parameter flexibly, but I will do sooner or later

## Results of the baselines
- Train data : 2017-5-17-BR-train-clean.tsv
- Dev data   : 2017-5-17-BR-dev-clean.tsv
- Test data  : 2017-5-17-BR-test-clean.tsv

### Heuristic Choose
#### Most Popular Choice 
- 2-Rentan : 5.3%
- 3-Rentan : 9.1%

#### Random Choice from Populars
From top 3
- 2-Rentan : 7.6%
- 3-Rentan : 7.3%

From top 5
- 2-Rentan : 7.2%
- 3-Rentan : 6.5%

From top 9
- 2-Rentan : 6.1%
- 3-Rentan : 5.1%
> Average scores for 10 iterations


### Multi Layer Perceptron
- 2-Rentan : 22.3%
- 3-Rentan : 8.1%
> See http://ryosuke-k.chobi.net/BoatRaceSharedTask/baseline.html for model description
