cmake_minimum_required(VERSION 3.5)
project(BaselineBoatRacePredictor)

set(CMAKE_CXX_STANDARD 14 )
SET(CMAKE_CXX_FLAGS "-std=c++14 -lboost_program_options -lboost_date_time -lboost_serialization -ldynet")

# look for Eigen
get_filename_component(EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
message("-- Eigen dir is " ${EIGEN3_INCLUDE_DIR})
#find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# look for dynet
get_filename_component(DYNET_INCLUDE_DIR "${DYNET_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
message("-- Dynet dir is " ${DYNET_INCLUDE_DIR})
include_directories(${DYNET_INCLUDE_DIR})
link_directories(${DYNET_INCLUDE_DIR}/build/dynet)


# look for Boost
if(DEFINED BOOST_ROOT OR DEFINED BOOSTROOT OR DEFINED ENV{BOOST_ROOT} OR DEFINED ENV{BOOSTROOT})
    set(Boost_NO_SYSTEM_PATHS ON)
    if(DEFINED ${Boost_INCLUDE_DIR})
        get_filename_component(Boost_INCLUDE_DIRS "${Boost_INCLUDE_DIRS}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
    endif()
endif()
set(Boost_REALPATH ON)
set(Boost_USE_STATIC_LIB ON)
find_package(Boost COMPONENTS program_options serialization date_time REQUIRED)
message("-- Boost dir is " ${Boost_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp src/myutils.h src/str2id.h src/race.h src/player.h src/mlp.h)
add_executable(BaselineBoatRacePredictor ${SOURCE_FILES} ${Boost_LIBRARIES})
target_link_libraries(BaselineBoatRacePredictor ${Boost_LIBRARIES} dynet)

