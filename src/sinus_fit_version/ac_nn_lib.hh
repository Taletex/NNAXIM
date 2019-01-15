#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <iomanip>
#include "tiny_dnn/tiny_dnn.h"

using namespace std;

// VARIABLES AND CONSTANTS //
const string BASE_PATH = "../../ac_nn/net_params/";
const int BASE_EPOCHS_NUMBER = 10000;
const int EPOCHS_NUMBER = 1000;
const int BATCH_NUMBER = 16;
const enum CONFIG : char { BASE = '0', APPROX1 = '1', APPROX2 = '2', APPROX3 = '3', APPROX4 = '4', APPROX5 = '5', APPROX6 = '6', APPROX7 = '7', APPROX8 = '8', APPROX9 = '9' };

// FUNCTIONS DECLARATION //
bool exists_file(const string& name);

tiny_dnn::network<tiny_dnn::sequential> create_network();

float test_network(tiny_dnn::network<tiny_dnn::sequential> net, char config, bool is_testing);

void set_approximation_bits(int *hidden_nlayer_bits, int *extern_nlayer_bits, char config);

tiny_dnn::network<tiny_dnn::sequential> truncate_weights(tiny_dnn::network<tiny_dnn::sequential> net, char config);

int saved_bits(tiny_dnn::network<tiny_dnn::sequential> net, char config);

void train_network(tiny_dnn::network<tiny_dnn::sequential> net, char config, bool is_testing);

void inizialize_base_configuration();

void save_results(float avg_errors_before_retrain[], float avg_errors_after_retrain[], int saved_bits_list[], char configs[]);

void automatic_test();