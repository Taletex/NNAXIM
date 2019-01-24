#include "tiny_dnn/tiny_dnn.h"

// CONSTANTS //
const std::string BASE_PATH = "../../ac_nn/net_params/";
const std::string BASE_CIFAR10_IMAGES_PATH = "../../ac_nn/";
const int BASE_EPOCHS_NUMBER = 150;
const int RETRAIN_EPOCHS_NUMBER = 5;
const int BASE_BATCH_NUMBER = 10;
const int RETRAIN_BATCH_NUMBER = 500;
const double LEARNING_RATE = 0.002;
const tiny_dnn::core::backend_t BACKEND_TYPE = tiny_dnn::core::default_engine();
const enum CONFIG : char { BASE = '0', APPROX1 = '1', APPROX2 = '2', APPROX3 = '3', APPROX4 = '4', APPROX5 = '5', APPROX6 = '6', APPROX7 = '7', APPROX8 = '8', APPROX9 = '9' };

// FUNCTIONS DECLARATION //
bool exists_file(const std::string& name);

tiny_dnn::network<tiny_dnn::sequential> create_network();

float test_network(tiny_dnn::network<tiny_dnn::sequential> net, char config, bool exec_ac, std::vector<tiny_dnn::vec_t> p_test_images, std::vector<tiny_dnn::label_t> p_test_labels);

void set_approximation_bits(int *hidden_nlayer_bits, int *extern_nlayer_bits, char config);

tiny_dnn::network<tiny_dnn::sequential> approximate_weights(tiny_dnn::network<tiny_dnn::sequential> net, char config);

int saved_bits(tiny_dnn::network<tiny_dnn::sequential> net, char config);

void train_network(tiny_dnn::network<tiny_dnn::sequential> net, char config, bool exec_ac, std::vector<tiny_dnn::vec_t> p_train_images, std::vector<tiny_dnn::label_t> p_train_labels, std::vector<tiny_dnn::vec_t> p_test_images, std::vector<tiny_dnn::label_t> p_test_labels);

void inizialize_base_configuration();

void save_results(float accuracy_before_retrain[], float accuracy_after_retrain[], int saved_bits_list[], char configs[]);

void automatic_test();

void ac_nn_test();
