#include "ac_nn_lib.hh"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <vector>

using namespace std;

/* roundb(f, 15) => keep 15 bits in the float, set the other bits to zero */
float roundb(float f, int bits) {
  union {															// num.i and num.f are mapped on same bits
    int i;
    float f;
  } num;

  bits  = 32 - bits;									
  num.f = f;
  num.i = num.i + (1 << (bits - 1));  // round instead of truncate
  num.i = num.i & (-1 << bits);				// AND bitwise between mask and rounded value
  return num.f;
}

/* Return true if the file exists, else false */
bool exists_file(const string& name) {
  if (FILE* file = fopen(name.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }
}

/* If a file containing the model of the network exists load the network from this file, else create a new model for the network. Finally it returns the network */
tiny_dnn::network<tiny_dnn::sequential> create_network() {
  tiny_dnn::network<tiny_dnn::sequential> net;

	if (exists_file(BASE_PATH + "base-net-model-training-json")) {
    // load the network from existing file
    net.load(BASE_PATH + "base-net-model-training-json", tiny_dnn::content_type::model,
             tiny_dnn::file_format::json);
    cout << "> Rete caricata dal modello salvato" << endl;
	}
	else {
    // create the cnn
		using conv = tiny_dnn::convolutional_layer;
		using pool = tiny_dnn::max_pooling_layer;
		using fc = tiny_dnn::fully_connected_layer;
		using relu = tiny_dnn::relu_layer;
		using softmax = tiny_dnn::softmax_layer;

		const size_t n_fmaps = 32;		// number of feature maps for upper layer
		const size_t n_fmaps2 = 64;		// number of feature maps for lower layer
		const size_t n_fc = 64;				// number of hidden units in fc layer
		
		net << conv(32, 32, 5, 3, n_fmaps, tiny_dnn::padding::same, true, 1, 1, 1, 1, BACKEND_TYPE)					// C1
				<< pool(32, 32, n_fmaps, 2, false, BACKEND_TYPE)																								// P2
				<< relu()																																												// activation
				<< conv(16, 16, 5, n_fmaps, n_fmaps, tiny_dnn::padding::same, true, 1, 1, 1, 1, BACKEND_TYPE)		// C3
				<< pool(16, 16, n_fmaps, 2, false, BACKEND_TYPE)																								// P4
				<< relu()																																												// activation
				<< conv(8, 8, 5, n_fmaps, n_fmaps2, tiny_dnn::padding::same, true, 1, 1, 1, 1, BACKEND_TYPE)    // C5
				<< pool(8, 8, n_fmaps2, 2, false, BACKEND_TYPE)																									// P6
				<< relu()																																												// activation
				<< fc(4 * 4 * n_fmaps2, n_fc, true, BACKEND_TYPE)																								// FC7
				<< relu()																																												// activation
				<< fc(n_fc, 10, true, BACKEND_TYPE) << softmax(10);																							// FC10

    net.save(BASE_PATH + "base-net-model-training-json", tiny_dnn::content_type::model, tiny_dnn::file_format::json);

		cout << "> Rete creata con successo (non era stato trovato nessun modello salvato della rete)" << endl;
  }

  return net;
}

/* Tests the network based on the selected configuration. Returns the classification accuracy of the nn */
float test_network(tiny_dnn::network<tiny_dnn::sequential> net, char config, bool is_testing) {
  
	// loads weights from existing file, if there is no file for the selected configuration it returns
  if (!is_testing) {
    if (exists_file(BASE_PATH + "net-weights-json_" + config)) {
      net.load(BASE_PATH + "net-weights-json_" + config,
               tiny_dnn::content_type::weights, tiny_dnn::file_format::json);
      cout << "> Pesi caricati dalla memoria" << endl;
    } else {
      cout << "> Non e' stato trovato alcun file contenente i pesi per questa configurazione. Non e' possibile procedere con il test." << endl;
      return 0;
    }
  }

	// carico le immagini
	cout << "> Caricamento dataset di test iniziato" << endl;
	std::vector<tiny_dnn::label_t> test_labels;
	std::vector<tiny_dnn::vec_t> test_images;
	parse_cifar10(BASE_CIFAR10_IMAGES_PATH + "cifar-10-batches-bin/test_batch.bin", &test_images, &test_labels, -1.0, 1.0, 0, 0);

	// test and show results
	cout << "> Test della rete iniziato" << endl;
	tiny_dnn::result res = net.test(test_images, test_labels);
	res.print_detail(std::cout);

	return ((float) res.num_success/ (float) res.num_total);
}

/* Sets the number of bits to be used for hidden and I/O layers for approximation in the specified configuration */
void set_approximation_bits(int *hidden_nlayer_bits, int *extern_nlayer_bits, char config) {
	switch (config) {
		case CONFIG::APPROX1:
			*hidden_nlayer_bits = 22;
			*extern_nlayer_bits = 22;
			break;
		case CONFIG::APPROX2:
			*hidden_nlayer_bits = 18;
			*extern_nlayer_bits = 18;
			break;
		case CONFIG::APPROX3:
			*hidden_nlayer_bits = 14;
			*extern_nlayer_bits = 14;
			break;
		case CONFIG::APPROX4:
			*hidden_nlayer_bits = 22;
			*extern_nlayer_bits = 32;
			break;
		case CONFIG::APPROX5:
			*hidden_nlayer_bits = 18;
			*extern_nlayer_bits = 32;
			break;
		case CONFIG::APPROX6:
			*hidden_nlayer_bits = 14;
			*extern_nlayer_bits = 32;
			break;
		case CONFIG::APPROX7:
			*hidden_nlayer_bits = 18;
			*extern_nlayer_bits = 22;
			break;
		case CONFIG::APPROX8:
			*hidden_nlayer_bits = 14;
			*extern_nlayer_bits = 22;
			break;
		case CONFIG::APPROX9:
			*hidden_nlayer_bits = 12;
			*extern_nlayer_bits = 18;
			break;
		default:
			*hidden_nlayer_bits = 32;
			*extern_nlayer_bits = 32;
			break;
	}
}

/* Truncates the weights of the net passed as argument according to the selected configuration */
tiny_dnn::network<tiny_dnn::sequential> truncate_weights(tiny_dnn::network<tiny_dnn::sequential> net, char config) {
	int hidden_nlayer_bits = 32;
	int extern_nlayer_bits = 32;
  
	set_approximation_bits(&hidden_nlayer_bits, &extern_nlayer_bits, config);

	vector<tiny_dnn::vec_t*> weights_list;
	for (size_t k = 0; k < net.depth(); k++) {
		int bits = ((k==0 || k == net.depth() - 1) ? extern_nlayer_bits : hidden_nlayer_bits);

    weights_list = net[k]->weights();

		if (bits < 32) {
			for (size_t i = 0; i < weights_list.size(); i++) {
				for (size_t j = 0; j < weights_list[i]->size(); j++) {
					weights_list[i]->at(j) = roundb(weights_list[i]->at(j), bits);
				}
			}
		}
  }

	return net;
}

/* Returns the number of bits saved thanks to the approximation in the weights bit size */
int saved_bits(tiny_dnn::network<tiny_dnn::sequential> net, char config) {
  int saved_bits = 0;
	int hidden_nlayer_bits = 32;
	int extern_nlayer_bits = 32;

	set_approximation_bits(&hidden_nlayer_bits, &extern_nlayer_bits, config);

	vector<tiny_dnn::vec_t*> weights_list;
  for (size_t k = 0; k < net.depth(); k++) {
		int bits = ((k == 0 || k == net.depth() - 1) ? extern_nlayer_bits : hidden_nlayer_bits);

    weights_list = net[k]->weights();

    for (size_t i = 0; i < weights_list.size(); i++) {
      saved_bits = saved_bits + weights_list[i]->size() * (32-bits);
    }
  }
  return saved_bits;
}

/* Trains the network with the selected configuration */
void train_network(tiny_dnn::network<tiny_dnn::sequential> net, char config, bool is_testing) {

	// loads weights from existing file if the configuration is not the basic one
  if (config != CONFIG::BASE && !is_testing) {
    if (exists_file(BASE_PATH + "net-weights-json_" + config)) {
      char input_retrain = ' ';
      string input       = " ";
      while (input_retrain != '1' && input_retrain != '2') {
        cout
          << "> Per questa configurazione esistono gia' dei pesi salvati. Vuoi "
             "effettuare l'allenamento a partire da questi pesi (digita 1) o "
             "dalla configurazione originale non approssimata (digita 2)?"
          << endl;
        cin >> input; input_retrain = input[0];
      }
      if (input_retrain == '1') {
        net.load(BASE_PATH + "net-weights-json_" + config, tiny_dnn::content_type::weights, tiny_dnn::file_format::json);
				cout << "> Pesi caricati dalla memoria" << endl;
      }
      else {
        net.load(BASE_PATH + "net-weights-json_0", tiny_dnn::content_type::weights, tiny_dnn::file_format::json);
        cout << "> Pesi caricati dalla memoria" << endl;
				net = truncate_weights(net, config);
			}
    } else {
      net.load(BASE_PATH + "net-weights-json_0", tiny_dnn::content_type::weights, tiny_dnn::file_format::json);
      cout << "> Pesi caricati dalla memoria" << endl;
			net = truncate_weights(net, config);
		}
	}

  // set learning parameters
  size_t batch_size = BATCH_NUMBER;																													// n samples for each network weight update
  int epochs        = (config == CONFIG::BASE) ? BASE_EPOCHS_NUMBER : EPOCHS_NUMBER;				// m presentation of all samples
  tiny_dnn::adam optimizer;																																	// optimizer
  cout << "> Parametri per il training impostati: batch size " << batch_size << " - epoche " << epochs << endl;

	// load cifar10 dataset
	cout << "> Caricamento cifar dataset" << endl;
	vector<tiny_dnn::label_t> train_labels, test_labels;
	vector<tiny_dnn::vec_t> train_images, test_images;

	for (int i = 1; i <= 5; i++) {
		parse_cifar10(BASE_CIFAR10_IMAGES_PATH + "cifar-10-batches-bin/data_batch_" + std::to_string(i) + ".bin", &train_images, &train_labels, -1.0, 1.0, 0, 0);
	}
	parse_cifar10(BASE_CIFAR10_IMAGES_PATH + "cifar-10-batches-bin/test_batch.bin", &test_images, &test_labels, -1.0, 1.0, 0, 0);
	cout << "> Caricamento cifar dataset completato" << endl;
	cout << "> Training iniziato" << endl;

	// inizialize display, timer and optimizer variables
	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;
	optimizer.alpha *= static_cast<tiny_dnn::float_t>(sqrt(BATCH_NUMBER) * LEARNING_RATE);

  // on_enumerate_epoch: this lambda function will be called after each epoch
  int epoch = 1;
  auto on_enumerate_epoch = [&]() {
		std::cout << "Epoca " << epoch << "/" << epochs << " completata. "
			<< t.elapsed() << "s trascorsi. Inizio test." << std::endl;
		++epoch;
		tiny_dnn::result res = net.test(test_images, test_labels);
		cout << res.num_success << "/" << res.num_total << std::endl;

		disp.restart(train_images.size());
		t.restart();
  };

  // on_enumerate_minibatch: this lambda function will be called after each minibatch (after weights update)
  auto on_enumerate_minibatch = [&]() {
		disp += BATCH_NUMBER;
		if (config != CONFIG::BASE)
			net = truncate_weights(net, config);
	};

  // learn
	net.train<tiny_dnn::cross_entropy>(optimizer, train_images, train_labels, BATCH_NUMBER, epochs, on_enumerate_minibatch, on_enumerate_epoch);
  
	// truncate and save the new weights
	if (config != CONFIG::BASE)
		net = truncate_weights(net, config);
	net.save(BASE_PATH + "net-weights-json_" + config, tiny_dnn::content_type::weights, tiny_dnn::file_format::json);
	
	// displays the number of bits saved thanks to the approximation
  cout << endl << "> Training terminato con successo!" << (config != CONFIG::BASE ? (" Sono stati risparmiati " + to_string(saved_bits(net, config)) + " bit") : "") << endl << endl;
}

/* Inizialize (train + test) the model and weights of the network on its base configuration (no approximation) */
void inizialize_base_configuration() {
  // train network
	tiny_dnn::network<tiny_dnn::sequential> net = create_network();
  train_network(net, CONFIG::BASE, false);
}

/* Saves results of test and prints them on console */
void save_results(float accuracy_before_retrain[], float accuracy_after_retrain[], int saved_bits_list[], char configs[]) {

	// formatting string to cout and file stream
	cout << endl << fixed << setprecision(2) << "+ ====================================== TEST RESULTS ===================================== +" << endl;
	cout << "| CONFIGURATION | ACCURACY LOSS BEFORE RETRAIN | ACCURACY LOSS AFTER RETRAIN | SAVED BITS |" << endl;
	for (int i = 0; i < 9; i++) {
		cout << "|" << "       " << configs[i] << "       " << "|" << "            " << ((accuracy_before_retrain[9] - accuracy_before_retrain[i]) * 100) << "%" << "            " << "|" << "           " << ((accuracy_after_retrain[9] - accuracy_after_retrain[i]) * 100) << "%" << "           " << "|" << " " << (saved_bits_list[i]) << " |" << "" << endl;
	}
	cout << "+ ====================================== ===== ===== ====================================== +" << endl << endl << endl; 

	ofstream file;
	file.open(BASE_PATH + "log.txt");
	file << fixed << setprecision(2) << "+ ============================================== TEST RESULTS ============================================= +\n";
	file << "| CONFIGURATION | ACCURACY LOSS BEFORE RETRAIN | ACCURACY LOSS AFTER RETRAIN | SAVED BITS |\n";
	for (int i = 0; i < (sizeof(configs) / sizeof(*configs)); i++) {
		file << "|" << "       " << configs[i] << "       " << "|" << "            " << ((accuracy_before_retrain[9] - accuracy_before_retrain[i]) * 100) << "%" << "            " << "|" << "           " << ((accuracy_after_retrain[9] - accuracy_after_retrain[i]) * 100) << "%" << "           " << "|" << " " << (saved_bits_list[i]) << " |" << "\n";
	}
	file << "+ ====================================== ===== ===== ====================================== +\n\n\n";
	file.close();

}

/* Automatic tests all approximate network configurations */
void automatic_test() {
  cout << "===== Test automatico iniziato! =====" << endl;
  tiny_dnn::network<tiny_dnn::sequential> net = create_network();
	char configs[9] = { CONFIG::APPROX1, CONFIG::APPROX2, CONFIG::APPROX3, CONFIG::APPROX4, CONFIG::APPROX5, CONFIG::APPROX6, CONFIG::APPROX7, CONFIG::APPROX8, CONFIG::APPROX9 };
	float accuracy_before_retrain[10] = {};
	float accuracy_after_retrain[10] = {};
	int saved_bits_list[10] = {};

	// settings metrics for base configuration
	cout << "CONFIGURAZIONE ORIGINALE" << endl;
	accuracy_before_retrain[9] = test_network(net, CONFIG::BASE, false);
	accuracy_after_retrain[9] = accuracy_before_retrain[9];
	saved_bits_list[9] = 0;
	cout << "-----------------------------------------------------" << endl;

	// settings metrics for approximated configurations
	for (int i = 0; i < (sizeof(configs)/sizeof(*configs)); i++) {
		cout << endl << "CONFIGURAZIONE " << string(1, configs[i]) << endl;
		net.load(BASE_PATH + "net-weights-json_0", tiny_dnn::content_type::weights, tiny_dnn::file_format::json);

		// truncate and test network
		cout << "> Troncamento e test" << endl;
		net = truncate_weights(net, configs[i]);
		accuracy_before_retrain[i] = test_network(net, configs[i], true);
	
		// retrain and test network
		cout << "> Retraining e test" << endl;
		train_network(net, configs[i], true);
		net.load(BASE_PATH + "net-weights-json_" + string(1, configs[i]), tiny_dnn::content_type::weights, tiny_dnn::file_format::json);
		accuracy_after_retrain[i] = test_network(net, configs[i], true);
		saved_bits_list[i] = saved_bits(net, configs[i]);
		cout << "Sono stati risparmiati " + to_string(saved_bits_list[i]) + " bit" << endl;

		cout << "-----------------------------------------------------" << endl;
	}
	
	save_results(accuracy_before_retrain, accuracy_after_retrain, saved_bits_list, configs);
}

