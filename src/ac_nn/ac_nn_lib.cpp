#include "ac_nn_lib.hh"

using namespace std;

// FUNCTIONS //
/* roundb(f, 15) => keep 15 bits in the float, set the other bits to zero */
float roundb(float f, int bits) {
  union {
    int i;
    float f;
  } num;

  bits  = 32 - bits;									// assuming sizeof(int) == sizeof(float) == 4
  num.f = f;
  num.i = num.i + (1 << (bits - 1));  // round instead of truncate
  num.i = num.i & (-1 << bits);
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

/* If a file containing the model of the network exists load the network from this file, else create a new model for the network. Finally it returns the network. */
tiny_dnn::network<tiny_dnn::sequential> create_network() {
  tiny_dnn::network<tiny_dnn::sequential> net;

	if (exists_file(BASE_PATH + "base-net-model-json")) {
    // load the network from existing file
    net.load(BASE_PATH + "base-net-model-json", tiny_dnn::content_type::model,
             tiny_dnn::file_format::json);
    cout << "> Rete caricata dal modello salvato" << endl;
	}
	else {
    // create a simple network with 2 layer of 10 neurons each
    net << tiny_dnn::fully_connected_layer(1, 10);
    net << tiny_dnn::tanh_layer();
    net << tiny_dnn::fully_connected_layer(10, 10);
    net << tiny_dnn::tanh_layer();
    net << tiny_dnn::fully_connected_layer(10, 1);

    net.save(BASE_PATH + "base-net-model-json", tiny_dnn::content_type::model, tiny_dnn::file_format::json);

		cout << "> Rete creata con successo (non e' stato trovato nessun modello salvato della rete)" << endl;
  }

  return net;
}


/* Tests the network based on the selected configuration. Returns the avg accuracy error of the nn */
float test_network(tiny_dnn::network<tiny_dnn::sequential> net,
                  char config,
                  bool is_testing) {
  // loads weights from existing file, if there is no file for the selected
  // configuration it returns
  if (!is_testing) {
    if (exists_file(BASE_PATH + "net-weights-json_" + config)) {
      net.load(BASE_PATH + "net-weights-json_" + config,
               tiny_dnn::content_type::weights, tiny_dnn::file_format::json);
      cout << "> Pesi caricati dalla memoria" << endl;
    } else {
      cout << "> Non e' stato trovato alcun file contenente i pesi per questa "
              "configurazione. Non e' possibile procedere con il test."
           << endl;
      return 0;
    }
  }

  cout << "> Test della rete iniziato" << endl;
  // compare prediction and desired output
  float fMaxError = 0.f;
  float errorSum  = 0.f;
  float count     = 0.f;
  for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
    tiny_dnn::vec_t xv = {x};
    float fPredicted   = net.predict(xv)[0];
    float fDesired     = sinf(x);

    cout << "  x=" << x << " sinX=" << fDesired << " predicted=" << fPredicted
         << endl;

    float fError = fabs(fPredicted - fDesired);

    // update max error
    if (fMaxError < fError) fMaxError = fError;

    // update avg error
    errorSum = errorSum + fError;
    count++;
  }

  cout << endl << "  max_error=" << fMaxError << endl;
  cout << "  avg_error=" << errorSum / count << endl;
  cout << endl << "> Test della rete completato con successo" << endl << endl;

	return (errorSum / count);
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
// TODO: DA ADATTARE ALLE VARIE CONFIG (devi cambiare il numero di bit in base
// alle config)
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
	
  // create input and desired output on a period (dataset di training)
  vector<tiny_dnn::vec_t> X;
  vector<tiny_dnn::vec_t> sinusX;
  for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
    tiny_dnn::vec_t vx    = {x};
    tiny_dnn::vec_t vsinx = {sinf(x)};

    X.push_back(vx);
    sinusX.push_back(vsinx);
  }

  // set learning parameters
  size_t batch_size = BATCH_NUMBER;		// n samples for each network weight update
  int epochs        = (config == CONFIG::BASE) ? BASE_EPOCHS_NUMBER : EPOCHS_NUMBER;  // m presentation of all samples
  tiny_dnn::adamax opt;								// optimizer
  cout << "> Parametri per il training impostati: batch size " << batch_size << " - epoche " << epochs << endl;

  // on_enumerate_epoch: this lambda function will be called after each epoch
  int iEpoch = 0;
  auto on_enumerate_epoch = [&]() {
    // compute loss and disp 1/100 of the time
    iEpoch++;
    if (iEpoch % 100) return;

    double loss = net.get_loss<tiny_dnn::mse>(X, sinusX);
    cout << "  epoch=" << iEpoch << "/" << epochs << " loss=" << loss << endl;
  };

  // on_enumerate_minibatch: this lambda function will be called after each minibatch (after weights update)
  auto on_enumerate_minibatch = [&]() {
		if (config != CONFIG::BASE)
			net = truncate_weights(net, config);
	};

  // learn
  cout << "> Training iniziato" << endl;
  net.fit<tiny_dnn::mse>(opt, X, sinusX, batch_size, epochs, on_enumerate_minibatch, on_enumerate_epoch);
  
	// truncate and save the new weights
	if (config != CONFIG::BASE)
		net = truncate_weights(net, config);
	net.save(BASE_PATH + "net-weights-json_" + config, tiny_dnn::content_type::weights, tiny_dnn::file_format::json);
	
	// displays the number of bits saved thanks to the approximation
  cout << endl << "> Training terminato con successo!" << (config != CONFIG::BASE ? (" Sono stati risparmiati " + to_string(saved_bits(net, config)) + " bit") : "") << endl << endl;
}

/* Inizialize (train + test) the model and weights of the network on its base configuration (no approximation) */
void inizialize_base_configuration() {
  tiny_dnn::network<tiny_dnn::sequential> net = create_network();
  train_network(net, CONFIG::BASE, false);
  test_network(net, CONFIG::BASE, false);
}

/* Saves results of test and prints them on console */
void save_results(float avg_errors_before_retrain[], float avg_errors_after_retrain[], int saved_bits_list[], char configs[]) {

	// formatting string to cout and file stream
	cout << endl << fixed << setprecision(2) << "+ ====================================== TEST RESULTS ===================================== +" << endl;
	cout << "| CONFIGURATION | ACCURACY LOSS BEFORE RETRAIN | ACCURACY LOSS AFTER RETRAIN | SAVED BITS |" << endl;
	for (int i = 0; i < 9; i++) {
		cout << "|" << "       " << configs[i] << "       " << "|" << "            " << ((avg_errors_before_retrain[i] / avg_errors_before_retrain[9]) * 100) << "%" << "            " << "|" << "           " << ((avg_errors_after_retrain[i] / avg_errors_after_retrain[9]) * 100) << "%" << "           " << "|" << " " << (saved_bits_list[i]) << " |" << "" << endl;
	}
	cout << "+ ====================================== ===== ===== ====================================== +" << endl << endl << endl; 

	ofstream file;
	file.open(BASE_PATH + "log.txt");
	file << fixed << setprecision(2) << "+ ============================================== TEST RESULTS ============================================= +\n";
	file << "| CONFIGURATION | ACCURACY LOSS BEFORE RETRAIN | ACCURACY LOSS AFTER RETRAIN | SAVED BITS |\n";
	for (int i = 0; i < (sizeof(configs) / sizeof(*configs)); i++) {
		file << "|" << "       " << configs[i] << "       " << "|" << "            " << ((avg_errors_before_retrain[i] / avg_errors_before_retrain[9]) * 100) << "%" << "            " << "|" << "           " << ((avg_errors_after_retrain[i] / avg_errors_after_retrain[9]) * 100) << "%" << "           " << "|" << " " << (saved_bits_list[i]) << " |" << "\n";
	}
	file << "+ ====================================== ===== ===== ====================================== +\n\n\n";
	file.close();

}

/* Automatic tests all approximate network configurations */
void automatic_test() {
  cout << "===== Test automatico iniziato! =====" << endl;
  tiny_dnn::network<tiny_dnn::sequential> net = create_network();
	char configs[9] = { CONFIG::APPROX1, CONFIG::APPROX2, CONFIG::APPROX3, CONFIG::APPROX4, CONFIG::APPROX5, CONFIG::APPROX6, CONFIG::APPROX7, CONFIG::APPROX8, CONFIG::APPROX9 };
	float avg_errors_before_retrain[10] = {};
	float avg_errors_after_retrain[10] = {};
	int saved_bits_list[10] = {};

	// settings metrics for base configuration
	cout << "CONFIGURAZIONE ORIGINALE" << endl;
	avg_errors_before_retrain[9] = test_network(net, CONFIG::BASE, false);
	avg_errors_after_retrain[9] = avg_errors_before_retrain[9];
	saved_bits_list[9] = 0;
	cout << "-----------------------------------------------------" << endl;

	// settings metrics for approximated configurations
	for (int i = 0; i < (sizeof(configs)/sizeof(*configs)); i++) {
		cout << endl << "CONFIGURAZIONE " << string(1, configs[i]) << endl;
		net.load(BASE_PATH + "net-weights-json_0", tiny_dnn::content_type::weights, tiny_dnn::file_format::json);

		// truncate and test network
		cout << "> Troncamento e test" << endl;
		net = truncate_weights(net, configs[i]);
		avg_errors_before_retrain[i] = test_network(net, configs[i], true);

		// retrain and test network
		cout << "> Retraining e test" << endl;
		train_network(net, configs[i], true);
		net.load(BASE_PATH + "net-weights-json_" + string(1, configs[i]), tiny_dnn::content_type::weights, tiny_dnn::file_format::json);
		avg_errors_after_retrain[i] = test_network(net, configs[i], true);
		saved_bits_list[i] = saved_bits(net, configs[i]);
		cout << "Sono stati risparmiati " + to_string(saved_bits_list[i]) + " bit" << endl;

		cout << "-----------------------------------------------------" << endl;
	}
	
	save_results(avg_errors_before_retrain, avg_errors_after_retrain, saved_bits_list, configs);
}

