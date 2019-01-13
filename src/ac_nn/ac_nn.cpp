#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace std;

// VARIABLES AND CONSTANTS //
const string BASE_PATH = "../../ac_nn/";
const int BASE_EPOCHS_NUMBER = 10000;
const int EPOCHS_NUMBER = 1000;
const int BATCH_NUMBER  = 16;
const enum CONFIG : char {BASE = '0', APPROX1 = '1', APPROX2 = '2', APPROX3 = '3', APPROX4 = '4', APPROX5 = '5', APPROX6 = '6', APPROX7 = '7', APPROX8 = '8', APPROX9 = '9'};

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


/* Tests the network based on the selected configuration */
void test_network(tiny_dnn::network<tiny_dnn::sequential> net,
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
      return;
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
}


/* Truncates the weights of the net passed as argument according to the selected configuration */
// TODO: Qui approssimiamo tutti i pesi di tutti i neuroni approssimandoli a 16
// bit. Bisogna adattare la funzione alla configurazione scelta quando useremo
// la nostra rete
tiny_dnn::network<tiny_dnn::sequential> truncate_weights(tiny_dnn::network<tiny_dnn::sequential> net, char config) {
  cout << "> Troncamento dei pesi" << endl;
	vector<tiny_dnn::vec_t*> weights_list;
	for (size_t k = 0; k < net.depth(); k++) {
    weights_list = net[k]->weights();

    for (size_t i = 0; i < weights_list.size(); i++) {
      for (size_t j = 0; j < weights_list[i]->size(); j++) {
        weights_list[i]->at(j) = roundb(weights_list[i]->at(j), 16);
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
  vector<tiny_dnn::vec_t*> weights_list;
  for (size_t k = 0; k < net.depth(); k++) {
    weights_list = net[k]->weights();

    for (size_t i = 0; i < weights_list.size(); i++) {
      saved_bits = saved_bits + weights_list[i]->size() * 16;
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
  auto on_enumerate_minibatch = [&]() { };

  // learn
  cout << "> Training iniziato" << endl;
  net.fit<tiny_dnn::mse>(opt, X, sinusX, batch_size, epochs, on_enumerate_minibatch, on_enumerate_epoch);
  
	// truncate and save the new weights
  if(config != CONFIG::BASE)
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

void automatic_test() {
  cout << "===== Test automatico iniziato! =====" << endl;
  tiny_dnn::network<tiny_dnn::sequential> net = create_network();

	cout << endl << "CONFIGURAZIONE " << string (1, CONFIG::APPROX1) << endl;
  net.load(BASE_PATH + "net-weights-json_0", tiny_dnn::content_type::weights,tiny_dnn::file_format::json);
  
	// truncate and test network
  cout << "> Troncamento e test" << endl;
	net = truncate_weights(net, CONFIG::APPROX1);
  test_network(net, CONFIG::APPROX1, true);

	// retrain and test network
  cout << "> Retraining e test" << endl;
  train_network(net, CONFIG::APPROX1, true);
  net.load(BASE_PATH + "net-weights-json_" + string (1, CONFIG::APPROX1), tiny_dnn::content_type::weights, tiny_dnn::file_format::json);
  test_network(net, CONFIG::APPROX1, true);
}

/* MAIN FUNCTION */
int main() {
  tiny_dnn::network<tiny_dnn::sequential> net;
  char input_config   = ' ';
  char input_op       = ' ';
  char input_training = ' ';
  string input        = " ";

	if (!exists_file(BASE_PATH + "net-weights-json_0")) {
    while (input_training != 'S' && input_training != 'N' && input_training != 's' && input_training != 'n') {
			cout << "> ATTENZIONE: non e' stato trovato il file contenente i pesi della configurazione di base (net-weights-json_0)" << endl; 
			cout << "> Non e' possibile procedere senza questo file. Vuoi allenare la rete per crearlo? (S/N)" << endl;
      cin >> input; input_training = input[0];
    }

		if (input_training == 'N' || input_training == 'n') {
      return 0;
    } else {
      inizialize_base_configuration();
    }
	}

	// Menu
	while (true) {

		cout << "============================================================================" << endl;
		cout << "                        BENVENUTO IN AC NN SIMULATOR                        " << endl;
		cout << "============================================================================" << endl;
		cout << "Seleziona una configurazione della rete su cui lavorare:" << endl;
		cout << "0) Configurazione originale (nessuna approssimazione)" << endl;
		cout << "1) Configurazione 1 (approssimazione su tutti i neuroni con 22 bit per peso)" << endl;
		cout << "2) Configurazione 2 (approssimazione su tutti i neuroni con 18 bit per peso)" << endl;
		cout << "3) Configurazione 3 (approssimazione su tutti i neuroni con 14 bit per peso)" << endl;
		cout << "4) Configurazione 4 (approssimazione sui neuroni degli hidden layer con 22 bit per peso)" << endl;
		cout << "5) Configurazione 5 (approssimazione sui neuroni degli hidden layer con 18 bit per peso)" << endl;
		cout << "6) Configurazione 6 (approssimazione sui neuroni degli hidden layer con 14 bit per peso)" << endl;
		cout << "7) Configurazione 7 (approssimazione sui neuroni degli hidden layer con 18 bit per peso e sui neuroni dei layer di input e output con 22 bit per peso)" << endl;
		cout << "8) Configurazione 8 (approssimazione sui neuroni degli hidden layer con 14 bit per peso e sui neuroni dei layer di input e output con 22 bit per peso)" << endl;
		cout << "9) Configurazione 9 (approssimazione sui neuroni degli hidden layer con 12 bit per peso e sui neuroni dei layer di input e output con 18 bit per peso)" << endl;
    cout << "T) Test automatico di tutte le configurazioni (ATTENZIONE: potrebbe richiedere molto tempo)" << endl;
		cout << "Q) Esci" << endl << endl;
		
    cin >> input; input_config = input[0];
    if (input_config == 'q' || input_config == 'Q') return 0;
    else if (input_config == 't' || input_config == 'T') { automatic_test(); continue; }
    else if (input_config < 48 || input_config > 58) continue;

		cout << "Hai scelto la configurazione #" << input_config << ". Quale operazione vuoi eseguire?" << endl;
		cout << "1) Allena la rete con la configurazione " << input_config << " (Allenamento rete con pesi troncati, troncamento dei nuovi pesi, salvataggio pesi)" << endl;
		cout << "2) Testa la rete con la configurazione " << input_config << " (Test della rete con i pesi salvati)" << endl;
		cout << "Per tornare indietro premi un tasto diverso da 1 e 2" << endl;
		cin >> input; input_op = input[0];
		switch (input_op) {
      case '1': 
				if (input_config != CONFIG::BASE) {
          net = create_network();
          train_network(net, input_config, false); 
        } else if (input_config == CONFIG::BASE) {
          cout << "> Non e' possibile riallenare la rete nella configurazione originale" << endl << endl;
        }
				break;
      case '2': 
				net = create_network();
				test_network(net, input_config, false);
				break;
			default: continue;
		}
  }
  
	// TODO: Algoritmo da implementare quando abbiamo tutte le config in modo da non dover fare una cosa alla volta
  // for approx orizzontale
  // for approx verticale
  // Approssimare con una certa config
  // Test con approssimazione fatta
  // Raccogliere i risultati del test e i pesi
  // Training della rete approssimata
  // Test della rete approssimata e riallenata
  // Raccogliere i risultati del test e i pesi

  // In totale vanno salvate: i pesi della configurazione di base + per ogni
  // configurazione salvare i pesi approssimati e i pesi approssimati dopo il
  // retraining.

  return 0;
}
