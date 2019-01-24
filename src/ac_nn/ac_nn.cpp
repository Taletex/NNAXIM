#include "ac_nn_lib.cpp"

int main() {
  tiny_dnn::network<tiny_dnn::sequential> net;
	char input_config = ' ';
	char input_op = ' ';
	char input_training = ' ';
	string input = " ";

	if (!exists_file(BASE_PATH + "net-weights-json_0")) {
		while (input_training != 'S' && input_training != 'N' && input_training != 's' && input_training != 'n') {
			cout << "> ATTENZIONE: non e' stato trovato il file contenente i pesi della configurazione di base (net-weights-json_0)" << endl;
			cout << "> Non e' possibile procedere senza questo file. Vuoi allenare la rete per crearlo? (S/N)" << endl;
			cin >> input; input_training = input[0];
		}

		if (input_training == 'N' || input_training == 'n') {
			return 0;
		}
		else {
			inizialize_base_configuration();
		}
	}

	// Menu
	while (true) {

		cout << endl << endl;
		cout << "+----------------------------------------------------------------------------------------------+" << endl;
		cout << "|                                 BENVENUTO IN NNAXIM                                          |" << endl;
		cout << "+----------------------------------------------------------------------------------------------+" << endl;
		cout << "|Seleziona una configurazione della rete su cui lavorare:                                      |" << endl;
		cout << "| 0) Configurazione originale (nessuna approssimazione)                                        |" << endl;
		cout << "| 1) Configurazione 1 (16 bit per tutti i neuroni della rete)                                  |" << endl;
		cout << "| 2) Configurazione 2 (14 bit per tutti i neuroni della rete)                                  |" << endl;
		cout << "| 3) Configurazione 3 (12 bit per tutti i neuroni della rete)                                  |" << endl;
		cout << "| 4) Configurazione 4 (16 bit per i neuroni degli hidden layer)                                |" << endl;
		cout << "| 5) Configurazione 5 (14 bit per i neuroni degli hidden layer)                                |" << endl;
		cout << "| 6) Configurazione 6 (12 bit per i neuroni degli hidden layer)                                |" << endl;
		cout << "| 7) Configurazione 7 (14 bit per i neuroni degli hidden layer, 16 per quelli degli I/O layer) |" << endl;
		cout << "| 8) Configurazione 8 (12 bit per i neuroni degli hidden layer, 14 per quelli degli I/O layer) |" << endl;
		cout << "| 9) Configurazione 9 (11 bit per i neuroni degli hidden layer, 12 per quelli degli I/O layer) |" << endl;
		cout << "| T) Test automatico di tutte le configurazioni                                                |" << endl;
		cout << "| A) Esecuzione dell'algoritmo di approximate computing per tutte le configurazioni            |" << endl;
		cout << "| Q) Esci                                                                                      |" << endl;
		cout << "+----------------------------------------------------------------------------------------------+" << endl << endl;

		cin >> input; input_config = input[0];
		if (input_config == 'q' || input_config == 'Q') return 0;
		else if (input_config == 't' || input_config == 'T') { automatic_test(); continue; }
		else if (input_config == 'a' || input_config == 'A') { ac_nn_test(); continue; }
		else if (input_config < 48 || input_config > 58) continue;

		cout << "Hai scelto la configurazione #" << input_config << ". Quale operazione vuoi eseguire?" << endl;
		if (input_config != CONFIG::BASE) { cout << "1) Allena la rete con la configurazione " << input_config << ". Cio' comporta i seguenti passi:" << endl << "   - allenamento rete con pesi approssimati;" << endl << "   - troncamento dei nuovi pesi;" << endl << "   - salvataggio dei nuovi pesi su file." << endl; }
		if (input_config == CONFIG::BASE) { cout << "1) Allena la rete con la configurazione " << input_config << endl; }
		cout << "2) Testa la rete con la configurazione " << input_config << " (Test della rete con i pesi salvati)" << endl;
		cout << "Per tornare indietro premi un tasto diverso da 1 e 2" << endl;

		cin >> input; input_op = input[0];
		switch (input_op) {
		case '1':
			net = create_network();
			train_network(net, input_config, false, vector<tiny_dnn::vec_t>(), vector<tiny_dnn::label_t>(), vector<tiny_dnn::vec_t>(), vector<tiny_dnn::label_t>());
			break;
		case '2':
			net = create_network();
			test_network(net, input_config, false, std::vector<tiny_dnn::vec_t>(), std::vector<tiny_dnn::label_t>());
			break;
		default: continue;
		}
	}

	return 0;
}
