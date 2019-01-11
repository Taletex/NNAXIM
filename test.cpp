#include <stdio.h>
#include <bitset>
#include <iostream>
#include <stdint.h>
#include <limits.h> /* for CHAR_BIT */

using namespace std;



/* formatted output of ieee-754 representation of float */
void show_ieee754 (float f)
{
    union {
        float f;
        uint32_t u;
    } fu = { .f = f };
    int i = sizeof f * CHAR_BIT;

    printf ("  ");
    while (i--)
        printf ("%d ", (fu.u >> i) & 0x1);

    putchar ('\n');
    printf (" |- - - - - - - - - - - - - - - - - - - - - - "
            "- - - - - - - - - -|\n");
    printf (" |s|      exp      |                  mantissa"
            "                   |\n\n");
}


// roundb(f, 15) => keep 15 bits in the float, set the other bits to zero
float roundb(float f, int bits) {
  union {
    int i;
    float f;
  } num;

  bits = 32 - bits;  // assuming sizeof(int) == sizeof(float) == 4
  num.f = f;
  
  cout << "num.f" << num.f << endl;
  cout << "num.i" << num.i << endl;
  
  num.i = num.i + (1 << (bits - 1));  // round instead of truncate (meglio così minimizziamo l'errore che stiamo aggiungendo a causa dell'uso di meno bit)
  num.i = num.i & (-1 << bits);
  return num.f;
}

int main() {
	float f;
	float aux;
	
	while(true){
		cout << "Inserisci un numero FP" << endl;
		cin >> f;
		
		aux = roundb(f, 16);
		printf ("\nIEEE-754 Single-Precision representation of: %f (NON TRONCATO) \n\n", f);
		show_ieee754 (f);
		
		printf ("\nIEEE-754 Single-Precision representation of: %f (TRONCATO) \n\n", aux);
		show_ieee754 (aux);
		
		cout << endl << endl << endl << endl << endl;
	}
}

