#include <iostream>
#include <vector>
#include <gmp.h>
#include <gmpxx.h>
#include "L2lu64.h"

using namespace std;

// Function to return large entries based on the index
int64_t get_large_entry(int i) {
    // You can modify these large entries as per your requirement
    int64_t large_entries[] = {
		293879283798787l,
		239873773737737l,
		887277373737773l,
		982883778387737l,
		178213723877272l,
		198731982739877l
    };
    return large_entries[i];
}

int main(int argc, char** argv)
{
	int t = 1000000;
	if (argc == 2)
		t = atoi(argv[1]);

	int d = 6;
	int64_t* L = new int64_t[d*d];

	for(int k = 0; k < t; k++) {
	
		// assign L
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < d; ++j) {
				if (i == d-1)
					L[i*d+j] = get_large_entry(j);
				else if (i == j) {
					L[i*d+j] = 1;
				}
				else {
					L[i*d+j] = 0;
				}
			}
		}

		// reduce L
		int64L2(L, d, d);
	}

    // Print the reduced basis
    cout << "Reduced basis:" << endl;
    printbasis(L, d, d, 0, 4);
	cout << endl << endl;

	delete[] L;

    return 0;
}

