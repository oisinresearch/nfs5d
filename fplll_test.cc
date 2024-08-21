#include <fplll.h>
#include <iostream>
#include <vector>
#include <gmp.h>
#include <gmpxx.h>

using namespace fplll;
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

	int d = 5;	

    // Initialize a d x d matrix in fplll
    ZZ_mat<mpz_t> B;
    B.resize(d, d);

	for(int k = 0; k < t; k++) {
		// Fill the matrix with the appropriate values
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < d; ++j) {
				if (j == d-1) {
					// Set the large entries on the diagonal
					B[i][j] = get_large_entry(i);
				} else if (i == j) {
					// Set the sub-diagonal entries to 1
					B[i][j] = 1;
				} else {
					// Set all other entries to 0
					B[i][j] = 0;
				}
			}
		}

		// Perform LLL reduction on the basis matrix
		lll_reduction(B, LLL_DEF_DELTA, LLL_DEF_ETA, LM_FAST, FT_DOUBLE, 0, LLL_DEFAULT);
	}

    // Print the reduced basis
    cout << "Reduced basis:" << endl;
    B.print(cout, d, d);
	cout << endl << endl;

    return 0;
}

