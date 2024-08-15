#include <cstdlib>
#include <stdint.h>	// int64_t
#include <iostream> // cout
#include <iomanip> // setprecision
#include "L2lu64.h"
#include <gmpxx.h>
#include "intpoly.h"
#include <cmath>	// sqrt
#include <fstream>	// file
#include <ctime>	// clock_t
#include <cstring>	// memset
#include <omp.h>
#include "mpz_poly.h"
#include <sstream>	// stringstream
#include <stack>	// stack

using std::cout;
using std::endl;
using std::flush;
//using std::vector;
using std::string;
using std::ifstream;
using std::fixed;
using std::scientific;
using std::setprecision;
using std::sort;
using std::to_string;
using std::hex;
using std::stringstream;
using std::stack;
using std::abs;

struct keyval {
	uint64_t id;
	uint8_t logp;
};

union int128_t {
	__int128 int128;
	int8_t bytes[16];
};

__int128 MASK64;

bool int128_compare(const int128_t &v1, const int128_t &v2)
{
	return v2.int128 < v1.int128;
}

inline int max(int u, int v);
bool bucket_sorter(keyval const& kv1, keyval const& kv2);
void slcsieve(int numlc, mpz_t* Ak, mpz_t* Bk, int Bmin, int Bmax, int Rmin, int Rmax,
	 int* sieve_p, int* sieve_r, int* sieve_n, int degf, keyval* M, uint64_t* m,
	 int mbb, int bb);
int64_t rel2A(int d, mpz_t* Ai, int64_t reli, int bb);
int64_t rel2B(int d, mpz_t* Bi, int64_t reli, int bb);
int enumerate5d(int d, int n, int64_t* L, keyval* M, uint64_t* m, uint8_t logp, int64_t p,
	int R, int nnmax, int mbb, int bb);
void printvector(int d, uint64_t v, int hB);
void printvectors(int d, vector<uint64_t> &M, int n, int hB);
inline int64_t gcd(int64_t a, int64_t b);
inline __int128 gcd128(__int128 a, __int128 b);
void GetlcmScalar(int B, mpz_t S, int* primes, int nump);
inline __int128 make_int128(uint64_t lo, uint64_t hi);
bool PollardPm1(mpz_t N, mpz_t S, mpz_t factor);
bool PollardPm1_mpz(mpz_t N, mpz_t S, mpz_t factor);
bool PollardPm1_int128(__int128 N, mpz_t S, __int128 &factor, int64_t mulo, int64_t muhi);
bool EECM(mpz_t N, mpz_t S, mpz_t factor, int d, int a, int X0, int Y0, int Z0);
bool EECM_mpz(mpz_t N, mpz_t S, mpz_t factor, int d, int a, int X0, int Y0, int Z0);
bool EECM_int128(__int128 N, mpz_t S, __int128 &factor, int d, int a, int X0, int Y0, int Z0, int64_t mulo, int64_t muhi);


int main(int argc, char** argv)
{
	// set constant
	MASK64 = 1L;
	MASK64 = MASK64 << 64;
	MASK64 = MASK64 - 1L;

	//cout << (uint64_t)(MASK64) << " " << (uint64_t)(MASK64 >> 64) << endl;

	if (argc != 17) {
		cout << endl << "Usage: ./slcsieve inputpoly factorbasefile d Amax Bmax N "
			"Bmin Bmax Rmin Rmax th0 th1 lpb cofmaxbits mbb bb" << endl << endl;
		cout << "    inputpoly       input polynomial in N/skew/C0..Ck/Y0..Y1 format" << endl;
		cout << "    factorbasefile  factor base produced with makesievebase" << endl;
		cout << "    d               sieving dimension, always 5 for the moment" << endl;
		cout << "    Amax            upper bound for A in A*x + B ideal generator" << endl;
		cout << "    Bmax            upper bound for B in A*x + B ideal generator" << endl;
		cout << "    N               number of workunits (think \"special-q\")" << endl;
		cout << "    Bmin            lower bound on sieving primes" << endl;
		cout << "    Bmax            upper bound on sieving primes" << endl;
		cout << "    Rmin            initial value of sieve radius" << endl;
		cout << "    Rmax            final value of sieve radius" << endl;
		cout << "    th0             sum(logp) threshold on side 0" << endl;
		cout << "    th1             sum(logp) threshold on side 1" << endl;
		cout << "    lpb             large prime bound for both sides (can be mpz_t)" << endl;
		cout << "    cofmaxbits      should be 11" << endl;
		cout << "    mbb             bits in lattice data limit (e.g. 29,30...)" << endl;
		cout << "    bb              bits in lattice coefficient range [-bb/2,bb/2]^d" << endl;
		cout << endl;
		return 0;
	}

	cout << "# ";
	for (int i = 0; i < argc; i++) cout << argv[i] << " ";
	cout << endl;

	bool verbose = false;
		
	if (verbose) cout << endl << "Reading input polynomial in file " << argv[1] << "..." << flush;
	//vector<mpz_class> fpoly;
	//vector<mpz_class> gpoly;
	mpz_t* fpoly = new mpz_t[20];	// max degree of 20.  Not the neatest
	mpz_t* gpoly = new mpz_t[20];	// max degree of 20.  Not the neatest
	for (int i = 0; i < 20; i++) {
		mpz_init(fpoly[i]);
		mpz_init(gpoly[i]);
	}
	string line;
	char linebuffer[100];
	ifstream file(argv[1]);
	getline(file, line);	// first line contains number n to factor
	getline(file, line);	// second line contains the skew
	line = line.substr(line.find_first_of(" ")+1);
	int64_t skew = strtoll(line.c_str(), NULL, 10); 
	// read nonlinear poly
	int degf = -1;
	if (verbose) cout << endl << "Side 0 polynomial f0 (ascending coefficients)" << endl;
	while (getline(file, line) && line.substr(0,1) == "c" ) {
		line = line.substr(line.find_first_of(" ")+1);
		//mpz_set_str(c, line.c_str(), 10);
		mpz_set_str(fpoly[++degf], line.c_str(), 10);
		//mpz_get_str(linebuffer, 10, fpoly[degf-1]);
		if (verbose) cout << line << endl;
	}
	//int degf = fpoly.size();
	// read other poly
	int degg = -1;
	bool read = true;
	if (verbose) cout << endl << "Side 1 polynomial f1: (ascending coefficients)" << endl;
	while (read && line.substr(0,1) == "Y" ) {
		line = line.substr(line.find_first_of(" ")+1);
		//mpz_set_str(c, line.c_str(), 10);
		mpz_set_str(gpoly[++degg], line.c_str(), 10);
		//mpz_get_str(linebuffer, 10, gpoly[degg-1]);
		if (verbose) cout << line << endl;
		read = static_cast<bool>(getline(file, line));
	}
	//int degg = gpoly.size();
	file.close();
	//mpz_clear(c);
	if (verbose) cout << endl << "Complete.  Degree f0 = " << degf << ", degree f1 = " << degg << "." << endl;

	if (verbose) cout << endl << "Starting sieve of Eratosthenes for small primes..." << endl;
	int max = 1<<21; // 10000000;// 65536;
	char* sieve = new char[max+1]();
	int* primes = new int[2097152]; //int[1077871]; // int[155611]; //new int[809228];	//new int[6542]; 	// 2039 is the 309th prime, largest below 2048
	for (int i = 2; i <= sqrt(max); i++)
		if(!sieve[i])
			for (int j = i*i; j <= max; j += i)
				if(!sieve[j]) sieve[j] = 1;
	int nump = 0;
	for (int i = 2; i <= max-1; i++)
		if (!sieve[i])
			primes[nump++] = i;
	if (verbose) cout << "Complete." << endl;

	// set up constants
	std::clock_t start; double timetaken = 0;
	// load factor base
	if (verbose) cout << endl << "Loading factor base..." << endl;
	start = clock();
	ifstream fbfile(argv[2]);
	start = clock();
	// read fbb
	getline(fbfile, line);
	int fbb = atoi(line.c_str());
	// read k0
	getline(fbfile, line);
	int k0 = atoi(line.c_str());
	int* sieve_p0 = new int[k0]();
	int* sieve_r0 = new int[degf * k0]();
	int* sieve_n0 = new int[k0]();
	for (int i = 0; i < k0; i++) {
		getline(fbfile, line);
		stringstream ss(line);
		string substr;
		getline(ss, substr, ',');
		sieve_p0[i] = atoi(substr.c_str());
		int j = 0;
		while( ss.good() ) {
			getline( ss, substr, ',' );
			sieve_r0[i*degf + j++] = atoi(substr.c_str());
		}
		sieve_n0[i] = j;
	}
	// read k1
	getline(fbfile, line);
	int k1 = atoi(line.c_str());
	int* sieve_p1 = new int[k1]();
	int* sieve_r1 = new int[degg * k1]();
	int* sieve_n1 = new int[k1]();
	for (int i = 0; i < k1; i++) {
		getline(fbfile, line);
		stringstream ss(line);
		string substr;
		getline(ss, substr, ',');
		sieve_p1[i] = atoi(substr.c_str());
		int j = 0;
		while( ss.good() ) {
			getline( ss, substr, ',' );
			sieve_r1[i*degg + j++] = atoi(substr.c_str());
		}
		sieve_n1[i] = j;
	}
	timetaken += ( clock() - start ) / (double) CLOCKS_PER_SEC;
	if (verbose) cout << "Complete.  Time taken: " << timetaken << "s" << endl;
	if (verbose) cout << "There are " << k0 << " factor base primes on side 0." << endl;
	if (verbose) cout << "There are " << k1 << " factor base primes on side 1." << endl;
	fbfile.close();

	int64_t p0max = sieve_p0[k0-1];
	int64_t p1max = sieve_p1[k1-1];

	int d = atoi(argv[3]);
	mpz_t maxA, maxB;
	mpz_init_set_str(maxA, argv[4], 10);
	mpz_init_set_str(maxB, argv[5], 10);
	int N = atoi(argv[6]);
	int Bmin = atoi(argv[7]);
	int Bmax = atoi(argv[8]);
	int Rmin = atoi(argv[9]);
	int Rmax = atoi(argv[10]);
	uint8_t th0 = atoi(argv[11]);
	uint8_t th1 = atoi(argv[12]);
	mpz_t lpb; mpz_init(lpb);
	mpz_init_set_str(lpb, argv[13], 10);
	int cofmaxbits = atoi(argv[14]);
	int mbb = atoi(argv[15]);
	int bb = atoi(argv[16]);
	int64_t cofmax = 1 << cofmaxbits;

	// main arrays
	uint64_t Mlen = 1<<(mbb);  // 268435456
	keyval* M = new keyval[Mlen];	// lattice { id, logp } pairs
	// allocate 512 bucket pointers in M
	uint64_t m[512];
	cout << fixed << setprecision(1);
	cout << "# lattice data will take " << Mlen*9 << " bytes (" << (double)(Mlen*9)/(1l<<30)
		<< "GB)." << endl;
	cout << setprecision(5);
	vector<int64_t> rel;

	mpz_t* pi = new mpz_t[8]; for (int i = 0; i < 8; i++) mpz_init(pi[i]);
	mpz_poly f0; mpz_poly f1; mpz_poly i1;
	mpz_poly_init(f0, degf); mpz_poly_init(f1, degg); mpz_poly_init(i1, 3);
	mpz_poly_set_mpz(f0, fpoly, degf);
	mpz_poly_set_mpz(f1, gpoly, degg);
	mpz_t N0; mpz_t N1;
	mpz_init(N0); mpz_init(N1);
	stringstream stream;
	mpz_t factor; mpz_init(factor); mpz_t p1; mpz_t p2; mpz_init(p1); mpz_init(p2); mpz_t t; mpz_init(t); 
	mpz_t S; mpz_init(S); GetlcmScalar(cofmax, S, primes, 669);	// max S = 5000
	char* str1 = (char*)malloc(20*sizeof(char));
	char* str2 = (char*)malloc(20*sizeof(char));

	// construct array to hold d - 2 elements of Z[x]
	mpz_t* Ai = new mpz_t[d - 2];
	mpz_t* Bi = new mpz_t[d - 2];
	for (int i = 0; i < d - 2; i++) {
		mpz_init_set_ui(Ai[i], 0);
		mpz_init_set_ui(Bi[i], 0);
	}

	gmp_randstate_t state;
	gmp_randinit_default(state);
	gmp_randseed_ui(state, 123ul);

	mpz_t A; mpz_init(A);
	mpz_t B; mpz_init(B);
	mpz_t g1; mpz_init(g1);

	int R = Rmin;
	int n = d;
	int mm;
	int64_t nn = 0;
	while (nn < N) {
		nn++;

		// clear M
		memset(M, 0, sizeof(M));
	
		// generate d - 2 random elements A*x + B
		for (int i = 0; i < d - 2; i++) {
			mpz_urandomm(Ai[i], state, maxA);
			//mpz_mul_ui(Ai[i], Ai[i], 7);
			//mpz_set_ui(Ai[i], 102987);
			mpz_urandomm(Bi[i], state, maxB);
			mpz_get_str(str1, 10, Ai[i]);
			mpz_get_str(str2, 10, Bi[i]);
			cout << "# " << str1 << "*x + " << str2 << endl;
		}

		// sieve side 0
		cout << "# Starting sieve on side 0..." << endl;
		start = clock();
		slcsieve(d, Ai, Bi, Bmin, Bmax, Rmin, Rmax,
			sieve_p0, sieve_r0, sieve_n0, degf, M, m, mbb, bb);
		timetaken = ( clock() - start ) / (double) CLOCKS_PER_SEC;
		cout << "# Finished! Time taken: " << timetaken << "s" << endl;
		int total = 0;
		for (int i = 0; i < 512; i++) {
			uint64_t mtop = i*(1<<(mbb-9));
			uint64_t mend = m[i];
			total += (mend - mtop);
		}				
		cout << "# Size of lattice point list is " << total << "." << endl;
		cout << "# Sorting bucket sieve data..." << endl;
		start = clock();
		rel.clear();
		int R0 = 0;
		for (int i = 0; i < 512; i++) {
			uint64_t mtop = i*(1<<(mbb-9));
			uint64_t mend = m[i];
			std::sort(M + mtop, M + mend, &bucket_sorter);
			keyval Mmtop = M[mtop];
			uint64_t lastid = Mmtop.id;
			int sumlogp = Mmtop.logp;
			for (uint64_t ii = mtop + 1; ii < mend; ii++) {
				keyval Mii = M[ii];
				uint64_t id = Mii.id;
				if (id == lastid) {
					sumlogp += Mii.logp;
				}
				else {
					if (sumlogp > th0) {
						int64_t A64 = rel2A(d, Ai, lastid, bb);
						int64_t B64 = rel2B(d, Bi, lastid, bb);
						int64_t g = gcd(A64, B64);
						A64 /= g; B64 /= g;
						if (A64 != 0 && B64 != 0 && abs(A64) != 1) {
							//if (R0 < 5) cout << A64 << "*x + " << B64 << " : " << lastid << endl;
							rel.push_back(lastid);
							R0++;
						}
					}
					lastid = id;
					sumlogp = Mii.logp;
				}
				if (ii == mend-1 && sumlogp > th0) {
					int64_t A64 = rel2A(d, Ai, id, bb);
					int64_t B64 = rel2B(d, Bi, id, bb);
					int64_t g = gcd(A64, B64);
					A64 /= g; B64 /= g;
					if (A64 != 0 && B64 != 0 && abs(A64) != 1) {
						//if (R0 < 5) cout << A64 << "*x + " << B64 << endl;
						rel.push_back(id);
						R0++;
					}
				}
			}
		}
		timetaken = ( clock() - start ) / (double) CLOCKS_PER_SEC;
		cout << "# Finished! Time taken: " << timetaken << "s" << endl << flush;
		cout << "# " << R0 << " candidates on side 0." << endl << flush;

		// sieve side 1
		cout << "# Starting sieve on side 1..." << endl;
		start = clock();
		slcsieve(d, Ai, Bi, Bmin, Bmax, Rmin, Rmax,
			sieve_p1, sieve_r1, sieve_n1, degg, M, m, mbb, bb);
		timetaken = ( clock() - start ) / (double) CLOCKS_PER_SEC;
		cout << "# Finished! Time taken: " << timetaken << "s" << endl;
		total = 0;
		for (int i = 0; i < 512; i++) {
			uint64_t mtop = i*(1<<(mbb-9));
			uint64_t mend = m[i];
			total += (mend - mtop);
		}		
		cout << "# Size of lattice point list is " << total << "." << endl;
		cout << "# Sorting bucket sieve data..." << endl;
		start = clock();
		int R1 = 0;
		for (int i = 0; i < 512; i++) {
			uint64_t mtop = i*(1<<(mbb-9));
			uint64_t mend = m[i];
			std::sort(M + mtop, M + mend, &bucket_sorter);
			keyval Mmtop = M[mtop];
			uint64_t lastid = Mmtop.id;
			int sumlogp = Mmtop.logp;
			for (uint64_t ii = mtop + 1; ii < mend; ii++) {
				keyval Mii = M[ii];
				uint64_t id = Mii.id;
				if (id == lastid) {
					sumlogp += Mii.logp;
				}
				else {
					if (sumlogp > th1) {
						int64_t A64 = rel2A(d, Ai, lastid, bb);
						int64_t B64 = rel2B(d, Bi, lastid, bb);
						int64_t g = gcd(A64, B64);
						A64 /= g; B64 /= g;
						if (A64 != 0 && B64 != 0 && abs(A64) != 1) {
							//if (R1 < 5) cout << A64 << "*x + " << B64 << " : " << lastid << endl;
							rel.push_back(lastid);
							R1++;
						}
					}
					lastid = id;
					sumlogp = Mii.logp;
				}
				if (ii == mend-1 && sumlogp > th1) {
					int64_t A64 = rel2A(d, Ai, id, bb);
					int64_t B64 = rel2B(d, Bi, id, bb);
					int64_t g = gcd(A64, B64);
					A64 /= g; B64 /= g;
					if (A64 != 0 && B64 != 0 && abs(A64) != 1) {
						//if (R1 < 5) cout << A64 << "*x + " << B64 << endl;
						rel.push_back(id);
						R1++;
					}
				}
			}
		}
		timetaken = ( clock() - start ) / (double) CLOCKS_PER_SEC;
		cout << "# Finished! Time taken: " << timetaken << "s" << endl << flush;
		cout << "# " << R1 << " candidates on side 1." << endl << flush;
		
		cout << "# Sorting candidate relation list..." << endl;
		start = clock();
		sort(rel.begin(), rel.end());
		timetaken = ( clock() - start ) / (double) CLOCKS_PER_SEC;
		cout << "# Finished! Time taken: " << timetaken << "s" << endl;
		
		// print list of potential relations
		int R = 0;
		for (int i = 0; i < rel.size(); i++)
		{
			if (i == rel.size() - 1) break;
			if (rel[i] == rel[i+1] && rel[i] != 0) {
				int64_t A64 = rel2A(d, Ai, rel[i], bb);
				int64_t B64 = rel2B(d, Bi, rel[i], bb);
				if (A64 != 0 && B64 != 0) {
					int64_t g = gcd(A64, B64);
					A64 /= g; B64 /= g;
					if (R < 10) cout << A64 << "*x + " << B64 << endl;
					R++;
				}
			}
		}
		cout << "# " << R << " potential relations found." << endl << flush;
	
		// compute and factor resultants as much as possible.
		int BASE = 16;
		stack<mpz_t*> QN; stack<int> Q; int algarr[3]; mpz_t* N;
		start = clock();
		R = 0;
		if (verbose) cout << "Starting cofactorizaion..." << endl;
		for (int i = 0; i <= (int)(rel.size()-1); i++) {
			if (rel[i] == rel[i+1] && rel[i] != 0) {
				// construct A*x + B from small linear combinations
				int64_t A64 = rel2A(d, Ai, rel[i], bb);
				int64_t B64 = rel2B(d, Bi, rel[i], bb);
				mpz_set_si(A, A64); mpz_set_si(B, B64);
				
				// remove content of A*x + B
				mpz_gcd(g1, A, B);
				mpz_divexact(A, A, g1);
				mpz_divexact(B, B, g1);

				// set relation principal ideal generater i1 = A*x + B and compute norms
				mpz_poly_setcoeff(i1, 1, A);
				mpz_poly_setcoeff(i1, 0, B);
				mpz_poly_resultant(N0, f0, i1);
				mpz_poly_resultant(N1, f1, i1);
				mpz_abs(N0, N0);
				mpz_abs(N1, N1);
				//cout << mpz_get_str(NULL, 10, N0) << endl;
				//cout << mpz_get_str(NULL, 10, N1) << endl;

				mpz_get_str(str1, 10, A);
				mpz_get_str(str2, 10, B);
				string str = string(str1) + "," + string(str2) + ":";
						
				// trial division on side 0
				int p = primes[0]; int k = 0; 
				while (p < sieve_p0[k0-1]) {
					int valp = 0;
					while (mpz_fdiv_ui(N0, p) == 0) {
						mpz_divexact_ui(N0, N0, p);
						valp++;
						stream.str("");
						stream << hex << p;
						str += stream.str() + ",";
					}
					if (p < 1000) {
						p = primes[++k];
						if (p > 1000) {
							k = 0;
							while (sieve_p0[k] < 1000) k++;
						}
					}
					else {
						p = sieve_p0[++k];
					}
				}
				bool isrel = true;
				bool cofactor = true;
				if (mpz_cmp_ui(N0, 1) == 0) { cofactor = false; }
				str += (cofactor ? "," : "");
				// cofactorization on side 0
				int ii = 0; while (!Q.empty()) Q.pop(); while (!QN.empty()) QN.pop();
				if (cofactor) {
					if (mpz_probab_prime_p(N0, 30) == 0) {  // cofactor definitely composite
						
						QN.push(&N0); Q.push(2); Q.push(1); Q.push(0); Q.push(3);
						while (!QN.empty()) {
							mpz_t* N = QN.top(); QN.pop();
							int l = Q.top(); Q.pop();
							int j = 0;
							bool factored = false;
							while (!factored) {
								int alg = Q.top(); Q.pop(); j++;
								switch (alg) {
									case 0: factored = PollardPm1(*N, S, factor);
											break;	
									case 1: factored = EECM(*N, S, factor, 25921, 83521, 19, 9537, 2737);
											break;	
									case 2: factored = EECM(*N, S, factor, 1681, 707281, 3, 19642, 19803);
											break;	
								}
								if ( !factored ) {
									if ( j >= l ) { isrel = false; break; }
								}
								else {
									mpz_set(p1, factor);
									mpz_divexact(p2, *N, factor);
									if (mpz_cmpabs(p1, p2) > 0) {
										 mpz_set(t, p1); mpz_set(p1, p2); mpz_set(p2, t);	// sort
									}
									// save remaining algs to array
									int lnext = l - j; int lt = lnext;
									while (lt--) { algarr[lt] = Q.top(); Q.pop(); }
									lt = lnext; if (lt) { while (lt--) Q.push(algarr[lnext-1-lt]); Q.push(lnext); }
									if (mpz_probab_prime_p(p1, 30)) {
										if (mpz_cmpabs(p1, lpb) > 0) { isrel = false; break; }
										else { mpz_get_str(str2, BASE, p1); str += str2; str += ","; }
									}
									else {
										if (!lnext) { isrel = false; break; }
										mpz_set(pi[ii], p1);
										QN.push(&pi[ii++]);
										lt = lnext; if (lt) { while (lt--) Q.push(algarr[lnext-1-lt]); Q.push(lnext); }
									}
									if (mpz_probab_prime_p(p2, 30)) {
										if (mpz_cmpabs(p2, lpb) > 0) { isrel = false; break; }
										else { mpz_get_str(str2, BASE, p2); str += str2; str += QN.empty() ? "" : ","; }
									}
									else {
										if (!lnext) { isrel = false; break; }
										mpz_set(pi[ii], p2);
										QN.push(&pi[ii++]);
										lt = lnext; if (lt) { while (lt--) Q.push(algarr[lnext-1-lt]); Q.push(lnext); }
									}
								}
							}
							if (!isrel) break;
						}
					}
					else {	// cofactor prime but is it < lpb?
						if (mpz_cmpabs(N0, lpb) > 0) isrel = false;
						else { mpz_get_str(str2, BASE, N0); str += str2; }
					}
				}
				
				str += ":";

				// trial division on side 1
				if (isrel) {
					p = primes[0]; k = 0;
					while (p < sieve_p1[k1-1]) {
						int valp = 0;
						while (mpz_fdiv_ui(N1, p) == 0) {
							mpz_divexact_ui(N1, N1, p);
							valp++;
							stream.str("");
							stream << hex << p;
							str += stream.str() + ",";
						}
						if (p < 1000) {
							p = primes[++k];
							if (p > 1000) {
								k = 0;
								while (sieve_p1[k] < 1000) k++;
							}
						}
						else {
							p = sieve_p1[++k];
						}
					}
					bool cofactor = true;
					if (mpz_cmp_ui(N1, 1) == 0) { cofactor = false; }
					str += (cofactor ? "," : "");
					// cofactorization on side 1
					ii = 0; while (!Q.empty()) Q.pop(); while (!QN.empty()) QN.pop();
					if (cofactor) {
						if (mpz_probab_prime_p(N1, 30) == 0) {  // cofactor definitely composite
							
							QN.push(&N1); Q.push(2); Q.push(1); Q.push(0); Q.push(3);
							while (!QN.empty()) {
								mpz_t* N = QN.top(); QN.pop();
								int l = Q.top(); Q.pop();
								int j = 0;
								bool factored = false;
								while (!factored) {
									int alg = Q.top(); Q.pop(); j++;
									switch (alg) {
										case 0: factored = PollardPm1(*N, S, factor);
												break;	
										case 1: factored = EECM(*N, S, factor, 25921, 83521, 19, 9537, 2737);
												break;	
										case 2: factored = EECM(*N, S, factor, 1681, 707281, 3, 19642, 19803);
												break;	
									}
									if ( !factored ) {
										if ( j >= l ) { isrel = false; break; }
									}
									else {
										mpz_set(p1, factor);
										mpz_divexact(p2, *N, factor);
										if (mpz_cmpabs(p1, p2) > 0) {
											 mpz_set(t, p1); mpz_set(p1, p2); mpz_set(p2, t);	// sort
										}
										// save remaining algs to array
										int lnext = l - j; int lt = lnext;
										while (lt--) { algarr[lt] = Q.top(); Q.pop(); }
										lt = lnext; if (lt) { while (lt--) Q.push(algarr[lnext-1-lt]); Q.push(lnext); }
										if (mpz_probab_prime_p(p1, 30)) {
											if (mpz_cmpabs(p1, lpb) > 0) { isrel = false; break; }
											else { mpz_get_str(str2, BASE, p1); str += str2; str += ","; }
										}
										else {
											if (!lnext) { isrel = false; break; }
											mpz_set(pi[ii], p1);
											QN.push(&pi[ii++]);
											lt = lnext; if (lt) { while (lt--) Q.push(algarr[lnext-1-lt]); Q.push(lnext); }
										}
										if (mpz_probab_prime_p(p2, 30)) {
											if (mpz_cmpabs(p2, lpb) > 0) { isrel = false; break; }
											else { mpz_get_str(str2, BASE, p2); str += str2; str += QN.empty() ? "" : ","; }
										}
										else {
											if (!lnext) { isrel = false; break; }
											mpz_set(pi[ii], p2);
											QN.push(&pi[ii++]);
											lt = lnext; if (lt) { while (lt--) Q.push(algarr[lnext-1-lt]); Q.push(lnext); }
										}
									}
								}
								if (!isrel) break;
							}
						}
						else {	// cofactor prime but is it < lpb?
							if (mpz_cmpabs(N1, lpb) > 0) isrel = false;
							else { mpz_get_str(str2, BASE, N1); str += str2; }
						}
					}

					if (isrel) { cout << str << endl; R++; }
				}
			}
		}
		timetaken = ( clock() - start ) / (double) CLOCKS_PER_SEC;
		cout << "# Finished! Cofactorization took " << timetaken << "s" << endl;
		cout << "# " << R << " actual relations found." << endl;
	}

	mpz_clear(g1);
	mpz_clear(B); mpz_clear(A);
	gmp_randclear(state);

	for (int i = 0; i < d - 2; i++) {
		mpz_clear(Bi[i]);
		mpz_clear(Ai[i]);
	}
	delete[] Ai; delete[] Bi;

	free(str1);
	free(str2);
	mpz_clear(S);
	mpz_clear(t); mpz_clear(p2); mpz_clear(p1);
	mpz_clear(factor);
	mpz_clear(lpb);
	mpz_clear(N1); mpz_clear(N0);
	mpz_poly_clear(f1); mpz_poly_clear(f0);
	mpz_clear(maxB); mpz_clear(maxA);
	delete[] M;
	for (int i = 0; i < 8; i++) mpz_clear(pi[i]); delete[] pi;
	delete[] sieve_n1;
	delete[] sieve_p1;
	delete[] sieve_r1;
	delete[] sieve_n0;
	delete[] sieve_p0;
	delete[] sieve_r0;
	delete[] primes;
	delete[] sieve;
	for (int i = 0; i < 20; i++) {
		mpz_clear(gpoly[i]);
		mpz_clear(fpoly[i]);
	}
	delete[] gpoly;
	delete[] fpoly;

	return 0;
}

int64_t rel2A(int d, mpz_t* Ai, int64_t reli, int bb)
{
	int64_t BB = 1<<bb;
	int64_t hB = 1<<(bb-1);
	mpz_t t; mpz_init_set_ui(t, 0);
	int64_t A = 0;
	for (int j = 0; j < d-2; j++) {
		int a = (reli >> (bb*j)) % BB - hB;
		mpz_mul_si(t, Ai[j], a);
		A += mpz_get_si(t);
	}
	int a = (reli >> (bb*(d-2))) % BB - hB;
	A += a;
	mpz_clear(t);
	return A;
}

int64_t rel2B(int d, mpz_t* Bi, int64_t reli, int bb)
{
	int64_t BB = 1<<bb;
	int64_t hB = 1<<(bb-1);
	mpz_t t; mpz_init_set_ui(t, 0);
	int64_t B = 0;
	for (int j = 0; j < d-2; j++) {
		int b = (reli >> (bb*j)) % BB - hB;
		mpz_mul_si(t, Bi[j], b);
		B += mpz_get_si(t);
	}
	int b = (reli >> (bb*(d-1))) % BB - hB;
	B -= b;
	mpz_clear(t);
	return B;
}

void slcsieve(int d, mpz_t* Ak, mpz_t* Bk, int Bmin, int Bmax, int Rmin, int Rmax,
	 int* sieve_p, int* sieve_r, int* sieve_n, int degf, keyval* M, uint64_t* m,
	int mbb, int bb)
{
	int n = d;
	int dn = d*n;
	int dd = d*d;
	int64_t L[25];
	int lastrow = (d-1)*d;

	int i = 0;
	while (sieve_p[i] < Bmin) i++;
	int R = Rmin;
	int64_t p = sieve_p[i];
	uint8_t logp = log2f(p);
	uint64_t mj = 0;
	for (int j = 0; j < 512; j++, mj += (1<<(mbb-9))) m[j] = mj;
	while (p < Bmax) {
		int ni = sieve_n[i];
		for (int j = 0; j < ni; j++) {
			int r = sieve_r[i*degf+j];

			// construct sieving lattice for this p
			for (int k = 0; k < dd; k++) L[k] = 0;
			for (int k = 0; k < d; k++) L[k*d+k] = 1;
			for (int k = 0; k < d - 2; k++) {
				// reduce Ak*x + Bk mod p
				int64_t Amodp = mpz_fdiv_ui(Ak[k], p);
				int64_t Bmodp = mpz_fdiv_ui(Bk[k], p);
				int64_t ri = (Amodp*r + Bmodp) % p;
				L[lastrow + k] = ri;
			}
			// last basis vector gets x - r
			L[lastrow + d - 2] = r;
			L[lastrow + d - 1] = p;

			// reduce L using LLL
			int64L2(L, d, d);
			
			// enumerate all vectors up to radius R in L, up to a max of 1000 vectors
			int nn = enumerate5d(d, d, L, M, m, logp, p, R, 1000, mbb, bb);
			to_string(nn);
		}
		
		// advance to next p
		i++;
		p = sieve_p[i];

		R = (int)(Rmin + (Rmax - Rmin)*((double)p - Bmin)/((double)Bmax - Bmin));
	}
}

void printvector(int d, uint64_t v, int bb)
{
	int hB = 1<<(bb-1);
	int FF = (1<<bb)-1;
	cout << flush;
	for (int i = 0; i < d-1; i++) cout << (int)((v>>(bb*i)) & FF)-hB << ",";
	cout << (int)(v>>(bb*(d-1)))-hB << endl;
}

void printvectors(int d, vector<uint64_t> &M, int n, int bb)
{
	int hB = 1<<(bb-1);
	int FF = (1<<bb)-1;
	cout << flush;
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < d-1; i++) cout << (int)((M[j]>>(bb*i)) & FF)-hB << ",";
		cout << (int)(M[j]>>(bb*(d-1)))-hB << endl;
	}
}

int enumerate5d(int d, int n, int64_t* L, keyval* M, uint64_t* m, uint8_t logp, int64_t p,
	int R, int nnmax, int mbb, int bb)
{
	int64_t hB = 1<<(bb-1);
	int bb2 = bb*2;
	int bb3 = bb*3;
	int bb4 = bb*4;
	int mmax = (1<<mbb)/512;

	int dn = d*n;
	float* b = new float[dn];
	float* uu = new float[dn]();
	float* bnorm = new float[d];
	float* sigma = new float[(d+1)*d];
	float* rhok = new float[d+1];
	int* rk = new int[d+1];
	int* vk = new int[d];
	float* ck = new float[d];
	int* wk = new int[d];
	int* common_part = new int[d];
	int32_t* c = new int32_t[d]();

	// Gram-Schmidt orthogonalization
	int64_t* borig = L;
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < d; k++) {
			uu[k*n + k] = 1;
			b[k*n + i] = (float)borig[k*n + i];
		}
		for (int j = 0; j < i; j++) {
			float dot1 = 0;
			float dot2 = 0;
			for (int k = 0; k < d; k++) {
				dot1 += borig[k*n + i] * b[k*n + j];
				dot2 += b[k*n + j] * b[k*n + j];
			}
			uu[j*n + i] = dot1 / dot2;
			for (int k = 0; k < d; k++) {
				b[k*n + i] -= uu[j*n + i] * b[k*n + j];
			}
		}
	}

	// compute orthogonal basis vector norms
	for (int i = 0; i < n; i++) {
		float N = 0;
		for (int k = 0; k < d; k++) N += b[k*n + i] * b[k*n + i];
		bnorm[i] = sqrt(N);
	}

	// set up variables
	for (int i = 0; i < d; i++)
		common_part[i] = 0;
	for (int k = 0; k < d + 1; k++) {
		for (int j = 0; j < n; j++)
			sigma[k*n + j] = 0;
		rhok[k] = 0;
		rk[k] = k;
		if (k < d) {
			vk[k] = 0;
			ck[k] = 0;
			wk[k] = 0;
		}
	}
	vk[0] = 1;
	int t = 1;
	int last_nonzero = 0;
	for (int l = t; l < n; l++) {
		for (int j = 0; j < d; j++) {
			common_part[j] += vk[l] * borig[j*n + l];
		}
	}
	int k = 0;
	int nn = 0;
	// enumerate lattice vectors (x,y,z,r,s,t) in sphere with radius R
	while (true) {
		rhok[k] = rhok[k+1] + (vk[k] - ck[k]) * (vk[k] - ck[k]) * bnorm[k] * bnorm[k];
		if (rhok[k] - 0.00005 <= R*R) {
			if (k == 0) {
				if (last_nonzero != 0 || nn == 0) {
					memset(c, 0, 4*d);
					bool keep = true;
					bool iszero = true;
					for (int j = 0; j < d; j++) {
						c[j] = 0;
						for (int i = 0; i <= t - 1; i++) {
							c[j] += vk[i] * borig[j*d + i] + common_part[j];
							if (abs(c[j]) >= hB) { keep = false; break; }
							if (c[j] != 0) iszero = false;
						}
						if (!keep) break;
					}
					if (c[0] < 0) {	// keep only one of { c, -c } (same information)
						for (int j = 0; j < d; j++) c[j] = -c[j];
					}
					// save vector
					if (keep && !iszero) {
						uint64_t id = c[0]+hB + ((c[1]+hB)<<bb) + ((c[2]+hB)<<bb2)
							+ ((c[3]+hB)<<bb3) + ((c[4]+hB)<<bb4);
						uint64_t mi = id % 509;	// number of buckets
						mi = mi*id % 509;
						mi = mi*id % 509;
						M[m[mi]] = (keyval){ id, logp };
						m[mi]++; // we are relying on the TLB
						int64_t mstart = mi*(1<<(mbb-9));
						if (m[mi] - mstart >= mmax) {
							cout << "mmax = " << mmax << endl;
							cout << "mi = " << mi << endl;
							cout << "mstart = " << mstart << endl;
							cout << "m[mi] = " << m[mi] << endl;
							//for (int i = 0; i < 512; i++) {
							//	mstart = i*(1<<(mbb-9));
							//	cout << "bucket " << i << " has " << m[i] - mstart << " elements." << endl;
							//}
							vector<uint64_t> V;
							for (int i = 0; i < m[mi] - mstart; i++) {
								V.push_back(M[mstart + i].id);
								if (M[mstart + i].id == 0)
									to_string(M[mstart + i].id);
							}
							std::sort(V.begin(), V.end());
							printvectors(d, V, 20, bb);
							cout << "Bucket overflow, likely memory corruption.  Exiting." << endl;
							exit(1);
						}
						nn++;
						//if (nn >= nnmax) break;
					}
				}
				if (vk[k] > ck[k]) vk[k] = vk[k] - wk[k];
				else vk[k] = vk[k] + wk[k];
				wk[k]++;
			}
			else {
				k--;
				rk[k] = max(rk[k], rk[k+1]);
				for (int i = rk[k+1]; i >= k + 1; i--) {
					sigma[i*n + k] = sigma[(i + 1)*n + k] + vk[i] * uu[k*n + i];
				}
				ck[k] = -sigma[(k + 1)*n + k];
				int vk_old = vk[k];
				vk[k] = floor(ck[k] + 0.5); wk[k] = 1;
				if (k >= t && k < n) {
					for (int j = 0; j < d; j++) {
						common_part[j] -= vk_old * borig[j*n + k];
						common_part[j] += vk[k] * borig[j*n + k];
					}
				}
			}
		}
		else {
			k++;
			if (k == n) break;
			rk[k] = k;
			if (k >= last_nonzero) {
				last_nonzero = k;
				int vk_old = vk[k];
				vk[k]++;
				if (k >= t && k < n) {
					for (int j = 0; j < d; j++) {
						common_part[j] -= vk_old * borig[j*n + k];
						common_part[j] += vk[k] * borig[j*n + k];
					}
				}
			}
			else {
				if (vk[k] > ck[k]) {
					int vk_old = vk[k];
					vk[k] = vk[k] - wk[k];
					if (k >= t && k < n) {
						for (int j = 0; j < d; j++) {
							common_part[j] -= vk_old * borig[j*n + k];
							common_part[j] += vk[k] * borig[j*n + k];
						}
					}
				}
				else {
					int vk_old = vk[k];
					vk[k] = vk[k] + wk[k];
					if (k >= t && k < n) {
						for (int j = 0; j < d; j++) {
							common_part[j] -= vk_old * borig[j*n + k];
							common_part[j] += vk[k] * borig[j*n + k];
						}
					}
				}
				wk[k]++;
			}
		}
	}

	delete[] c;
	delete[] common_part;
	delete[] wk;
	delete[] ck;
	delete[] vk;
	delete[] rk;
	delete[] rhok;
	delete[] sigma;
	delete[] bnorm;
	delete[] uu;
	delete[] b;

	return nn;
}

inline bool bucket_sorter(keyval const& kv1, keyval const& kv2)
{
	return kv2.id < kv1.id;
}


inline int max(int u, int v)
{
	int m = u;
	if (v > m) m = v;
	return m;
}


inline int64_t gcd(int64_t a, int64_t b)
{
	a = abs(a);
	b = abs(b);
	int64_t c;
	while (b != 0) {
		c = b;
		b = a % c;
		a = c;
	}
	return a;
}

void GetlcmScalar(int B, mpz_t S, int* primes, int nump)
{
	mpz_t* tree = new mpz_t[nump];
	
	// Construct product tree
	int n = 0;
	mpz_t pe; mpz_init(pe); mpz_t pe1; mpz_init(pe1);
	int p = 2;
	while (p < B) {
		// Set tree[n] = p^e, such that p^e < B < p^(e+1)
		mpz_set_ui(pe, p);
		mpz_mul_ui(pe1, pe, p);
		while (mpz_cmp_ui(pe1, B) < 0) { mpz_set(pe, pe1); mpz_mul_ui(pe1, pe, p); }
		mpz_init(tree[n]);
		mpz_set(tree[n], pe);
		n++;
		p = primes[n];
	}
	mpz_clear(pe); mpz_clear(pe1);
	
	// Coalesce product tree
	uint64_t treepos = n - 1;
	while (treepos > 0) {
		for (int i = 0; i <= treepos; i += 2) {
			if(i < treepos)
				mpz_lcm(tree[i/2], tree[i],tree[i + 1]);
			else
				mpz_set(tree[i/2], tree[i]);
		}
		for (int i = (treepos >> 1); i < treepos - 1; i++) mpz_set_ui(tree[i + 1], 1);
		treepos = treepos >> 1;
	}
	// tree[0] is the lcm of all primes with powers bounded by B
	mpz_set(S, tree[0]);
		
	for (int i = 0; i < n; i++) mpz_clear(tree[i]);
	delete[] tree;
}


bool PollardPm1(mpz_t N, mpz_t S, mpz_t factor)
{
	int bitlen = mpz_sizeinbase(N, 2);
	if (0) { //bitlen < 64) {
		// convert N to __int128
		__int128 N128 = make_int128(mpz_getlimbn(N,0), mpz_getlimbn(N, 1));
		__int128 factor128 = 1;
		int64_t mulo = 0; int64_t muhi = 0;
		bool factored = PollardPm1_int128(N128, S, factor128, mulo, muhi);
		// convert factor128 to mpz_t
		mp_limb_t* factor_limbs = mpz_limbs_modify(factor, 2);
		factor_limbs[0] = factor128 & MASK64;
		factor_limbs[1] = factor128 >> 64;
		//if (factored) cout << mpz_get_str(NULL,10,N) << " = c * " << mpz_get_str(NULL,10,factor128) << " factored!" << endl; 
		return factored;
	}
	else if (0) { //bitlen < 128) {
		// convert N to __int128
		__int128 N128 = make_int128(mpz_getlimbn(N,0), mpz_getlimbn(N, 1));
		__int128 factor128 = 1;
		int64_t mulo = 0; int64_t muhi = 0;
		bool factored = PollardPm1_int128(N128, S, factor128, mulo, muhi);
		// convert factor128 to mpz_t
		mp_limb_t* factor_limbs = mpz_limbs_modify(factor, 2);
		factor_limbs[0] = factor128 & MASK64;
		factor_limbs[1] = factor128 >> 64;
		return factored;
	}
	else {
		return PollardPm1_mpz(N, S, factor);
	}
}


bool PollardPm1_mpz(mpz_t N, mpz_t S, mpz_t factor)
{
	int L = mpz_sizeinbase(S, 2); // exact for base = power of 2
	mpz_t g; mpz_init_set_ui(g, 2);
	  
	// Scalar multiplication using square and multiply
	for (int i = 2; i <= L; i++) {
		// square
		mpz_mul(g, g, g);
		mpz_mod(g, g, N);
		if (mpz_tstbit(S, L - i) == 1) {
			// multiply
			mpz_mul_2exp(g, g, 1);
			if (mpz_cmpabs(g, N) >= 0) mpz_sub(g, g, N);
		}
	}
	// subtract 1
	mpz_sub_ui(g, g, 1);
	// compute gcd
	mpz_gcd(factor, N, g);
	bool result = mpz_cmpabs_ui(factor, 1) > 0 && mpz_cmpabs(factor, N) < 0;
	//if (result) cout << endl << endl << "\t\t\tP-1 worked!!!!" << endl << endl;
	mpz_clear(g);
	return result;
}


inline __int128 gcd128(__int128 a, __int128 b)
{
	a = a < 0 ? -a : a;
	b = b < 0 ? -b : b;
	__int128 c;
	while (b != 0) {
		c = b;
		b = a % c;
		a = c;
	}
	return a;
}


bool PollardPm1_int128(__int128 N, mpz_t S, __int128 &factor, int64_t mulo, int64_t muhi)
{
	int L = mpz_sizeinbase(S, 2); // exact for base = power of 2
	__int128 g = 2;
	  
	// Scalar multiplication using square and multiply
	for (int i = 2; i <= L; i++) {
		// square
		g = g * g;
		g = g % N;
		if (mpz_tstbit(S, L - i) == 1) {
			// multiply
			g = g * 2;
			if (g >= N) g -= N; 
		}
	}
	// subtract 1
	g = g - 1;
	// compute gcd
	factor = gcd128(N, g);
	bool result = (factor > 1) && (factor < N);
	//if (result) cout << endl << endl << "\t\t\tP-1 worked!!!!" << endl << endl;
	return result;
}


bool EECM(mpz_t N, mpz_t S, mpz_t factor, int d, int a, int X0, int Y0, int Z0)
{
	int bitlen = mpz_sizeinbase(N, 2);
	if (0) { //bitlen < 44) {
		//cout << mpz_get_str(NULL,10,N) << endl; 
		// convert N to __int128
		__int128 N128 = make_int128(mpz_getlimbn(N,0), mpz_getlimbn(N, 1));
		__int128 factor128 = 1;
		int64_t mulo = 0; int64_t muhi = 0;
		bool factored = EECM_int128(N128, S, factor128, d, a, X0, Y0, Z0, mulo, muhi);
		// convert factor128 to mpz_t
		mp_limb_t* factor_limbs = mpz_limbs_modify(factor, 2);
		factor_limbs[0] = factor128 & MASK64;
		factor_limbs[1] = factor128 >> 64; //if (factored) cout << mpz_get_str(NULL,10,N) << " factored!" << endl; 
		return factored;
	}
	else if (0) { //bitlen < 128) {
		// convert N to __int128
		__int128 N128 = make_int128(mpz_getlimbn(N,0), mpz_getlimbn(N, 1));
		__int128 factor128 = 1;
		int64_t mulo = 0; int64_t muhi = 0;
		bool factored = EECM_int128(N128, S, factor128, d, a, X0, Y0, Z0, mulo, muhi);
		// convert factor128 to mpz_t
		mp_limb_t* factor_limbs = mpz_limbs_modify(factor, 2);
		factor_limbs[0] = factor128 & MASK64;
		factor_limbs[1] = factor128 >> 64;
		return factored;
	}
	else {
		return EECM_mpz(N, S, factor, d, a, X0, Y0, Z0);
	}
}


/* ScalarMultiplyEdwards
 * 
 * Multiply a point [X0:Y0:Z0] on a twisted edwards curve by a scalar multiple
 * d	d parameter of twisted Edwards curve
 * a	a parameter of twisted Edwards curve
 * X0,Y0,Z0	point on curve to multiply, in projective coordinates
 * N	we work modulo N
 * S	scalar multiple
 * L	length of S in bits");
*/
bool EECM_mpz(mpz_t N, mpz_t S, mpz_t factor, int d, int a, int X0, int Y0, int Z0)
{
	mpz_t SX, SY, SZ;
	mpz_t A, B, B2, B3, C, dC, B2mC, D, CaD, B2mCmD, E, EmD, F, AF, G, AG, aC, DmaC, H, Hx2, J;
	mpz_t X0aY0, X0aY0xB, X0aY0xB_mCmD, X, Y, Z, mulmod;
	
	mpz_init(A); mpz_init(B); mpz_init(B2); mpz_init(B3); mpz_init(C); mpz_init(dC); mpz_init(B2mC); 
	mpz_init(D); mpz_init(CaD); mpz_init(B2mCmD); mpz_init(E); mpz_init(EmD); mpz_init(F); mpz_init(AF);
	mpz_init(G); mpz_init(AG); mpz_init(aC); mpz_init(DmaC); mpz_init(H); mpz_init(Hx2); mpz_init(J);
	mpz_init(X0aY0xB); mpz_init(X0aY0xB_mCmD); mpz_init(X); mpz_init(Y); mpz_init(Z);
	
	mpz_init_set_ui(SX, X0);
	mpz_init_set_ui(SY, Y0);
	mpz_init_set_ui(SZ, Z0);
	
	mpz_init(mulmod);
//int len=mpz_sizeinbase(N,2);if(len > 64 && len < 128) cout << mpz_get_str(NULL,10,N) << endl; 
	int L = mpz_sizeinbase(S, 2); // exact for base = power of 2
	
	// Scalar multiplication using double & add algorithm
	// doubling formula: [2](x:y:z) = ((B-C-D)*J:F*(E-D):F*J)
	for(int i = 2; i <= L; i++) {
		// double
		mpz_add(B, SX, SY);
		mpz_mul(mulmod, B, B); mpz_mod(B2, mulmod, N);
		mpz_mul(mulmod, SX, SX); mpz_mod(C, mulmod, N);
		mpz_mul(mulmod, SY, SY); mpz_mod(D, mulmod, N);
		mpz_mul_ui(E, C, a);
		mpz_add(F, E, D);
		mpz_mul(mulmod, SZ, SZ); mpz_mod(H, mulmod, N);
		mpz_mul_2exp(Hx2, H, 1);
		mpz_sub(J, F, Hx2);
		mpz_add(CaD, C, D);
		mpz_sub(B2mCmD, B2, CaD);
		mpz_mul(X, B2mCmD, J);
		mpz_sub(EmD, E, D);
		mpz_mul(Y, F, EmD);
		mpz_mul(Z, F, J);
		mpz_mod(SX, X, N);
		mpz_mod(SY, Y, N);
		mpz_mod(SZ, Z, N);
		if(mpz_tstbit(S, L - i) == 1) {
			// add
			mpz_mul_ui(A, SZ, Z0);
			mpz_add(B, SX, SY);
			mpz_mul(mulmod, A, A); mpz_mod(B3, mulmod, N);
			mpz_mul_ui(C, SX, X0);
			mpz_mul_ui(D, SY, Y0);
			mpz_mul_ui(dC, C, d);
			mpz_add(CaD, C, D);
			mpz_mul(mulmod, dC, D); mpz_mod(E, mulmod, N);
			mpz_sub(F, B3, E);
			mpz_add(G, B3, E);
			mpz_mul_ui(mulmod, B, X0+Y0); mpz_mod(X0aY0xB, mulmod, N);
			mpz_sub(X0aY0xB_mCmD, X0aY0xB, CaD);
			mpz_mul(mulmod, A, F); mpz_mod(AF, mulmod, N);
			mpz_mul(X, AF, X0aY0xB_mCmD);
			mpz_mul(mulmod, A, G); mpz_mod(AG, mulmod, N);
			mpz_mul_ui(aC, C, a);
			mpz_sub(DmaC, D, aC);
			mpz_mul(Y, AG, DmaC);
			mpz_mul(Z, F, G);
			mpz_mod(SX, X, N);
			mpz_mod(SY, Y, N);
			mpz_mod(SZ, Z, N);
		}
	}
	// try to retrieve factor
	mpz_gcd(factor, N, SX);

	mpz_clear(mulmod); mpz_clear(SZ); mpz_clear(SY); mpz_clear(SX);
	mpz_clear(A); mpz_clear(B); mpz_clear(B2); mpz_clear(B3); mpz_clear(C); mpz_clear(dC); mpz_clear(B2mC); 
	mpz_clear(D); mpz_clear(CaD); mpz_clear(B2mCmD); mpz_clear(E); mpz_clear(EmD); mpz_clear(F); mpz_clear(AF);
	mpz_clear(G); mpz_clear(AG); mpz_clear(aC); mpz_clear(DmaC); mpz_clear(H); mpz_clear(Hx2); mpz_clear(J);
	mpz_clear(X0aY0xB); mpz_clear(X0aY0xB_mCmD); mpz_clear(X); mpz_clear(Y); mpz_clear(Z);
	
	bool result = mpz_cmpabs_ui(factor, 1) > 0 && mpz_cmpabs(factor, N) < 0;
	//if (result) cout << endl << endl << "\t\t\tEECM worked!!!!" << endl << endl;
	return result;
}



/* ScalarMultiplyEdwards (__int128 version)
 * 
 * Multiply a point [X0:Y0:Z0] on a twisted edwards curve by a scalar multiple
 * d	d parameter of twisted Edwards curve
 * a	a parameter of twisted Edwards curve
 * X0,Y0,Z0	point on curve to multiply, in projective coordinates
 * N	we work modulo N
 * S	scalar multiple
 * L	length of S in bits");
*/
bool EECM_int128(__int128 N, mpz_t S, __int128 &factor, int d, int a, int X0, int Y0, int Z0, int64_t mulo, int64_t muhi)
{
	__int128 SX, SY, SZ;
	__int128 A, B, B2, B3, C, dC, B2mC, D, CaD, B2mCmD, E, EmD, F, AF, G, AG, aC, DmaC, H, Hx2, J;
	__int128 X0aY0, X0aY0xB, X0aY0xB_mCmD, X, Y, Z, mulmod;
   
   	SX = X0;
	SY = Y-1;
	SZ = Z0;
	int L = mpz_sizeinbase(S, 2); // exact for base = power of 2
	
	// Scalar multiplication using double & add algorithm
	// doubling formula: [2](x:y:z) = ((B-C-D)*J:F*(E-D):F*J)
	for(int i = 2; i <= L; i++) {
		// double
		B = SX + SY;
		B2 = (B * B) % N;
		C = (SX * SX) % N;
		D = (SY * SY) % N;
		E = C * a;
		F = E + D;
		H = (SZ * SZ) % N;
		Hx2 = H << 1;
		J = F - Hx2;
		CaD = C + D;
		B2mCmD = B2 - CaD;
		X = B2mCmD * J;
		EmD = E - D;
		Y = F * EmD;
		Z = F * J;
		SX = X % N;
		SY = Y % N;
		SZ = Z % N;
		if(mpz_tstbit(S, L - i) == 1) {
			// add
			A = SZ * Z0;
			B = SX + SY;
			B3 = (A * A) % N;
			C = SX * X0;
			D = SY * Y0;
			dC = C * d;
			CaD = C + D;
			E = (dC * D) % N;
			F = B3 - E;
			G = B3 + E;
			X0aY0xB = (B * (X0 + Y0)) % N;
			X0aY0xB_mCmD = X0aY0xB - CaD;
			AF = (A * F) % N;
			X = AF * X0aY0xB_mCmD;
			AG = (A * G) % N;
			aC = C * a;
			DmaC = D - aC;
			Y = AG * DmaC;
			Z = F * G;
			SX = X % N;
			SY = Y % N;
			SZ = Z % N;
		}
	}
	// try to retrieve factor
	factor = gcd128(N, SX);

	bool result = (factor > 0) && (factor < N);
	//if (result) cout << endl << endl << "\t\t\tEECM worked!!!!" << endl << endl;
	return result;
}


inline __int128 make_int128(uint64_t lo, uint64_t hi)
{
	__int128 N = hi;
	N = N << 64;
	N += lo;
	return N;
}


