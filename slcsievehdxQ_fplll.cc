#include <cstdlib>
#include <stdint.h>	// int64_t
#include <iostream> // cout
#include <iomanip> // setprecision
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
#include <fplll.h>
#include "L2lu64.h"

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

class enumvar {
	public:
		float* b;
		float* uu;
		float* bnorm;
		float* sigma;
		float* rhok;
		float* ck;
		int* rk;
		int* vk;
		int* wk;
		int* common_part;
		int64_t* c;
		enumvar(int d, int n) {
			b = new float[d*n];
			uu = new float[d*d];
			bnorm = new float[d];
			sigma = new float[(d+1)*d];
			rhok = new float[d+1];
			rk = new int[d+1];
			vk = new int[d];
			ck = new float[d];
			wk = new int[d];
			common_part = new int[d];
			c = new int64_t[d]();
		}
		~enumvar() {
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
		}
};


struct keyval {
	uint64_t id;
	uint8_t logp;
};

__int128 MASK64;

inline int max(int u, int v);
bool bucket_sorter(keyval const& kv1, keyval const& kv2);
void slcsieve(int numlc, mpz_t* Ak, mpz_t* Bk, int Bmin, int Bmax, int Rmin, int Rmax,
	 int* sieve_p, int* sieve_r, int* sieve_n, int degf, keyval* M, uint64_t* m,
	 int mbb, int bb, int64_t Q0, int64_t R0, mpz_poly f);
void mpz_set_uint128(mpz_t z, __int128 a);
void matmul(int d, ZZ_mat<mpz_t>& C, ZZ_mat<mpz_t>& A, ZZ_mat<mpz_t>& B, mpz_t &t);
void matdivexact_ui(int d, ZZ_mat<mpz_t>& A, uint64_t Q);
void matadj(int d, ZZ_mat<mpz_t> &M, ZZ_mat<mpz_t> &C, ZZ_mat<mpz_t> &MC,
	ZZ_mat<mpz_t> &Madj, mpz_t &t);
void negmat(int d, ZZ_mat<mpz_t> &A);
int64_t rel2A(int d, mpz_t* Ak, int64_t* L, int64_t relid, int bb);
int64_t rel2B(int d, mpz_t* Bk, int64_t* L, int64_t relid, int bb);
int enumeratehd(int d, int n, int64_t* L, keyval* M, uint64_t* m, uint8_t logp, int64_t p,
	int R, int bmax, int64_t* blacklist, int nnmax, int mbb, int bb, enumvar* v1);
int get_blacklist(ZZ_mat<mpz_t> &L, int d, mpz_poly f, int degf, ZZ_mat<mpz_t> &QLinv,
	mpz_t* Ak, mpz_t* Bk, int64_t Q0, int64_t R0, int Rmin, int bb, int bmax, int64_t* blacklist);
void printZZ_mat(ZZ_mat<mpz_t> &L, int d, int n, int pr, int w);
void printvector(int d, uint64_t v, int hB);
void printvectors(int d, vector<uint64_t> &M, int n, int hB);
inline int64_t modinv(int64_t x, int64_t m);
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

	if (argc != 19) {
		cout << endl << "Usage: ./slcsieve inputpoly factorbasefile d Amax Bmax N "
			"Bmin Bmax Rmin Rmax th0 th1 lpb cofmaxbits mbb bb" << endl << endl;
		cout << "    inputpoly       input polynomial in N/skew/C0..Ck/Y0..Y1 format" << endl;
		cout << "    factorbasefile  factor base produced with makesievebase" << endl;
		cout << "    d               sieving dimension" << endl;
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
		cout << "    Q               special-Q to match (can be up to 2^64)" << endl;
		cout << "    R               root of sieving polynomial mod Q" << endl;
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
	int64_t Q0 = strtoll(argv[17], NULL, 10);
	int64_t RR = strtoll(argv[18], NULL, 10);
	__int128 Q = static_cast<__int128>(Q0);
	__int128 R = static_cast<__int128>(RR);

	// main arrays
	uint64_t Mlen = 1ul << mbb;  // 268435456
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
	mpz_t* Ak = new mpz_t[d - 2];
	mpz_t* Bk = new mpz_t[d - 2];
	for (int k = 0; k < d - 2; k++) {
		mpz_init_set_ui(Ak[k], 0);
		mpz_init_set_ui(Bk[k], 0);
	}
	int dd = d*d;
	int64_t* L = new int64_t[dd];

	gmp_randstate_t state;
	gmp_randinit_default(state);
	gmp_randseed_ui(state, 123ul);

	mpz_t A; mpz_init(A);
	mpz_t B; mpz_init(B);
	mpz_t g1; mpz_init(g1);

	int n = d;
	int64_t nn = 0;
	while (nn < N) {
		nn++;

		// clear M
		memset(M, 0, sizeof(M));
	
		// generate d - 2 random elements A*x + B
		for (int k = 0; k < d - 2; k++) {
			mpz_urandomm(Ak[k], state, maxA);
			mpz_urandomm(Bk[k], state, maxB);
			mpz_get_str(str1, 10, Ak[k]);
			mpz_get_str(str2, 10, Bk[k]);
			cout << "# " << str1 << "*x + " << str2 << endl;
		}

		// compute L1
		ZZ_mat<mpz_t> L1;
		L1.resize(d, d);
		for (int k = 0; k < d; k++) {
			for (int l = 0; l < d; l++) {
				if (l == d - 1 && k < d-2) {
					// reduce Ak*x + Bk mod Q0
					__int128 AmodQ = static_cast<__int128>(mpz_fdiv_ui(Ak[k], Q0));
					__int128 BmodQ = static_cast<__int128>(mpz_fdiv_ui(Bk[k], Q0));
					__int128 Ri = (AmodQ*R + BmodQ) % Q;
					L1[k][l] = Ri;
				}
				else if (k == l) {
					L1[k][l] = 1;
				}
				else {
					L1[k][l] = 0;
				}
			}
		}
		// last basis vector gets x - R
		L1[d-2][d-1] = RR;
		L1[d-1][d-1] = Q0;

		// reduce L using LLL
		lll_reduction(L1, LLL_DEF_DELTA, LLL_DEF_ETA, LM_HEURISTIC, FT_DEFAULT,
			0, LLL_DEFAULT);

		// convert column-major L1 to flattened row-major L
		for (int k = 0; k < d; k++)
			for (int l = 0; l < d; l++)
				L[k*d + l] = L1(l, k).get_si();

		// sieve side 0
		cout << "# Starting sieve on side 0..." << endl;
		start = clock();

		slcsieve(d, Ak, Bk, Bmin, Bmax, Rmin, Rmax,
			sieve_p0, sieve_r0, sieve_n0, degf, M, m, mbb, bb, Q0, RR, f0);
		timetaken = ( clock() - start ) / (double) CLOCKS_PER_SEC;
		cout << "# Finished! Time taken: " << timetaken << "s" << endl;
		uint64_t total = 0;
		for (int i = 0; i < 512; i++) {
			uint64_t mtop = i*(1ul<<(mbb-9));
			uint64_t mend = m[i];
			total += (mend - mtop);
		}				
		cout << "# Size of lattice point list is " << total << "." << endl;
		cout << "# Sorting bucket sieve data..." << endl;
		start = clock();
		rel.clear();
		int R0 = 0;
		for (int i = 0; i < 512; i++) {
			uint64_t mtop = i*(1ul<<(mbb-9));
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
						int64_t A64 = rel2A(d, Ak, L, lastid, bb);
						int64_t B64 = rel2B(d, Bk, L, lastid, bb);
						if (A64 != 0 && B64 != 0 && abs(A64) != 1) {
							int64_t g = gcd(A64, B64);
							A64 /= g; B64 /= g;
							if (R0 < 5) cout << A64 << "*x + " << B64 << " : " << lastid << endl;
							rel.push_back(lastid);
							R0++;
						}
					}
					lastid = id;
					sumlogp = Mii.logp;
				}
				if (ii == mend-1 && sumlogp > th0) {
					int64_t A64 = rel2A(d, Ak, L, id, bb);
					int64_t B64 = rel2B(d, Bk, L, id, bb);
					if (A64 != 0 && B64 != 0 && abs(A64) != 1) {
						int64_t g = gcd(A64, B64);
						A64 /= g; B64 /= g;
						if (R0 < 5) cout << A64 << "*x + " << B64 << endl;
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
		slcsieve(d, Ak, Bk, Bmin, Bmax, Rmin, Rmax,
			sieve_p1, sieve_r1, sieve_n1, degg, M, m, mbb, bb, Q0, RR, f1);
		timetaken = ( clock() - start ) / (double) CLOCKS_PER_SEC;
		cout << "# Finished! Time taken: " << timetaken << "s" << endl;
		total = 0;
		for (int i = 0; i < 512; i++) {
			uint64_t mtop = i*(1ul<<(mbb-9));
			uint64_t mend = m[i];
			total += (mend - mtop);
		}		
		cout << "# Size of lattice point list is " << total << "." << endl;
		cout << "# Sorting bucket sieve data..." << endl;
		start = clock();
		int R1 = 0;
		for (int i = 0; i < 512; i++) {
			uint64_t mtop = i*(1ul<<(mbb-9));
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
						int64_t A64 = rel2A(d, Ak, L, lastid, bb);
						int64_t B64 = rel2B(d, Bk, L, lastid, bb);
						if (A64 != 0 && B64 != 0 && abs(A64) != 1) {
							int64_t g = gcd(A64, B64);
							A64 /= g; B64 /= g;
							if (R1 < 5) cout << A64 << "*x + " << B64 << " : " << lastid << endl;
							rel.push_back(lastid);
							R1++;
						}
					}
					lastid = id;
					sumlogp = Mii.logp;
				}
				if (ii == mend-1 && sumlogp > th1) {
					int64_t A64 = rel2A(d, Ak, L, id, bb);
					int64_t B64 = rel2B(d, Bk, L, id, bb);
					if (A64 != 0 && B64 != 0 && abs(A64) != 1) {
						int64_t g = gcd(A64, B64);
						A64 /= g; B64 /= g;
						if (R1 < 5) cout << A64 << "*x + " << B64 << endl;
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
		int T = 0;
		for (int i = 0; i < rel.size(); i++)
		{
			if (i == rel.size() - 1) break;
			if (rel[i] == rel[i+1] && rel[i] != 0) {
				int64_t A64 = rel2A(d, Ak, L, rel[i], bb);
				int64_t B64 = rel2B(d, Bk, L, rel[i], bb);
				if (A64 != 0 && B64 != 0) {
					int64_t g = gcd(A64, B64);
					A64 /= g; B64 /= g;
					if (T < 10) cout << A64 << "*x + " << B64 << endl;
					T++;
				}
			}
		}
		cout << "# " << T << " potential relations found." << endl << flush;
	
		// compute and factor resultants as much as possible.
		int BASE = 16;
		stack<mpz_t*> QN; stack<int> Qu; int algarr[3]; mpz_t* N;
		start = clock();
		T = 0;
		if (verbose) cout << "Starting cofactorizaion..." << endl;
		for (int i = 0; i <= (int)(rel.size()-1); i++) {
			if (rel[i] == rel[i+1] && rel[i] != 0) {
				// construct A*x + B from small linear combinations
				int64_t A64 = rel2A(d, Ak, L, rel[i], bb);
				int64_t B64 = rel2B(d, Bk, L, rel[i], bb);
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
				int ii = 0; while (!Qu.empty()) Qu.pop(); while (!QN.empty()) QN.pop();
				if (cofactor) {
					if (mpz_probab_prime_p(N0, 30) == 0) {  // cofactor definitely composite
						
						QN.push(&N0); Qu.push(2); Qu.push(1); Qu.push(0); Qu.push(3);
						while (!QN.empty()) {
							mpz_t* N = QN.top(); QN.pop();
							int l = Qu.top(); Qu.pop();
							int j = 0;
							bool factored = false;
							while (!factored) {
								int alg = Qu.top(); Qu.pop(); j++;
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
									while (lt--) { algarr[lt] = Qu.top(); Qu.pop(); }
									lt = lnext; if (lt) { while (lt--) Qu.push(algarr[lnext-1-lt]); Qu.push(lnext); }
									if (mpz_probab_prime_p(p1, 30)) {
										if (mpz_cmpabs(p1, lpb) > 0) { isrel = false; break; }
										else { mpz_get_str(str2, BASE, p1); str += str2; str += ","; }
									}
									else {
										if (!lnext) { isrel = false; break; }
										mpz_set(pi[ii], p1);
										QN.push(&pi[ii++]);
										lt = lnext; if (lt) { while (lt--) Qu.push(algarr[lnext-1-lt]); Qu.push(lnext); }
									}
									if (mpz_probab_prime_p(p2, 30)) {
										if (mpz_cmpabs(p2, lpb) > 0) { isrel = false; break; }
										else { mpz_get_str(str2, BASE, p2); str += str2; str += QN.empty() ? "" : ","; }
									}
									else {
										if (!lnext) { isrel = false; break; }
										mpz_set(pi[ii], p2);
										QN.push(&pi[ii++]);
										lt = lnext; if (lt) { while (lt--) Qu.push(algarr[lnext-1-lt]); Qu.push(lnext); }
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
					ii = 0; while (!Qu.empty()) Qu.pop(); while (!QN.empty()) QN.pop();
					if (cofactor) {
						if (mpz_probab_prime_p(N1, 30) == 0) {  // cofactor definitely composite
							
							QN.push(&N1); Qu.push(2); Qu.push(1); Qu.push(0); Qu.push(3);
							while (!QN.empty()) {
								mpz_t* N = QN.top(); QN.pop();
								int l = Qu.top(); Qu.pop();
								int j = 0;
								bool factored = false;
								while (!factored) {
									int alg = Qu.top(); Qu.pop(); j++;
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
										while (lt--) { algarr[lt] = Qu.top(); Qu.pop(); }
										lt = lnext; if (lt) { while (lt--) Qu.push(algarr[lnext-1-lt]); Qu.push(lnext); }
										if (mpz_probab_prime_p(p1, 30)) {
											if (mpz_cmpabs(p1, lpb) > 0) { isrel = false; break; }
											else { mpz_get_str(str2, BASE, p1); str += str2; str += ","; }
										}
										else {
											if (!lnext) { isrel = false; break; }
											mpz_set(pi[ii], p1);
											QN.push(&pi[ii++]);
											lt = lnext; if (lt) { while (lt--) Qu.push(algarr[lnext-1-lt]); Qu.push(lnext); }
										}
										if (mpz_probab_prime_p(p2, 30)) {
											if (mpz_cmpabs(p2, lpb) > 0) { isrel = false; break; }
											else { mpz_get_str(str2, BASE, p2); str += str2; str += QN.empty() ? "" : ","; }
										}
										else {
											if (!lnext) { isrel = false; break; }
											mpz_set(pi[ii], p2);
											QN.push(&pi[ii++]);
											lt = lnext; if (lt) { while (lt--) Qu.push(algarr[lnext-1-lt]); Qu.push(lnext); }
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

					if (isrel) { cout << str << endl; T++; }
				}
			}
		}
		timetaken = ( clock() - start ) / (double) CLOCKS_PER_SEC;
		cout << "# Finished! Cofactorization took " << timetaken << "s" << endl;
		cout << "# " << T << " actual relations found." << endl;
	}

	mpz_clear(g1);
	mpz_clear(B); mpz_clear(A);
	gmp_randclear(state);

	delete[] L;
	for (int i = 0; i < d - 2; i++) {
		mpz_clear(Bk[i]);
		mpz_clear(Ak[i]);
	}
	delete[] Ak; delete[] Bk;

	free(str1);
	free(str2);
	mpz_clear(S);
	mpz_clear(t); mpz_clear(p2); mpz_clear(p1);
	mpz_clear(factor);
	mpz_clear(lpb);
	mpz_clear(N1); mpz_clear(N0);
	mpz_poly_clear(i1); mpz_poly_clear(f1); mpz_poly_clear(f0);
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

int64_t rel2A(int d, mpz_t* Ak, int64_t* L, int64_t relid, int bb)
{
	int64_t BB = 1<<bb;
	int64_t hB = 1<<(bb-1);
	mpz_t t; mpz_init_set_ui(t, 0);
	// compute v = L*(vector from reli)
	int* v = new int[d]();
	int* u = new int[d]();
	for (int i = 0; i < d; i++) u[i] = (relid >> (bb*i)) % BB - hB;
	for (int j = 0; j < d; j++) {
		for (int i = 0; i < d; i++) {
			v[j] += L[j*d + i] * u[i];
		}
	}		
	int64_t A = 0;
	for (int j = 0; j < d-2; j++) {		
		mpz_mul_si(t, Ak[j], v[j]);
		A += mpz_get_si(t);
	}
	A += v[d-2];
	delete[] u;
	delete[] v;
	mpz_clear(t);
	return A;
}

int64_t rel2B(int d, mpz_t* Bk, int64_t* L, int64_t relid, int bb)
{
	int64_t BB = 1<<bb;
	int64_t hB = 1<<(bb-1);
	mpz_t t; mpz_init_set_ui(t, 0);
	// compute v = L*(vector from reli)
	int* v = new int[d]();
	int* u = new int[d]();
	for (int i = 0; i < d; i++) u[i] = (relid >> (bb*i)) % BB - hB;
	for (int j = 0; j < d; j++) {
		for (int i = 0; i < d; i++) {
			v[j] += L[j*d + i] * u[i];
		}
	}		
	int64_t B = 0;
	for (int j = 0; j < d-2; j++) {
		mpz_mul_si(t, Bk[j], v[j]);
		B += mpz_get_si(t);
	}
	B -= v[d-1];
	delete[] u;
	delete[] v;
	mpz_clear(t);
	return B;
}

void slcsieve(int d, mpz_t* Ak, mpz_t* Bk, int Bmin, int Bmax, int Rmin, int Rmax,
	 int* sieve_p, int* sieve_r, int* sieve_n, int degf, keyval* M, uint64_t* m,
	int mbb, int bb, int64_t Q0, int64_t R0, mpz_poly f)
{
	ZZ_mat<mpz_t> L;
	ZZ_mat<mpz_t> QLinv;
	ZZ_mat<mpz_t> L2;
	ZZ_mat<mpz_t> L3;
	ZZ_mat<mpz_t> C;
	ZZ_mat<mpz_t> LC;
	L.resize(d, d);
	QLinv.resize(d, d);
	L2.resize(d, d);
	L3.resize(d, d);
	C.resize(d, d);
	LC.resize(d, d);
	int n = d;
	int dn = d*n;
	int dd = d*d;
	int64_t* L4 = new int64_t[dd];
	int64_t* L5 = new int64_t[dd];
	enumvar* v1 = new enumvar(d, n);
	mpz_t t; mpz_init(t);
	mpz_t PQz; mpz_init(PQz);
	mpz_t Rrz; mpz_init(Rrz);
	mpz_t Rriz; mpz_init(Rriz);
	__int128 Q = static_cast<__int128>(Q0);
	__int128 R = static_cast<__int128>(R0);

	// compute L
	for (int k = 0; k < d; k++) {
		for (int l = 0; l < d; l++) {
			if (l == d - 1 && k < d-2) {
				// reduce Ak*x + Bk mod Q0
				__int128 AmodQ = static_cast<__int128>(mpz_fdiv_ui(Ak[k], Q0));
				__int128 BmodQ = static_cast<__int128>(mpz_fdiv_ui(Bk[k], Q0));
				__int128 Ri = (AmodQ*R + BmodQ) % Q;
				L[k][l] = Ri;
			}
			else if (k == l) {
				L[k][l] = 1;
			}
			else {
				L[k][l] = 0;
			}
		}
	}
	// last basis vector gets x - R
	L[d-2][d-1] = R;
	L[d-1][d-1] = Q;

	// reduce L using LLL
	lll_reduction(L, LLL_DEF_DELTA, LLL_DEF_ETA, LM_HEURISTIC, FT_DEFAULT,
		0, LLL_DEFAULT);

	// compute QLinv = Q*L^-1 = det(L)*L^-1 = matadj(L)
	matadj(d, L, C, LC, QLinv, t);

	__int128* Amodp = new __int128[d-2];
	__int128* Bmodp = new __int128[d-2];
	__int128* Amodq = new __int128[d-2];
	__int128* Bmodq = new __int128[d-2];
	for (int k = 0; k < d - 2; k++) {
		// reduce Ak*x + Bk mod Q
		Amodq[k] = static_cast<__int128>(mpz_fdiv_ui(Ak[k], Q0));
		Bmodq[k] = static_cast<__int128>(mpz_fdiv_ui(Bk[k], Q0));
	}

	int bmax = 40;
	int64_t* blacklist = new int64_t[bmax];
	//int numb = get_blacklist(L, d, f, degf, QLinv, Ak, Bk, Q0, R0, Rmin + 30, bb, bmax, blacklist);
	int numb = 0;
	cout << "# Number of blacklist vectors: " << to_string(numb) << endl;

	int i = 0;
	while (sieve_p[i] < Bmin) i++;
	int imin = i;
	int64_t nntotal = 0;
	int Rcurrent = Rmin;
	int64_t p = sieve_p[i];
	uint8_t logp = log2f(p);
	uint64_t mj = 0;
	for (int j = 0; j < 512; j++, mj += (1ul<<(mbb-9))) m[j] = mj;
	int64_t gap = (int64_t)(((double)Bmax - p) / 10.0);
	int pc = 0;
	int64_t nextmark = p + gap;
	while (p < Bmax) {
		int ni = sieve_n[i];
		for (int k = 0; k < d - 2; k++) {
			// reduce Ak*x + Bk mod p
			Amodp[k] = static_cast<__int128>(mpz_fdiv_ui(Ak[k], p));
			Bmodp[k] = static_cast<__int128>(mpz_fdiv_ui(Bk[k], p));
		}
		__int128 qinvmodp = static_cast<__int128>(modinv(Q0, p));
		__int128 pinvmodq = static_cast<__int128>(modinv(p, Q0));
		__int128 P = static_cast<__int128>(p);
		__int128 PQ = P*Q;
		mpz_set_ui(PQz, Q0); mpz_mul_ui(PQz, PQz, p);
		for (int j = 0; j < ni; j++) {
			int r = sieve_r[i*degf+j];
			__int128 r1 = static_cast<__int128>(r);

			// construct sieving lattice for this p
			for (int k = 0; k < d; k++) {
				for (int l = 0; l < d; l++) {
					if (l == d - 1 && k < d-2) {
						__int128 ri = (Amodp[k]*r1 + Bmodp[k]) % P;
						__int128 Ri = (Amodq[k]*R + Bmodq[k]) % Q;
						// compute Rri with CRT
						__int128 Rri = Q * ((ri * qinvmodp) % P) + P * ((Ri * pinvmodq) % Q);
						if (Rri > PQ) Rri -= PQ;
						mpz_set_uint128(Rriz, Rri);
						mpz_set(L2(k, l).get_data(), Rriz);
					}
					else if (k == l) {
						L2[k][l] = 1;
					}
					else {
						L2[k][l] = 0;
					}
				}
			}
			// last basis vector gets x - Rr
			__int128 Rr = Q * ((r1 * qinvmodp) % P) + P * ((R * pinvmodq) % Q);
			if (Rr > PQ) Rr -= PQ;
			mpz_set_uint128(Rrz, Rr);
			mpz_set(L2(d-2,d-1).get_data(), Rrz);
			mpz_set(L2(d-1,d-1).get_data(), PQz);

			// reduce L2 using fplll
			lll_reduction(L2, LLL_DEF_DELTA, LLL_DEF_ETA, LM_HEURISTIC, FT_DEFAULT,
				0, LLL_DEFAULT);
			
			// compute L3
			matmul(d, L3, QLinv, L2, t);
			matdivexact_ui(d, L3, Q0);

			// convert column-major L3 to flattened row-major L4
			for (int k = 0; k < d; k++)
				for (int l = 0; l < d; l++)
					L4[k*d + l] = L3(l, k).get_si();

			// computing sieving lattice L5 by extracting valid basis vectors from L4
			n = 0;
			for (int k = 0; k < d; k++) {
				int64_t A = 0; int64_t B = 0;
				for (int l = 0; l < d-2; l++) {
					A += L2(k, l).get_si() * mpz_get_si(Ak[l]);
					B += L2(k, l).get_si() * mpz_get_si(Bk[l]);
				}
				A += L2(k, d-2).get_si();
				B -= L2(k, d-1).get_si();
				if (A != 0 && B != 0) {
					for (int l = 0; l < d; l++) L5[l*d + n] = L4[l*d + k];
					n++;
				}
			}				
			
			// enumerate all vectors up to radius R in L5, up to a max of 1000 vectors
			int nn = enumeratehd(d, n, L5, M, m, logp, p, Rcurrent, numb, blacklist, 1000,
				mbb, bb, v1);
			nntotal += nn;
		}
		
		// advance to next p
		i++;
		p = sieve_p[i];
		if (p > nextmark && pc < 90) {
			pc += 10;
			cout << "# " << pc << "\% of primes sieved..." << endl;
			nextmark += gap;
		}

		Rcurrent = (int)(Rmin + (Rmax - Rmin)*((double)p - Bmin)/((double)Bmax - Bmin));
	}
	cout << "# 100\% of primes sieved." << endl;
	cout << "# Average of " << (int)((double)nntotal/(i-imin)) << " lattice points per prime." << endl;

	// clear memory
	delete[] blacklist;
	delete[] Bmodq; delete[] Amodq; delete[] Bmodp; delete[] Amodp;
	mpz_clear(Rriz); mpz_clear(Rrz); mpz_clear(PQz); mpz_clear(t);
	delete v1;
	delete[] L5; delete[] L4;;
}

void mpz_set_uint128(mpz_t z, __int128 a)
{
	uint64_t hilo[2] = { static_cast<uint64_t>(a >> 64), static_cast<uint64_t>(a) };
	/* Initialize z and a */
	mpz_import(z, 2, 1, sizeof(uint64_t), 0, 0, hilo);
}

void matmul(int d, ZZ_mat<mpz_t>& C, ZZ_mat<mpz_t>& A, ZZ_mat<mpz_t>& B, mpz_t &t)
{
    // Perform matrix multiplication C = A * B
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            mpz_set_ui(C(j, i).get_data(), 0);  // C[i][j] = 0
            for (int k = 0; k < d; k++) {
                mpz_mul(t, A(k, i).get_data(), B(j, k).get_data());  // t = A[i][k] * B[k][j]
                mpz_add(C(j, i).get_data(), C(j, i).get_data(), t);  // C[i][j] += t
            }
        }
    }
}

void matdivexact_ui(int d, ZZ_mat<mpz_t>& A, uint64_t Q)
{
    // divide all entries of A exactly by Q in-place
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
        	mpz_divexact_ui(A(i, j).get_data(), A(i, j).get_data(), Q);
        }
    }
}

void matadj(int d, ZZ_mat<mpz_t> &M, ZZ_mat<mpz_t> &C, ZZ_mat<mpz_t> &MC,
	ZZ_mat<mpz_t> &Madj, mpz_t &t)
{
	int dd = d*d;
	C.fill(0);
	for (int k = 0; k < d; k++) C[k][k] = 1;
    MC.fill(0);
    int i = 0;
    __int128 ai;
    while (true) {
        i++;
        matmul(d, MC, M, C, t);
        if (i == d) {
            ai = 0;
			for (int k = 0; k < d; k++) ai += MC[k][k].get_si();
			ai = -ai / d;
            Madj = C;
			negmat(d, MC);
            break;
        }
        C = MC;
		ai = 0;
		for (int k = 0; k < d; k++) ai += C[k][k].get_si();
		ai = -ai / i;
		for (int k = 0; k < d; k++) {
			// there is no mpz_add_si in gnu-mp
			if (ai >= 0)
				mpz_add_ui(C(k, k).get_data(), C(k, k).get_data(), ai);
			else
				mpz_sub_ui(C(k, k).get_data(), C(k, k).get_data(), -ai);
		}
    }
}

void negmat(int d, ZZ_mat<mpz_t> &A)
{
	for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            mpz_neg(A(i, j).get_data(), A(i, j).get_data());  // A[i][j] = -A[i][j]
        }
    }
}

void printZZ_mat(ZZ_mat<mpz_t> &L, int d, int n, int pr, int w)
{
	for (int j = 0; j < d; j++) {
		for (int i = 0; i < n; i++) {
			cout << fixed << setprecision(pr) << setw(w) << mpz_get_str(NULL, 10, L(i, j).get_data()) << (i<n-1?",":"") << "\t";
		}
		cout << ";\\" << endl << flush;
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

int enumeratehd(int d, int n, int64_t* L, keyval* M, uint64_t* m, uint8_t logp, int64_t p,
	int R, int bmax, int64_t* blacklist, int nnmax, int mbb, int bb, enumvar* v1)
{
	int64_t hB = 1<<(bb-1);
	int mmax = (1ul<<mbb)/512;
	for (int i = 0; i < d*d; i++) v1->uu[i] = 0;

	// Gram-Schmidt orthogonalization
	int64_t* borig = L;
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < d; k++) {
			v1->uu[k*d + k] = 1;
			v1->b[k*n + i] = (float)borig[k*d + i];
		}
		for (int j = 0; j < i; j++) {
			float dot1 = 0;
			float dot2 = 0;
			for (int k = 0; k < d; k++) {
				dot1 += borig[k*d + i] * v1->b[k*n + j];
				dot2 += v1->b[k*n + j] * v1->b[k*n + j];
			}
			v1->uu[j*d + i] = dot1 / dot2;
			for (int k = 0; k < d; k++) {
				v1->b[k*n + i] -= v1->uu[j*d + i] * v1->b[k*n + j];
			}
		}
	}

	// compute orthogonal basis vector norms
	for (int i = 0; i < n; i++) {
		float N = 0;
		for (int k = 0; k < d; k++) N += v1->b[k*n + i] * v1->b[k*n + i];
		v1->bnorm[i] = sqrt(N);
	}

	// set up variables
	for (int i = 0; i < d; i++)
		v1->common_part[i] = 0;
	for (int k = 0; k < d + 1; k++) {
		for (int j = 0; j < n; j++)
			v1->sigma[k*d + j] = 0;
		v1->rhok[k] = 0;
		v1->rk[k] = k;
		if (k < d) {
			v1->vk[k] = 0;
			v1->ck[k] = 0;
			v1->wk[k] = 0;
		}
	}
	v1->vk[0] = 1;
	int t = 1;
	int last_nonzero = 0;
	for (int l = t; l < n; l++) {
		for (int j = 0; j < d; j++) {
			v1->common_part[j] += v1->vk[l] * borig[j*d + l];
		}
	}
	int k = 0;
	int nn = 0;
	// enumerate lattice vectors (x,y,z,r,s,t) in sphere with radius R
	while (true) {
		v1->rhok[k] = v1->rhok[k+1] + (v1->vk[k] - v1->ck[k]) * (v1->vk[k] - v1->ck[k]) * 
			v1->bnorm[k] * v1->bnorm[k];
		if (v1->rhok[k] - 0.00005 <= R*R) {
			if (k == 0) {
				if (last_nonzero != 0 || nn == 0) {
					memset(v1->c, 0, 8*d);  // note:  typeof(v1->c) is int64_t, 8 bytes
					bool keep = true;
					bool iszero = true;
					for (int j = 0; j < d; j++) {
						v1->c[j] = 0;
						for (int i = 0; i <= t - 1; i++) {
							v1->c[j] += v1->vk[i] * borig[j*d + i] + v1->common_part[j];
							if (abs(v1->c[j]) >= hB) { keep = false; break; }
							if (v1->c[j] != 0) iszero = false;
						}
						if (!keep) break;
					}
					if (v1->c[0] < 0) {	// keep only one of { c, -c } (same information)
						for (int j = 0; j < d; j++) v1->c[j] = -v1->c[j];
					}
					uint64_t id = 0;
					for (int l = 0; l < d; l++) id += (v1->c[l] + hB) << (l * bb);
					// do not keep blacklisted vectors
					for (int l = 0; l < bmax; l++) {
						if (id == blacklist[l]) {
							keep = false;
							break;
						}
					}
					// save vector
					if (keep && !iszero) {
						uint64_t mi = id % 509;	// number of buckets
						mi = (mi*mi*mi) % 509;
						M[m[mi]] = (keyval){ id, logp };
						m[mi]++; // we are relying on the TLB
						int64_t mstart = mi*(1ul<<(mbb-9));
						if (m[mi] - mstart >= mmax) {
							cout << "mmax = " << mmax << endl;
							cout << "mi = " << mi << endl;
							cout << "mstart = " << mstart << endl;
							cout << "m[mi] = " << m[mi] << endl;
							//for (int i = 0; i < 512; i++) {
							//	mstart = i*(1ul<<(mbb-9));
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
							cout << "Bucket overflow, likely memory corruption." << endl;
							FILE* out;
							out = fopen("bucket.out", "w+");
							fprintf(out, "bucket %d", mi);
							fprintf(out, "\n");
							for (int l = 0; l < m[mi] - mstart; l++) {
								fprintf(out, "%ld", M[mstart + l].id);
								fprintf(out, "\n");
							}
							fclose(out);
							exit(1);
						}
						nn++;
						//if (nn >= nnmax) break;
					}
				}
				if (v1->vk[k] > v1->ck[k]) v1->vk[k] = v1->vk[k] - v1->wk[k];
				else v1->vk[k] = v1->vk[k] + v1->wk[k];
				v1->wk[k]++;
			}
			else {
				k--;
				v1->rk[k] = max(v1->rk[k], v1->rk[k+1]);
				for (int i = v1->rk[k+1]; i >= k + 1; i--) {
					v1->sigma[i*d + k] = v1->sigma[(i + 1)*d + k] + v1->vk[i] * v1->uu[k*d + i];
				}
				v1->ck[k] = -v1->sigma[(k + 1)*d + k];
				int vk_old = v1->vk[k];
				v1->vk[k] = floor(v1->ck[k] + 0.5); v1->wk[k] = 1;
				if (k >= t && k < n) {
					for (int j = 0; j < d; j++) {
						v1->common_part[j] -= vk_old * borig[j*d + k];
						v1->common_part[j] += v1->vk[k] * borig[j*d + k];
					}
				}
			}
		}
		else {
			k++;
			if (k == n) break;
			v1->rk[k] = k;
			if (k >= last_nonzero) {
				last_nonzero = k;
				int vk_old = v1->vk[k];
				v1->vk[k]++;
				if (k >= t && k < n) {
					for (int j = 0; j < d; j++) {
						v1->common_part[j] -= vk_old * borig[j*d + k];
						v1->common_part[j] += v1->vk[k] * borig[j*d + k];
					}
				}
			}
			else {
				if (v1->vk[k] > v1->ck[k]) {
					int vk_old = v1->vk[k];
					v1->vk[k] = v1->vk[k] - v1->wk[k];
					if (k >= t && k < n) {
						for (int j = 0; j < d; j++) {
							v1->common_part[j] -= vk_old * borig[j*d + k];
							v1->common_part[j] += v1->vk[k] * borig[j*d + k];
						}
					}
				}
				else {
					int vk_old = v1->vk[k];
					v1->vk[k] = v1->vk[k] + v1->wk[k];
					if (k >= t && k < n) {
						for (int j = 0; j < d; j++) {
							v1->common_part[j] -= vk_old * borig[j*d + k];
							v1->common_part[j] += v1->vk[k] * borig[j*d + k];
						}
					}
				}
				v1->wk[k]++;
			}
		}
	}

	return nn;
}

int get_blacklist(ZZ_mat<mpz_t> &L, int d, mpz_poly f, int degf, ZZ_mat<mpz_t> &QLinv,
	mpz_t* Ak, mpz_t* Bk, int64_t Q0, int64_t R0, int Rmin, int bb, int bmax, int64_t* blacklist)
{
	ZZ_mat<mpz_t> L2;
	ZZ_mat<mpz_t> L3;
	L2.resize(d, d);
	L3.resize(d, d);
	int n = d;
	int dd = d*d;
	int64_t* L4 = new int64_t[dd];
	int64_t* L5 = new int64_t[dd];
	enumvar* v1 = new enumvar(d, n);
	mpz_t t; mpz_init(t);
	mpz_t PQz; mpz_init(PQz);
	mpz_t Rrz; mpz_init(Rrz);
	mpz_t Rriz; mpz_init(Rriz);
	int64_t p = 1ul << 60;
	mpz_t pz; mpz_init_set_ui(pz, p);
	int64_t* roots = new int64_t[degf]();
	int64_t* fmodp = new int64_t[degf+1]();
	int nr = 0;
	while (nr == 0) {
		mpz_add_ui(pz, pz, 1);
		mpz_nextprime(pz, pz);
		p = mpz_get_ui(pz);
		for (int i = 0; i <= degf; i++) {			
			mpz_poly_getcoeff(t, i, f);
			fmodp[i] = mpz_fdiv_ui(t, p);
		}
		nr = polrootsmod(fmodp, degf, roots, p);
	}
	int64_t r = roots[0];

	__int128* Amodp = new __int128[d-2];
	__int128* Bmodp = new __int128[d-2];
	__int128* Amodq = new __int128[d-2];
	__int128* Bmodq = new __int128[d-2];
	for (int k = 0; k < d - 2; k++) {
		// reduce Ak*x + Bk mod Q
		Amodq[k] = static_cast<__int128>(mpz_fdiv_ui(Ak[k], Q0));
		Bmodq[k] = static_cast<__int128>(mpz_fdiv_ui(Bk[k], Q0));
	}
	for (int k = 0; k < d - 2; k++) {
		// reduce Ak*x + Bk mod p
		Amodp[k] = static_cast<__int128>(mpz_fdiv_ui(Ak[k], p));
		Bmodp[k] = static_cast<__int128>(mpz_fdiv_ui(Bk[k], p));
	}
	__int128 qinvmodp = static_cast<__int128>(modinv(Q0, p));
	__int128 pinvmodq = static_cast<__int128>(modinv(p, Q0));
	__int128 P = static_cast<__int128>(p);
	__int128 Q = static_cast<__int128>(Q0);
	__int128 R = static_cast<__int128>(R0);
	__int128 PQ = P*Q;
	mpz_set_ui(PQz, Q0); mpz_mul_ui(PQz, PQz, p);
	__int128 r1 = static_cast<__int128>(r);

	// construct sieving lattice for this p
	for (int k = 0; k < d; k++) {
		for (int l = 0; l < d; l++) {
			if (l == d - 1 && k < d-2) {
				__int128 ri = (Amodp[k]*r1 + Bmodp[k]) % P;
				__int128 Ri = (Amodq[k]*R + Bmodq[k]) % Q;
				// compute Rri with CRT
				__int128 Rri = Q * ((ri * qinvmodp) % P) + P * ((Ri * pinvmodq) % Q);
				if (Rri > PQ) Rri -= PQ;
				mpz_set_uint128(Rriz, Rri);
				mpz_set(L2(k, l).get_data(), Rriz);
			}
			else if (k == l) {
				L2[k][l] = 1;
			}
			else {
				L2[k][l] = 0;
			}
		}
	}
	// last basis vector gets x - Rr
	__int128 Rr = Q * ((r1 * qinvmodp) % P) + P * ((R * pinvmodq) % Q);
	if (Rr > PQ) Rr -= PQ;
	mpz_set_uint128(Rrz, Rr);
	mpz_set(L2(d-2,d-1).get_data(), Rrz);
	mpz_set(L2(d-1,d-1).get_data(), PQz);

	// reduce L2 using fplll
	lll_reduction(L2, LLL_DEF_DELTA, LLL_DEF_ETA, LM_HEURISTIC, FT_DEFAULT,
		0, LLL_DEFAULT);
	
	// compute L3
	matmul(d, L3, QLinv, L2, t);
	matdivexact_ui(d, L3, Q0);

	// convert column-major L3 to flattened row-major L4
	for (int k = 0; k < d; k++)
		for (int l = 0; l < d; l++)
			L4[k*d + l] = L3(l, k).get_si();

	// computing blacklist lattice L5 by extracting invalid basis vectors from L4
	n = 0;
	for (int k = 0; k < d; k++) {
		int64_t A = 0; int64_t B = 0;
		for (int l = 0; l < d-2; l++) {
			A += L2(k, l).get_si() * mpz_get_si(Ak[l]);
			B += L2(k, l).get_si() * mpz_get_si(Bk[l]);
		}
		A += L2(k, d-2).get_si();
		B -= L2(k, d-1).get_si();
		if (A == 0 && B == 0) {
			for (int l = 0; l < d; l++) L5[l*d + n] = L4[l*d + k];
			n++;
		}
	}
	
	// enumerate all vectors up to radius R in L5, up to a max of 1000 vectors
	int mbb = 16;
	uint64_t m[512] = { 0 };
	uint64_t mj = 0;
	for (int j = 0; j < 512; j++, mj += (1ul<<(mbb-9))) m[j] = mj;
	keyval* M = new keyval[1ul << mbb]();
	for (int i = 0; i < 512; i++) {
		uint64_t mtop = i*(1ul<<(mbb-9));
		uint64_t mend = m[i];
	}
	int nn = enumeratehd(d, n, L5, M, m, 0, p, Rmin, 0, blacklist, 1000, mbb, bb, v1);
	if (nn >= bmax) {
		cout << "Too many vectors to blacklist.  Increase Amax/Bmax.  Exiting..." << endl;
		exit(1);
	}

	int k = 0;
	for (int i = 0; i < 512; i++) {
		int64_t mstart = i*(1ul<<(mbb-9));
		for (int j = mstart; j < m[i]; j++) {
			blacklist[k++] = M[j].id;
		}
	}

	// free memory
	delete[] M;
	delete[] Bmodq; delete[] Amodq; delete[] Bmodp; delete[] Amodp;
	delete[] fmodp; delete[] roots;
	mpz_clear(Rriz); mpz_clear(Rrz); mpz_clear(PQz); mpz_clear(t);
	delete v1;
	delete[] L5; delete[] L4;;
	
	return k;
}

inline bool bucket_sorter(keyval const& kv1, keyval const& kv2)
{
	return kv2.id < kv1.id;
}

inline int64_t modinv(int64_t x, int64_t m)
{
    int64_t m0 = m, t, q;
    int64_t y0 = 0, y = 1;
    if (m == 1) return 1;
    while (x > 1) {
        q = x / m;
        t = m, m = x % m, x = t;
        t = y0, y0 = y - q * y0, y = t;
    }
    if (y < 0) y += m0;
    return y;
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


