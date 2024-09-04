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

__int128 MASK64;

int enumeratehd(int d, int n, int64_t* L, int N1, int N2, int nnmax, int bb, enumvar* v1);
int l2norm(ZZ_mat<mpz_t> &L, int d, int k);
inline int max(int u, int v);
void mpz_set_uint128(mpz_t z, __int128 a);
void matmul(int d, ZZ_mat<mpz_t>& C, ZZ_mat<mpz_t>& A, ZZ_mat<mpz_t>& B, mpz_t &t);
void matdivexact_ui(int d, ZZ_mat<mpz_t>& A, uint64_t Q);
void matadj(int d, ZZ_mat<mpz_t> &M, ZZ_mat<mpz_t> &C, ZZ_mat<mpz_t> &MC,
	ZZ_mat<mpz_t> &Madj, mpz_t &t);
void negmat(int d, ZZ_mat<mpz_t> &A);
int64_t rel2A(int d, mpz_t* Ak, int64_t* L, int64_t relid, int bb);
int64_t rel2B(int d, mpz_t* Bk, int64_t* L, int64_t relid, int bb);
void printZZ_mat(ZZ_mat<mpz_t> &L, int d, int n, int pr, int w);
void printvector(int d, uint64_t v, int hB);
void printvectors(int d, vector<uint64_t> &M, int n, int hB);
inline int64_t modinv(int64_t x, int64_t m);
inline int64_t gcd(int64_t a, int64_t b);
inline __int128 gcd128(__int128 a, __int128 b);

int main(int argc, char** argv)
{
	// set constant
	MASK64 = 1L;
	MASK64 = MASK64 << 64;
	MASK64 = MASK64 - 1L;

	if (argc != 12) {
		cout << endl << "Usage: ./showlattice inputpoly d Q R p r" << endl << endl;
		cout << "    inputpoly       input polynomial in N/skew/C0..Ck/Y0..Y1 format" << endl;
		cout << "    d               sieving dimension" << endl;
		cout << "    Amax            upper bound for A in A*x + B ideal generator" << endl;
		cout << "    Bmax            upper bound for B in A*x + B ideal generator" << endl;
		cout << "    Q               special-Q (can be up to 2^64)" << endl;
		cout << "    R               root of sieving polynomial mod Q" << endl;
		cout << "    p               sieving prime p (can be up to 2^64)" << endl;
		cout << "    r               root of sieving polynomial mod p" << endl;
		cout << "    N1              count vectors with norm >= N1" << endl;
		cout << "    N2              enumerate vectors with norm <= N2" << endl;
		cout << "    bb              absolute coordinate bit length" << endl;
		cout << endl;
		return 0;
	}

	cout << "# ";
	for (int i = 0; i < argc; i++) cout << argv[i] << " ";
	cout << endl;

	bool verbose = false;
		
	if (verbose) cout << endl << "Reading input polynomial in file " << argv[1] << "..." << flush;
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
	file.close();
	if (verbose) cout << endl << "Complete.  Degree f0 = " << degf << ", degree f1 = " << degg << "." << endl;

	mpz_poly f0; mpz_poly f1; mpz_poly i1;
	mpz_poly_init(f0, degf); mpz_poly_init(f1, degg); mpz_poly_init(i1, 3);
	mpz_poly_set_mpz(f0, fpoly, degf);
	mpz_poly_set_mpz(f1, gpoly, degg);
	char* str1 = (char*)malloc(20*sizeof(char));
	char* str2 = (char*)malloc(20*sizeof(char));

	int d = atoi(argv[2]);
	mpz_t maxA, maxB;
	mpz_init_set_str(maxA, argv[3], 10);
	mpz_init_set_str(maxB, argv[4], 10);
	int64_t Q0 = strtoll(argv[5], NULL, 10);
	int64_t RR = strtoll(argv[6], NULL, 10);
	__int128 Q = static_cast<__int128>(Q0);
	__int128 R = static_cast<__int128>(RR);

	gmp_randstate_t state;
	gmp_randinit_default(state);
	gmp_randseed_ui(state, 123ul);

	// construct array to hold d - 2 elements of Z[x]
	mpz_t* Ak = new mpz_t[d - 2];
	mpz_t* Bk = new mpz_t[d - 2];
	for (int k = 0; k < d - 2; k++) {
		mpz_init_set_ui(Ak[k], 0);
		mpz_init_set_ui(Bk[k], 0);
	}
	// generate d - 2 random elements A*x + B
	cout << "ei = [\\" << endl;
	for (int k = 0; k < d - 2; k++) {
		mpz_urandomm(Ak[k], state, maxA);
		mpz_urandomm(Bk[k], state, maxB);
		mpz_get_str(str1, 10, Ak[k]);
		mpz_get_str(str2, 10, Bk[k]);
		cout << str1 << "*x + " << str2;
		if (k < d - 3) cout << ",\\" << endl;
	}
	cout << "]" << endl << endl;

	// compute L
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
	L[d-2][d-1] = RR;
	L[d-1][d-1] = Q0;

	// reduce L using LLL
	lll_reduction(L, LLL_DEF_DELTA, LLL_DEF_ETA, LM_HEURISTIC, FT_DEFAULT,
		0, LLL_DEFAULT);

	// print special-Q lattice L
	cout << "special-Q lattice L = " << endl;
	printZZ_mat(L, d, d, 0, 3);
	cout << endl;

	cout << "norms of basis vectors are" << endl;
	for (int k = 0; k < d; k++) {
		cout << l2norm(L, d, k);
		if (k < d - 1) cout << "," << flush;
	}
	cout << endl << endl;

	enumvar* v1 = new enumvar(d, d);
	int dd = d*d;
	int64_t* L4 = new int64_t[dd];
	int64_t* L5 = new int64_t[dd];
	mpz_t t; mpz_init(t);
	mpz_t PQz; mpz_init(PQz);
	mpz_t Rrz; mpz_init(Rrz);
	mpz_t Rriz; mpz_init(Rriz);

	int64_t p = strtoll(argv[7], NULL, 10);
	int64_t r = strtoll(argv[8], NULL, 10);

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
	
	// compute QLinv = Q*L^-1 = det(L)*L^-1 = matadj(L)
	matadj(d, L, C, LC, QLinv, t);

	// compute L3
	matmul(d, L3, QLinv, L2, t);
	matdivexact_ui(d, L3, Q0);

	// convert column-major L3 to flattened row-major L4
	for (int k = 0; k < d; k++)
		for (int l = 0; l < d; l++)
			L4[k*d + l] = L3(l, k).get_si();

	// print sieving lattice L4
	cout << "Sieving lattice Lp = " << endl;
	printbasis(L4, d, d, 0, 3);
	cout << endl;

	cout << "norms of basis vectors are" << endl;
	for (int k = 0; k < d; k++) {
		cout << l2norm(L3, d, k);
		if (k < d - 1) cout << "," << flush;
	}
	cout << endl << endl;

	// computing sieving lattice L5 by extracting valid basis vectors from L4
	cout << "basis vectors which give A*x + B = 0 (indexed from zero): " << flush;
	int n = 0;
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
		else {
			cout << k << "," << flush;
		}
	}
	cout << endl;
	
	int N1 = atoi(argv[9]);
	int N2 = atoi(argv[10]);
	int bb = atoi(argv[11]);
	int64_t hB = 1<<(bb-1);
	int nn = enumeratehd(d, n, L5, N1, N2, 1000, bb, v1);
	cout << "reached " << nn << " vectors in [" << -hB << "," << hB << "]^" << d <<
		" in Lp giving nonzero A*x + B in norm range [" << N1 << "," << N2 << "]" <<
		endl << endl;

	// free memory
	delete[] L5; delete[] L4;
	delete v1;
	delete[] Bmodq; delete[] Amodq; delete[] Bmodp; delete[] Amodp;
	gmp_randclear(state);
	for (int i = 0; i < d - 2; i++) {
		mpz_clear(Bk[i]);
		mpz_clear(Ak[i]);
	}
	delete[] Ak; delete[] Bk;
	free(str2);
	free(str1);
	mpz_clear(t);
	mpz_poly_clear(f1); mpz_poly_clear(f0);
	mpz_clear(maxB); mpz_clear(maxA);
	for (int i = 0; i < 20; i++) {
		mpz_clear(gpoly[i]);
		mpz_clear(fpoly[i]);
	}
	delete[] gpoly;
	delete[] fpoly;

	return 0;
}

int l2norm(ZZ_mat<mpz_t> &L, int d, int k)
{
	int n = 0;
	for (int i = 0; i < d; i++) n += L(k, i).get_si() * L(k, i).get_si();
	return n;
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
			cout << fixed << setprecision(pr) << setw(w) <<
				mpz_get_str(NULL, 10, L(i, j).get_data()) << (i<n-1?",":"") << "\t";
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


int enumeratehd(int d, int n, int64_t* L, int N1, int N2, int nnmax, int bb, enumvar* v1)
{
	int64_t hB = 1<<(bb-1);
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
		double N = v1->rhok[k] - 0.00005;
		if (N <= N2) {
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
					if (N < N1) keep = false;
					// save vector
					if (keep && !iszero) {
						//for (int j = 0; j < d; j++) cout << v1->c[j] << "," << flush;
						//cout << endl;
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
