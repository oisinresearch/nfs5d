#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <gmpxx.h>
#include <cmath>
#include <fstream>
#include <ctime>
#include <cstring>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stack>
#include <string>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <utility>

#include "intpoly.h"
#include "mpz_poly.h"
#include "L2lu64.h"

using namespace std;

#if defined(__SIZEOF_INT128__)
typedef unsigned __int128 uint128_t;
typedef __int128 int128_t;
#else
#error "128-bit integers are required for this NFS build."
#endif

int128_t MASK64;

// Use two 256-bit registers to represent the 8-dimension vector
struct vec8 {
    __m256i lo;  // coords 0-3
    __m256i hi; // coords 4-7
};

// ==================== DATA STRUCTURES ====================
struct KeyVal {
    uint64_t id;
    uint8_t logp;
};

struct Hit {
    int64_t coords[8];
    uint64_t norm;
};

struct SieveSide {
    int k;
    vector<int> p;
    vector<int> r;
    vector<int> n;
    vector<int> r_offset;
    uint8_t threshold;
};

struct GrayWorkspace {
    vector<pair<int, int>> steps;

    GrayWorkspace(int d) {
        build_gray(d - 1, 1, steps);
    }

    // Recursively generates the sequence of AVX additions/subtractions
    // to traverse the entire {-1, 0, 1}^d space, changing 1 dimension at a time.
    void build_gray(int d_idx, int dir, vector<pair<int, int>>& st) {
        if (d_idx == 0) {
            st.push_back({0, dir});
            st.push_back({0, dir});
            return;
        }
        build_gray(d_idx - 1, dir, st);
        st.push_back({d_idx, dir});
        build_gray(d_idx - 1, -dir, st);
        st.push_back({d_idx, dir});
        build_gray(d_idx - 1, dir, st);
    }
};

struct SieveWorkspace {
    vector<double> b, uu, bnorm, sigma, rhok, ck;
    vector<int> rk, vk, wk, common_part;
    vector<int64_t> c;
    L2Workspace l2ws;
    GrayWorkspace gws;
    uint64_t m_idx;    // Single pointer for linear filling

    SieveWorkspace(int d)
        : b(d*d), uu(d*d), bnorm(d), sigma((d+1)*d), rhok(d+1),
          ck(d), rk(d+1), vk(d), wk(d), common_part(d), c(d),
          l2ws(d, d), gws(d), m_idx(0) {}
};

// ==================== FORWARD DECLARATIONS ====================
int64_t rel2A(int d, mpz_ptr* Ai, int64_t reli, int bb);
int64_t rel2B(int d, mpz_ptr* Bi, int64_t reli, int bb);
void slcsieve(int d, mpz_ptr* Ak, mpz_ptr* Bk, int Bmin, int Bmax, int Rmin, int Rmax,
              const SieveSide& side, int degf, KeyVal* M, int mbb, int bb, SieveWorkspace& ws);
int enumeratehdgray(int d, int64_t* L, KeyVal* M, uint8_t logp,
                    int bb, SieveWorkspace& ws, int max_keep);
int enumeratehd(int d, int n, int64_t* L, KeyVal* M, uint64_t* m, uint8_t logp, int64_t p,
                int R, SieveWorkspace& ws, int mbb, int bb);
inline int64_t gcd(int64_t a, int64_t b);
void GetlcmScalar(int B, mpz_t S, int* primes, int nump);
bool PollardPm1(mpz_ptr N, mpz_t S, mpz_t factor);
bool PollardPm1_mpz(mpz_t N, mpz_t S, mpz_t factor);
bool PollardPm1_int128(int128_t N, mpz_t S, int128_t &factor, int64_t, int64_t);
bool EECM(mpz_ptr N, mpz_t S, mpz_t factor, int d, int a, int X0, int Y0, int Z0);
bool EECM_mpz(mpz_t N, mpz_t S, mpz_t factor, int d, int a, int X0, int Y0, int Z0);
bool EECM_int128(int128_t N, mpz_t S, int128_t &factor, int d, int a, int X0, int Y0, int Z0, int64_t, int64_t);

// ==================== I/O HELPERS ====================

void parse_polynomial(const char* filename, mpz_poly f0, mpz_poly f1, double &skew, int &degf, int &degg) {
    ifstream file(filename);
    if (!file.is_open()) { cerr << "Error: Could not open poly file." << endl; exit(1); }
    string line;
    degf = 0; degg = 0;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t colon = line.find(':');
        if (colon == string::npos) continue;
        string key = line.substr(0, colon);
        string val_str = line.substr(colon + 1);
        val_str.erase(0, val_str.find_first_not_of(" \t"));
        if (key == "skew") {
            skew = stod(val_str);
        } else if (key[0] == 'c') {
            int idx = stoi(key.substr(1));
            degf = max(degf, idx);
            mpz_t v; mpz_init_set_str(v, val_str.c_str(), 10);
            mpz_poly_setcoeff(f0, idx, v);
            mpz_clear(v);
        } else if (key[0] == 'Y') {
            int idx = stoi(key.substr(1));
            degg = max(degg, idx);
            mpz_t v; mpz_init_set_str(v, val_str.c_str(), 10);
            mpz_poly_setcoeff(f1, idx, v);
            mpz_clear(v);
        }
    }
    f0->deg = degf; f1->deg = degg;
}

void load_factor_base(const char* filename, SieveSide* sides, uint8_t* th) {
    ifstream fbfile(filename);
    if (!fbfile.is_open()) { cerr << "Error: Could not open FB file." << endl; exit(1); }
    string line;
    getline(fbfile, line); // Skip header
    for (int s = 0; s < 2; s++) {
        getline(fbfile, line);
        sides[s].k = stoi(line);
        sides[s].threshold = th[s];
        int offset = 0;
        for (int i = 0; i < sides[s].k; i++) {
            getline(fbfile, line);
            stringstream ss(line);
            string val;
            getline(ss, val, ',');
            int p = stoi(val);
            sides[s].p.push_back(p);
            int count = 0;
            while (getline(ss, val, ',')) {
                sides[s].r.push_back(stoi(val));
                count++;
            }
            sides[s].n.push_back(count);
            sides[s].r_offset.push_back(offset);
            offset += count;
        }
    }
}

// ==================== SIEVE LOGIC ====================

inline bool bucket_sorter(const KeyVal& a, const KeyVal& b) { return a.id < b.id; }

void collect_candidates(uint8_t threshold, KeyVal* M, uint64_t m_count, 
                        int d, mpz_ptr* Ai, mpz_ptr* Bi, int bb, vector<int64_t>& rel_list)
{
    if (m_count == 0) return;

    // Now we sort the entire populated buffer at once
    sort(M, M + m_count, bucket_sorter);

    uint64_t lastid = M[0].id;
    int sumlogp = M[0].logp;

    for (uint64_t i = 1; i < m_count; i++) {
        if (M[i].id == lastid) {
            sumlogp += M[i].logp;
        } else {
            if (sumlogp > threshold) {
                int64_t A64 = rel2A(d, Ai, lastid, bb);
                int64_t B64 = rel2B(d, Bi, lastid, bb);
                int64_t g_val = gcd(A64, B64);
                if (A64 != 0 && B64 != 0 && abs(A64 / g_val) != 1) {
                    rel_list.push_back(lastid);
                }
            }
            lastid = M[i].id;
            sumlogp = M[i].logp;
        }
    }
    // Check final item
    if (sumlogp > threshold) rel_list.push_back(lastid);
}

void slcsieve(int d, mpz_ptr* Ak, mpz_ptr* Bk, int Bmin, int Bmax, int Rmin, int Rmax,
              const SieveSide& side, int degf, KeyVal* M, int mbb, int bb, SieveWorkspace& ws)
{
    vector<int64_t> L(d * d);
    int lastrow = (d - 1) * d;
    ws.m_idx = 0;
    int kmax = side.k;

    int nn = 0;
    int i = 0;
    while (i < kmax && side.p[i] < Bmin) i++;
    while (i < kmax && side.p[i] < Bmax) {
        int p = side.p[i];
        uint8_t logp = (uint8_t)log2(p);
        int ni = side.n[i];
        
        for (int j = 0; j < ni; j++) {
            int r = side.r[side.r_offset[i] + j];
            fill(L.begin(), L.end(), 0LL);
            for (int k = 0; k < d; k++) L[k * d + k] = 1;
            for (int k = 0; k < d - 2; k++) {
                L[lastrow + k] = (mpz_fdiv_ui(Ak[k], p) * r + mpz_fdiv_ui(Bk[k], p)) % p;
            }
            L[lastrow + d - 2] = r; 
            L[lastrow + d - 1] = p;

            int64L2(L.data(), d, d, ws.l2ws);
            
            nn = enumeratehdgray(d, L.data(), M, logp, bb, ws, 50);
        }
        i++;
        
        // Safety break to prevent buffer overflow of M
        if (ws.m_idx > (1ull << mbb) - 1000) break; 
    }
}

int enumeratehdgray(int d, int64_t* L, KeyVal* M, uint8_t logp,
                    int bb, SieveWorkspace& ws, int max_keep)
{
    if (d > 8) return 0;

    vec8 B[8];
    int64_t hB = 1LL << (bb - 1);
    int64_t limit = hB - 1;

    // 1. Setup Basis Vectors (Column-indexed, Row-major storage)
    for (int i = 0; i < d; i++) {
        alignas(32) int64_t temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int j = 0; j < d; j++) {
            // Your layout: Vector i is {L[i], L[i+d], L[i+2d]...}
            temp[j] = L[j * d + i];
        }
        B[i].lo = _mm256_load_si256((__m256i*)temp);
        B[i].hi = _mm256_load_si256((__m256i*)(temp + 4));
    }

    // 2. Initialize v = -sum(B_i)
    // Ensure this matches the start of your ws.gws.steps sequence!
    vec8 v = { _mm256_setzero_si256(), _mm256_setzero_si256() };
    for (int i = 0; i < d; i++) {
        v.lo = _mm256_sub_epi64(v.lo, B[i].lo);
        v.hi = _mm256_sub_epi64(v.hi, B[i].hi);
    }

    __m256i v_limit = _mm256_set1_epi64x(limit);
    __m256i v_nlimit = _mm256_set1_epi64x(-limit);
    uint64_t start_idx = ws.m_idx;

    auto evaluate_v = [&]() {
        __m256i gt_l = _mm256_cmpgt_epi64(v.lo, v_limit);
        __m256i lt_l = _mm256_cmpgt_epi64(v_nlimit, v.lo);
        __m256i gt_h = _mm256_cmpgt_epi64(v.hi, v_limit);
        __m256i lt_h = _mm256_cmpgt_epi64(v_nlimit, v.hi);

        __m256i out_any = _mm256_or_si256(_mm256_or_si256(gt_l, lt_l),
                                          _mm256_or_si256(gt_h, lt_h));

        if (_mm256_testz_si256(out_any, out_any)) {
            alignas(32) int64_t res[8];
            _mm256_store_si256((__m256i*)res, v.lo);
            _mm256_store_si256((__m256i*)(res + 4), v.hi);

            int first_nonzero_idx = -1;
            for (int i = 0; i < d; i++) {
                if (res[i] != 0) {
                    first_nonzero_idx = i;
                    break;
                }
            }

            if (first_nonzero_idx != -1) {
                if (res[first_nonzero_idx] < 0) {
                    for (int k = 0; k < d; k++) res[k] = -res[k];
                }

                uint64_t key = 0;
                for (int i = 0; i < d; i++) {
                    key ^= (uint64_t)(res[i] + hB) << (i * bb);
                }
                M[ws.m_idx++] = {key, logp};
            }
        }
    };

    // 3. Walk
    evaluate_v();
    for (const auto& step : ws.gws.steps) {
        // Assume step.first is index, step.second is +1 or -1
        if (step.second == 1) {
            v.lo = _mm256_add_epi64(v.lo, B[step.first].lo);
            v.hi = _mm256_add_epi64(v.hi, B[step.first].hi);
        } else {
            v.lo = _mm256_sub_epi64(v.lo, B[step.first].lo);
            v.hi = _mm256_sub_epi64(v.hi, B[step.first].hi);
        }
        evaluate_v();
    }

    uint64_t nn = ws.m_idx - start_idx;
    if (nn > (uint64_t)max_keep) {
        for (uint64_t i = 0; i < (uint64_t)max_keep; i++) {
            // Replace with a faster PRNG if available in your ws
            uint64_t j = i + ((start_idx + i) * 2654435761ULL) % (nn - i);
            std::swap(M[start_idx + i], M[start_idx + j]);
        }
        ws.m_idx = start_idx + max_keep;
        nn = max_keep;
    }

    return nn;
}

int enumeratehd(int d, int n, int64_t* L, KeyVal* M, uint64_t* m, uint8_t logp, int64_t p,
                int R, SieveWorkspace& ws, int mbb, int bb)
{
    int64_t hB = 1LL << (bb - 1);
    double R2 = (double)R * R;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < d; k++) {
            ws.uu[k * d + k] = 1.0;
            ws.b[k * n + i] = (double)L[k * n + i];
        }
        for (int j = 0; j < i; j++) {
            double dot1 = 0.0, dot2 = 0.0;
            for (int k = 0; k < d; k++) {
                dot1 += L[k * n + i] * ws.b[k * n + j];
                dot2 += ws.b[k * n + j] * ws.b[k * n + j];
            }
            ws.uu[j * d + i] = dot1 / dot2;
            for (int k = 0; k < d; k++) ws.b[k * n + i] -= ws.uu[j * d + i] * ws.b[k * n + j];
        }
    }
    for (int i = 0; i < n; i++) {
        double N = 0.0;
        for (int k = 0; k < d; k++) N += ws.b[k * n + i] * ws.b[k * n + i];
        ws.bnorm[i] = sqrt(N);
    }
    fill(ws.common_part.begin(), ws.common_part.end(), 0);
    fill(ws.rhok.begin(), ws.rhok.end(), 0.0);
    fill(ws.rk.begin(), ws.rk.end(), 0);
    fill(ws.vk.begin(), ws.vk.end(), 0);
    fill(ws.wk.begin(), ws.wk.end(), 0);
    fill(ws.ck.begin(), ws.ck.end(), 0.0);
    ws.vk[0] = 1;
    int t = 1, last_nonzero = 0, k = 0, nn = 0;
    for (int l = t; l < n; l++) for (int j = 0; j < d; j++) ws.common_part[j] += ws.vk[l] * L[j * n + l];
    while (true) {
        ws.rhok[k] = ws.rhok[k + 1] + (ws.vk[k] - ws.ck[k]) * (ws.vk[k] - ws.ck[k]) * ws.bnorm[k] * ws.bnorm[k];
        if (ws.rhok[k] <= R2 + 1e-5) {
            if (k == 0) {
                bool keep = true, iszero = true;
                for (int j = 0; j < d; j++) {
                    int64_t coord = ws.common_part[j];
                    for (int i = 0; i < t; i++) {
                        coord += ws.vk[i] * L[j * n + i];
                        if (abs(coord) >= hB) { keep = false; break; }
                        if (coord != 0) iszero = false;
                    }
                    if (!keep) break;
                    ws.c[j] = coord;
                }
                if (keep && !iszero) {
                    if (ws.c[0] < 0) for (int j = 0; j < d; j++) ws.c[j] = -ws.c[j];
                    uint64_t id = 0;
                    for (int l = 0; l < d; l++) id |= (uint64_t)(ws.c[l] + hB) << (l * (uint64_t)bb);
                    uint64_t mi = id % 509; // Simple hash
                    M[m[mi]] = {id, logp}; m[mi]++; nn++;
                }
                if (ws.vk[k] > ws.ck[k]) ws.vk[k] -= ws.wk[k]; else ws.vk[k] += ws.wk[k];
                ws.wk[k]++;
            } else {
                k--;
                ws.rk[k] = max(ws.rk[k], ws.rk[k + 1]);
                for (int i = ws.rk[k + 1]; i >= k + 1; i--) ws.sigma[i * n + k] = ws.sigma[(i + 1) * n + k] + ws.vk[i] * ws.uu[k * d + i];
                ws.ck[k] = -ws.sigma[(k + 1) * n + k];
                int vk_old = ws.vk[k];
                ws.vk[k] = (int)floor(ws.ck[k] + 0.5);
                ws.wk[k] = 1;
                if (k >= t && k < n) for (int j = 0; j < d; j++) { ws.common_part[j] -= vk_old * L[j * n + k]; ws.common_part[j] += ws.vk[k] * L[j * n + k]; }
            }
        } else {
            k++; if (k == n) break;
            ws.rk[k] = k;
            if (k >= last_nonzero) {
                last_nonzero = k; int vk_old = ws.vk[k]; ws.vk[k]++;
                if (k >= t && k < n) for (int j = 0; j < d; j++) { ws.common_part[j] -= vk_old * L[j * n + k]; ws.common_part[j] += ws.vk[k] * L[j * n + k]; }
            } else {
                int vk_old = ws.vk[k];
                if (ws.vk[k] > ws.ck[k]) ws.vk[k] -= ws.wk[k]; else ws.vk[k] += ws.wk[k];
                if (k >= t && k < n) for (int j = 0; j < d; j++) { ws.common_part[j] -= vk_old * L[j * n + k]; ws.common_part[j] += ws.vk[k] * L[j * n + k]; }
                ws.wk[k]++;
            }
        }
    }
    return nn;
}

// ==================== RELATIONS & COFACTORING ====================

int64_t rel2A(int d, mpz_ptr* Ai, int64_t reli, int bb) {
    int64_t BB = 1LL << bb, hB = 1LL << (bb - 1), A = 0;
    for (int j = 0; j < d - 2; j++) A += mpz_get_si(Ai[j]) * (((reli >> (bb * j)) & (BB - 1)) - hB);
    return A + (((reli >> (bb * (d - 2))) & (BB - 1)) - hB);
}

int64_t rel2B(int d, mpz_ptr* Bi, int64_t reli, int bb) {
    int64_t BB = 1LL << bb, hB = 1LL << (bb - 1), B = 0;
    for (int j = 0; j < d - 2; j++) B += mpz_get_si(Bi[j]) * (((reli >> (bb * j)) & (BB - 1)) - hB);
    return B - (((reli >> (bb * (d - 1))) & (BB - 1)) - hB);
}

inline int64_t gcd(int64_t a, int64_t b) { a = abs(a); b = abs(b); while (b) { int64_t t = b; b = a % b; a = t; } return a; }

void trial_divide_side(mpz_t N, const vector<int>& fb_p, const vector<int>& small_primes, string& str, stringstream& stream) {
    if (small_primes.empty() || fb_p.empty()) return;
    int p = small_primes[0], k = 0, max_small = 1000;
    while (p < fb_p.back()) {
        while (mpz_fdiv_ui(N, p) == 0) {
            mpz_divexact_ui(N, N, p);
            if (!str.empty() && str.back() != ':' && str.back() != ',') str += ",";
            stream.str(""); stream << hex << p; 
            str += stream.str();
        }
        if (p < max_small) {
            if ((size_t)(k + 1) < small_primes.size()) p = small_primes[++k];
            else p = max_small + 1;
            if (p > max_small) { k = 0; while ((size_t)k < fb_p.size() && fb_p[k] < max_small) k++; }
        } else {
            if ((size_t)(k + 1) < fb_p.size()) p = fb_p[++k]; else break;
        }
    }
}

bool cofactorize_side(mpz_t N, mpz_t S, const mpz_class& lpb, string& str, int BASE,
                      stack<mpz_ptr>QN, stack<int>& Q, int* algarr, mpz_t* pi,
                      mpz_t factor, mpz_t p1, mpz_t p2, mpz_t t)
{
    if (mpz_cmp_ui(N, 1) == 0) return true;
    if (mpz_probab_prime_p(N, 30)) {
        if (mpz_cmpabs(N, lpb.get_mpz_t()) > 0) return false;
        if (!str.empty() && str.back() != ':' && str.back() != ',') str += ",";
        char buf[32]; mpz_get_str(buf, BASE, N); str += buf; 
        return true;
    }
    while (!Q.empty()) Q.pop(); while (!QN.empty()) QN.pop();
    QN.push(N); Q.push(2); Q.push(1); Q.push(0); Q.push(3);
    while (!QN.empty()) {
        mpz_ptr Ncur = QN.top(); QN.pop();
        int l = Q.top(); Q.pop();
        int j = 0; bool factored = false;
        while (!factored) {
            int alg = Q.top(); Q.pop(); j++;
            if (alg == 0) factored = PollardPm1(Ncur, S, factor);
            else if (alg == 1) factored = EECM(Ncur, S, factor, 25921, 83521, 19, 9537, 2737);
            else if (alg == 2) factored = EECM(Ncur, S, factor, 1681, 707281, 3, 19642, 19803);
            if (!factored) { if (j >= l) return false; }
            else {
                mpz_set(p1, factor); mpz_divexact(p2, Ncur, factor);
                if (mpz_cmpabs(p1, p2) > 0) { mpz_set(t, p1); mpz_set(p1, p2); mpz_set(p2, t); }
                int lnext = l - j;
                for (int lt = 0; lt < lnext; lt++) algarr[lt] = Q.top(), Q.pop();
                for (int lt = 0; lt < lnext; lt++) Q.push(algarr[lnext - 1 - lt]);
                if (lnext) Q.push(lnext);
                
                // Process p1
                if (mpz_probab_prime_p(p1, 30)) {
                    if (mpz_cmpabs(p1, lpb.get_mpz_t()) > 0) return false;
                    if (!str.empty() && str.back() != ':' && str.back() != ',') str += ",";
                    char buf[32]; mpz_get_str(buf, BASE, p1); str += buf;
                } else if (lnext) { mpz_set(pi[0], p1); QN.push(pi[0]); }
                else return false;
                
                // Process p2
                if (mpz_probab_prime_p(p2, 30)) {
                    if (mpz_cmpabs(p2, lpb.get_mpz_t()) > 0) return false;
                    if (!str.empty() && str.back() != ':' && str.back() != ',') str += ",";
                    char buf[32]; mpz_get_str(buf, BASE, p2); str += buf;
                } else if (lnext) { mpz_set(pi[1], p2); QN.push(pi[1]); }
                else return false;
            }
        }
    }
    return true;
}

// ==================== ORIGINAL CORE FUNCTIONS (Pollard, EECM, GetlcmScalar) ====================
void GetlcmScalar(int B, mpz_t S, int* primes, int nump)
{
    mpz_t* tree = new mpz_t[nump];
    mpz_t pe, pe1;
    mpz_init(pe); mpz_init(pe1);

    int n = 0;
    int p = 2;
    while (p < B) {
        mpz_set_ui(pe, p);
        mpz_mul_ui(pe1, pe, p);
        while (mpz_cmp_ui(pe1, B) < 0) {
            mpz_set(pe, pe1);
            mpz_mul_ui(pe1, pe, p);
        }
        mpz_init(tree[n]);
        mpz_set(tree[n], pe);
        n++;
        p = primes[n];
    }
    mpz_clear(pe); mpz_clear(pe1);

    uint64_t treepos = n - 1;
    while (treepos > 0) {
        for (int i = 0; i <= treepos; i += 2) {
            if (i < treepos)
                mpz_lcm(tree[i/2], tree[i], tree[i + 1]);
            else
                mpz_set(tree[i/2], tree[i]);
        }
        for (int i = (treepos >> 1); i < treepos - 1; i++) mpz_set_ui(tree[i + 1], 1);
        treepos = treepos >> 1;
    }
    mpz_set(S, tree[0]);

    for (int i = 0; i < n; i++) mpz_clear(tree[i]);
    delete[] tree;
}

inline __int128 make_int128(uint64_t lo, uint64_t hi)
{
    __int128 N = hi;
    N = N << 64;
    N += lo;
    return N;
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

bool PollardPm1(mpz_t N, mpz_t S, mpz_t factor)
{
    int bitlen = mpz_sizeinbase(N, 2);
    if (bitlen < 64) {
        __int128 N128 = make_int128(mpz_getlimbn(N,0), mpz_getlimbn(N,1));
        __int128 factor128 = 1;
        PollardPm1_int128(N128, S, factor128, 0, 0);
        mp_limb_t* fl = mpz_limbs_modify(factor, 2);
        fl[0] = factor128 & MASK64;
        fl[1] = factor128 >> 64;
        return (factor128 > 1 && factor128 < N128);
    }
    return PollardPm1_mpz(N, S, factor);
}

bool PollardPm1_mpz(mpz_t N, mpz_t S, mpz_t factor)
{
    int L = mpz_sizeinbase(S, 2);
    mpz_t g; mpz_init_set_ui(g, 2);
    for (int i = 2; i <= L; i++) {
        mpz_mul(g, g, g);
        mpz_mod(g, g, N);
        if (mpz_tstbit(S, L - i) == 1) {
            mpz_mul_2exp(g, g, 1);
            if (mpz_cmpabs(g, N) >= 0) mpz_sub(g, g, N);
        }
    }
    mpz_sub_ui(g, g, 1);
    mpz_gcd(factor, N, g);
    bool result = mpz_cmpabs_ui(factor, 1) > 0 && mpz_cmpabs(factor, N) < 0;
    mpz_clear(g);
    return result;
}

bool PollardPm1_int128(__int128 N, mpz_t S, __int128 &factor, int64_t, int64_t)
{
    int L = mpz_sizeinbase(S, 2);
    __int128 g = 2;
    for (int i = 2; i <= L; i++) {
        g = g * g % N;
        if (mpz_tstbit(S, L - i) == 1) {
            g = g * 2;
            if (g >= N) g -= N;
        }
    }
    g -= 1;
    factor = gcd128(N, g);
    return (factor > 1 && factor < N);
}

bool EECM(mpz_t N, mpz_t S, mpz_t factor, int d, int a, int X0, int Y0, int Z0)
{
    int bitlen = mpz_sizeinbase(N, 2);
    if (bitlen < 64) {
        __int128 N128 = make_int128(mpz_getlimbn(N,0), mpz_getlimbn(N,1));
        __int128 factor128 = 1;
        EECM_int128(N128, S, factor128, d, a, X0, Y0, Z0, 0, 0);
        mp_limb_t* fl = mpz_limbs_modify(factor, 2);
        fl[0] = factor128 & MASK64;
        fl[1] = factor128 >> 64;
        return (factor128 > 1 && factor128 < N128);
    }
    return EECM_mpz(N, S, factor, d, a, X0, Y0, Z0);
}

bool EECM_mpz(mpz_t N, mpz_t S, mpz_t factor, int d, int a, int X0, int Y0, int Z0)
{
    mpz_t SX, SY, SZ, A, B, B2, B3, C, dC, B2mC, D, CaD, B2mCmD, E, EmD, F, AF, G, AG, aC, DmaC, H, Hx2, J;
    mpz_t X0aY0xB, X0aY0xB_mCmD, X, Y, Z, mulmod;
    mpz_init(SX); mpz_init_set_ui(SX, X0);
    mpz_init(SY); mpz_init_set_ui(SY, Y0);
    mpz_init(SZ); mpz_init_set_ui(SZ, Z0);
    mpz_init(A); mpz_init(B); mpz_init(B2); mpz_init(B3); mpz_init(C); mpz_init(dC); mpz_init(B2mC);
    mpz_init(D); mpz_init(CaD); mpz_init(B2mCmD); mpz_init(E); mpz_init(EmD); mpz_init(F); mpz_init(AF);
    mpz_init(G); mpz_init(AG); mpz_init(aC); mpz_init(DmaC); mpz_init(H); mpz_init(Hx2); mpz_init(J);
    mpz_init(X0aY0xB); mpz_init(X0aY0xB_mCmD); mpz_init(X); mpz_init(Y); mpz_init(Z); mpz_init(mulmod);

    int L = mpz_sizeinbase(S, 2);
    for(int i = 2; i <= L; i++) {
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
    mpz_gcd(factor, N, SX);

    mpz_clear(mulmod); mpz_clear(SZ); mpz_clear(SY); mpz_clear(SX);
    mpz_clear(A); mpz_clear(B); mpz_clear(B2); mpz_clear(B3); mpz_clear(C); mpz_clear(dC); mpz_clear(B2mC);
    mpz_clear(D); mpz_clear(CaD); mpz_clear(B2mCmD); mpz_clear(E); mpz_clear(EmD); mpz_clear(F); mpz_clear(AF);
    mpz_clear(G); mpz_clear(AG); mpz_clear(aC); mpz_clear(DmaC); mpz_clear(H); mpz_clear(Hx2); mpz_clear(J);
    mpz_clear(X0aY0xB); mpz_clear(X0aY0xB_mCmD); mpz_clear(X); mpz_clear(Y); mpz_clear(Z);

    return mpz_cmpabs_ui(factor, 1) > 0 && mpz_cmpabs(factor, N) < 0;
}

bool EECM_int128(__int128 N, mpz_t S, __int128 &factor, int d, int a, int X0, int Y0, int Z0, int64_t, int64_t)
{
    __int128 SX = X0, SY = Y0, SZ = Z0;
    __int128 A, B, B2, B3, C, dC, B2mC, D, CaD, B2mCmD, E, EmD, F, AF, G, AG, aC, DmaC, H, Hx2, J;
    __int128 X0aY0xB, X0aY0xB_mCmD, X, Y, Z, mulmod;

    int L = mpz_sizeinbase(S, 2);
    for(int i = 2; i <= L; i++) {
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
    factor = gcd128(N, SX);
    return (factor > 1 && factor < N);
}

// ==================== MAIN ====================
int main(int argc, char** argv)
{
    MASK64 = ((int128_t)1 << 64) - 1;
    if (argc != 17) {
        cerr << "Usage: ./slcsievehdx inputpoly fb d Amax Bmax N Bmin Bmax Rmin Rmax th0 th1 lpb bits mbb bb" << endl;
        return 1;
    }

    // 1. Initial configuration
    int d = atoi(argv[3]);
    mpz_class maxA(argv[4]), maxB(argv[5]), lpb(argv[13]);
    int N_units = atoi(argv[6]), Bmin = atoi(argv[7]), Bmax = atoi(argv[8]);
    int Rmin = atoi(argv[9]), Rmax = atoi(argv[10]);
    uint8_t th[2] = {(uint8_t)atoi(argv[11]), (uint8_t)atoi(argv[12])};
    int64_t cofmax = 1LL << atoi(argv[14]);
    int mbb = atoi(argv[15]), bb = atoi(argv[16]);

    // 2. Load Polynomials
    mpz_poly f0, f1;
    mpz_poly_init(f0, 10); mpz_poly_init(f1, 10);
    double skew = 1.0; int degf, degg;
    parse_polynomial(argv[1], f0, f1, skew, degf, degg);

    // 3. Load Factor Base
    SieveSide sides[2];
    load_factor_base(argv[2], sides, th);

    // 4. Prime Generation & Memory Allocation
    vector<int> small_primes;
    {
        int maxs = 1 << 21; vector<char> sieve(maxs + 1, 0);
        for (int i = 2; i * i <= maxs; i++) if (!sieve[i]) for (int j = i * i; j <= maxs; j += i) sieve[j] = 1;
        for (int i = 2; i <= maxs; i++) if (!sieve[i]) small_primes.push_back(i);
    }
    vector<KeyVal> M(1ull << mbb);
    uint64_t m[512];

    // 5. Preparation for Cofactorization
    gmp_randstate_t state; gmp_randinit_default(state); gmp_randseed_ui(state, 123ul);
    vector<mpz_class> Ai(d - 2), Bi(d - 2);
    vector<mpz_ptr> Ai_ptr(d - 2);
    vector<mpz_ptr> Bi_ptr(d - 2);

    for (int i = 0; i < d - 2; i++) { 
        Ai_ptr[i] = Ai[i].get_mpz_t(); 
        Bi_ptr[i] = Bi[i].get_mpz_t(); 
    }

    mpz_poly i1; mpz_poly_init(i1, 3);
    mpz_t N0, N1, S, factor, p1, p2, t, g1, A, B, pi[8];
    mpz_init(N0); mpz_init(N1); mpz_init(S); mpz_init(factor); mpz_init(p1); mpz_init(p2);
    mpz_init(t); mpz_init(g1); mpz_init(A); mpz_init(B);
    for (int i = 0; i < 8; i++) mpz_init(pi[i]);
    GetlcmScalar(cofmax, S, small_primes.data(), small_primes.size());


    // 6. Main Processing Loop
    for (int nn = 0; nn < N_units; nn++) {
        for (int i = 0; i < d - 2; i++) {
            mpz_urandomm(Ai_ptr[i], state, maxA.get_mpz_t());
            mpz_urandomm(Bi_ptr[i], state, maxB.get_mpz_t());
            char str1[64], str2[64];
            mpz_get_str(str1, 10, Ai_ptr[i]);
            mpz_get_str(str2, 10, Bi_ptr[i]);
            cout << "# " << str1 << "*x + " << str2 << endl;
        }

        SieveWorkspace ws(d);
        vector<int64_t> rel0, rel1, common;

        // Side 0 sieve
        cout << "# Starting sieve on side 0..." << endl;
        std::clock_t start = clock();
        slcsieve(d, Ai_ptr.data(), Bi_ptr.data(), Bmin, Bmax, Rmin, Rmax, sides[0], degf, M.data(), mbb, bb, ws);
        double timetaken = (clock() - start) / (double)CLOCKS_PER_SEC;
        cout << "# Finished! Time taken: " << timetaken << "s" << endl;
        cout << "# Size of lattice point list is " << ws.m_idx << "." << endl;

        cout << "# Sorting bucket sieve data..." << endl;
        start = clock();
        collect_candidates(sides[0].threshold, M.data(), ws.m_idx, d, Ai_ptr.data(), Bi_ptr.data(), bb, rel0);
        timetaken = (clock() - start) / (double)CLOCKS_PER_SEC;
        cout << "# Finished! Time taken: " << timetaken << "s" << endl;
        cout << "# " << rel0.size() << " candidates on side 0." << endl;

        // Side 1 sieve
        cout << "# Starting sieve on side 1..." << endl;
        start = clock();
        slcsieve(d, Ai_ptr.data(), Bi_ptr.data(), Bmin, Bmax, Rmin, Rmax, sides[1], degg, M.data(), mbb, bb, ws);
        timetaken = (clock() - start) / (double)CLOCKS_PER_SEC;
        cout << "# Finished! Time taken: " << timetaken << "s" << endl;
        cout << "# Size of lattice point list is " << ws.m_idx << "." << endl;

        cout << "# Sorting bucket sieve data..." << endl;
        start = clock();
        collect_candidates(sides[1].threshold, M.data(), ws.m_idx, d, Ai_ptr.data(), Bi_ptr.data(), bb, rel1);
        timetaken = (clock() - start) / (double)CLOCKS_PER_SEC;
        cout << "# Finished! Time taken: " << timetaken << "s" << endl;
        cout << "# " << rel1.size() << " candidates on side 1." << endl;

        // Intersection
        cout << "# Sorting candidate relation list..." << endl;
        start = clock();
        sort(rel0.begin(), rel0.end());
        sort(rel1.begin(), rel1.end());
        set_intersection(rel0.begin(), rel0.end(), rel1.begin(), rel1.end(), back_inserter(common));
        timetaken = (clock() - start) / (double)CLOCKS_PER_SEC;
        cout << "# Finished! Time taken: " << timetaken << "s" << endl;
        cout << "# " << common.size() << " potential relations found." << endl;

        // Cofactorization
        cout << "# Starting cofactorization..." << endl;
        start = clock();
        int R = 0;
        int samples = 0;
        stringstream stream;
        for (int64_t id : common) {
            mpz_set_si(A, rel2A(d, Ai_ptr.data(), id, bb));
            mpz_set_si(B, rel2B(d, Bi_ptr.data(), id, bb));
            mpz_gcd(g1, A, B); mpz_divexact(A, A, g1); mpz_divexact(B, B, g1);
            mpz_poly_setcoeff(i1, 1, A); mpz_poly_setcoeff(i1, 0, B);
            mpz_poly_resultant(N0, f0, i1); mpz_poly_resultant(N1, f1, i1);
            mpz_abs(N0, N0); mpz_abs(N1, N1);

            string str = mpz_get_str(NULL, 10, A) + (string)"," + mpz_get_str(NULL, 10, B) + ":";
            stack<mpz_ptr> QN; stack<int> Q; int algarr[3];

            trial_divide_side(N0, sides[0].p, small_primes, str, stream);
            if (cofactorize_side(N0, S, lpb, str, 16, QN, Q, algarr, pi, factor, p1, p2, t)) {
                str += ":";
                trial_divide_side(N1, sides[1].p, small_primes, str, stream);
                if (cofactorize_side(N1, S, lpb, str, 16, QN, Q, algarr, pi, factor, p1, p2, t)) {
                    cout << str << endl;
                    R++;
                }
            }
            if (samples < 10) {
                cout << str << endl;
                samples++;
            }
        }
        timetaken = (clock() - start) / (double)CLOCKS_PER_SEC;
        cout << "# Finished! Cofactorization took " << timetaken << "s" << endl;
        cout << "# " << R << " actual relations found." << endl;
    }

    // 7. Cleanup
    mpz_poly_clear(f0); mpz_poly_clear(f1); mpz_poly_clear(i1);
    mpz_clear(N0); mpz_clear(N1); mpz_clear(S); mpz_clear(factor);
    for (int i = 0; i < 8; i++) mpz_clear(pi[i]);
    gmp_randclear(state);
    return 0;
}

