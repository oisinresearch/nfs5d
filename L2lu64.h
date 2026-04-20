#ifndef L2lu_h
#define L2lu_h

#include <vector>
#include <algorithm>
#include <cstdint>

struct L2Workspace {
    std::vector<__int128> b;
    std::vector<__int128> G;
    std::vector<double>   rr;
    std::vector<double>   uu;

    L2Workspace(int max_d, int max_n)
        : b(max_d * max_n),
          G(max_n * max_n),
          rr(max_n * max_n),
          uu(max_n * max_n) {}

    void reset(int n) {
        std::fill(G.begin(),  G.begin()  + n*n, (__int128)0);
        std::fill(rr.begin(), rr.begin() + n*n, 0.0);
        std::fill(uu.begin(), uu.begin() + n*n, 0.0);
    }
};

void int64L2(int64_t* b, int d, int n, L2Workspace& ws);
void printbasis(int64_t* b, int d, int n, int pr, int w);
void copysquarefloatarray(int64_t* src, int64_t* dest, int d);
void luinv(int64_t* b, int64_t* M, int d);
int64_t fadlev4d(int64_t* A, int64_t *M);

#endif /* L2lu_h */
