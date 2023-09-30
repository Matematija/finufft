#ifndef FINUFFT_UTILS_H
#define FINUFFT_UTILS_H

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <complex>
#include <cstdint>

#include <cuComplex.h>
#include <cufinufft/types.h>
#include <cufinufft/defs.h>

#include <sys/time.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 || defined(__clang__)
#else
__inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN
        // (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

namespace cufinufft {
namespace utils {

// jfm timer class
class CNTime {
  public:
    void start();
    double restart();
    double elapsedsec();

  private:
    struct timeval initial;
};

// ahb math helpers
CUFINUFFT_BIGINT next235beven(CUFINUFFT_BIGINT n, CUFINUFFT_BIGINT b);

template <typename T>
T infnorm(int n, std::complex<T> *a) {
    T nrm = 0.0;
    for (int m = 0; m < n; ++m) {
        T aa = real(conj(a[m]) * a[m]);
        if (aa > nrm)
            nrm = aa;
    }
    return sqrt(nrm);
}

template <typename T>
void arrayrange(CUFINUFFT_BIGINT n, T* a, T *lo, T *hi)
// With a a length-n array, writes out min(a) to lo and max(a) to hi,
// so that all a values lie in [lo,hi].
// If n==0, lo and hi are not finite.
// MM: Does this need to be a kernel ?! I guess we can use thrust::minmax_element.
{
  *lo = INFINITY; *hi = -INFINITY; // Where is INFINITY?
  for (CUFINUFFT_BIGINT m=0; m<n; ++m) {
    if (a[m]<*lo) *lo = a[m];
    if (a[m]>*hi) *hi = a[m];
  }
}

template <typename T>
void arraywidcen(CUFINUFFT_BIGINT n, T* a, T *w, T *c)
// Writes out w = half-width and c = center of an interval enclosing all a[n]'s
// Only chooses a nonzero center if this increases w by less than fraction
// ARRAYWIDCEN_GROWFRAC defined in defs.h.
// This prevents rephasings which don't grow nf by much. 6/8/17
// If n==0, w and c are not finite.
{
  T lo,hi;
  arrayrange(n,a,&lo,&hi);
  *w = (hi-lo)/2;
  *c = (hi+lo)/2;
  if (std::abs(*c)<ARRAYWIDCEN_GROWFRAC*(*w)) {
    *w += std::abs(*c);
    *c = 0.0;
  }
}

} // namespace utils
} // namespace cufinufft

#endif
