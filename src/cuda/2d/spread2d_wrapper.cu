#include <cassert>
#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cufinufft/memtransfer.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>

#include "spreadinterp2d.cuh"

using namespace cufinufft::common;
using namespace cufinufft::memtransfer;

namespace cufinufft {
namespace spreadinterp {

template <typename T>
int cufinufft_spread2d(int nf1, int nf2, cuda_complex<T> *d_fw, int M, T *d_kx, T *d_ky, cuda_complex<T> *d_c,
                       cufinufft_plan_t<T> *d_plan)
/*
    This c function is written for only doing 2D spreading. See
    test/spread2d_test.cu for usage.

    Melody Shih 07/25/19
    not allocate,transfer and free memories on gpu. Shih 09/24/20
*/
{
    d_plan->kx = d_kx;
    d_plan->ky = d_ky;
    d_plan->c = d_c;
    d_plan->fw = d_fw;

    int ier;
    d_plan->nf1 = nf1;
    d_plan->nf2 = nf2;
    d_plan->M = M;
    d_plan->maxbatchsize = 1;

    ier = ALLOCGPUMEM2D_PLAN(d_plan);
    ier = ALLOCGPUMEM2D_NUPTS(d_plan);

    if (d_plan->opts.gpu_method == 1) {
        ier = CUSPREAD2D_NUPTSDRIVEN_PROP(nf1, nf2, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread2d_nuptsdriven_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }

    if (d_plan->opts.gpu_method == 2) {
        ier = CUSPREAD2D_SUBPROB_PROP(nf1, nf2, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread2d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }

    ier = cuspread2d<T>(d_plan, 1);

    freegpumemory2d<T>(d_plan);

    return ier;
}

template <typename T>
int cuspread2d(cufinufft_plan_t<T> *d_plan, int blksize)
/*
    A wrapper for different spreading methods.

    Methods available:
    (1) Non-uniform points driven
    (2) Subproblem

    Melody Shih 07/25/19
*/
{
    int nf1 = d_plan->nf1;
    int nf2 = d_plan->nf2;
    int M = d_plan->M;

    int ier;
    switch (d_plan->opts.gpu_method) {
    case 1: {
        ier = cuspread2d_nuptsdriven<T>(nf1, nf2, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread2d_gpu_nuptsdriven" << std::endl;
            return 1;
        }
    } break;
    case 2: {
        ier = cuspread2d_subprob<T>(nf1, nf2, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread2d_gpu_subprob" << std::endl;
            return 1;
        }
    } break;
    default:
        std::cout << "error: incorrect method, should be 1,2,3" << std::endl;
        return 2;
    }

    return ier;
}

template <typename T>
int cuspread2d_nuptsdriven_prop(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan) {
    if (d_plan->opts.gpu_sort) {

        int bin_size_x = d_plan->opts.gpu_binsizex;
        int bin_size_y = d_plan->opts.gpu_binsizey;
        if (bin_size_x < 0 || bin_size_y < 0) {
            std::cout << "error: invalid binsize (binsizex, binsizey) = (";
            std::cout << bin_size_x << "," << bin_size_y << ")" << std::endl;
            return 1;
        }

        int numbins[2];
        numbins[0] = ceil((T)nf1 / bin_size_x);
        numbins[1] = ceil((T)nf2 / bin_size_y);

        T *d_kx = d_plan->kx;
        T *d_ky = d_plan->ky;

        int *d_binsize = d_plan->binsize;
        int *d_binstartpts = d_plan->binstartpts;
        int *d_sortidx = d_plan->sortidx;
        int *d_idxnupts = d_plan->idxnupts;

        int pirange = d_plan->spopts.pirange;

        checkCudaErrors(cudaMemset(d_binsize, 0, numbins[0] * numbins[1] * sizeof(int)));
        calc_bin_size_noghost_2d<<<(M + 1024 - 1) / 1024, 1024>>>(
            M, nf1, nf2, bin_size_x, bin_size_y, numbins[0], numbins[1], d_binsize, d_kx, d_ky, d_sortidx, pirange);
        int n = numbins[0] * numbins[1];
        thrust::device_ptr<int> d_ptr(d_binsize);
        thrust::device_ptr<int> d_result(d_binstartpts);
        thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);

        calc_inverse_of_global_sort_index_2d<<<(M + 1024 - 1) / 1024, 1024>>>(
            M, bin_size_x, bin_size_y, numbins[0], numbins[1], d_binstartpts, d_sortidx, d_kx, d_ky, d_idxnupts,
            pirange, nf1, nf2);
    } else {
        int *d_idxnupts = d_plan->idxnupts;

        trivial_global_sort_index_2d<<<(M + 1024 - 1) / 1024, 1024>>>(M, d_idxnupts);
    }

    return 0;
}

template <typename T>
int cuspread2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
    dim3 threadsPerBlock;
    dim3 blocks;

    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    int pirange = d_plan->spopts.pirange;
    int *d_idxnupts = d_plan->idxnupts;
    T es_c = d_plan->spopts.ES_c;
    T es_beta = d_plan->spopts.ES_beta;
    T sigma = d_plan->spopts.upsampfac;

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    cuda_complex<T> *d_c = d_plan->c;
    cuda_complex<T> *d_fw = d_plan->fw;

    threadsPerBlock.x = 16;
    threadsPerBlock.y = 1;
    blocks.x = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocks.y = 1;
    if (d_plan->opts.gpu_kerevalmeth) {
        for (int t = 0; t < blksize; t++) {
            spread_2d_nupts_driven<T, 1><<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M,
                                                                      ns, nf1, nf2, es_c, es_beta, sigma, d_idxnupts,
                                                                      pirange);
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            spread_2d_nupts_driven<T, 0><<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M,
                                                                      ns, nf1, nf2, es_c, es_beta, sigma, d_idxnupts,
                                                                      pirange);
        }
    }

    return 0;
}

template <typename T>
int cuspread2d_subprob_prop(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan)
/*
    This function determines the properties for spreading that are independent
    of the strength of the nodes,  only relates to the locations of the nodes,
    which only needs to be done once.
*/
{
    int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    if (bin_size_x < 0 || bin_size_y < 0) {
        std::cout << "error: invalid binsize (binsizex, binsizey) = (";
        std::cout << bin_size_x << "," << bin_size_y << ")" << std::endl;
        return 1;
    }
    int numbins[2];
    numbins[0] = ceil((T)nf1 / bin_size_x);
    numbins[1] = ceil((T)nf2 / bin_size_y);

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;

    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx = d_plan->sortidx;
    int *d_numsubprob = d_plan->numsubprob;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_idxnupts = d_plan->idxnupts;

    int *d_subprob_to_bin = NULL;

    int pirange = d_plan->spopts.pirange;

    checkCudaErrors(cudaMemset(d_binsize, 0, numbins[0] * numbins[1] * sizeof(int)));
    calc_bin_size_noghost_2d<<<(M + 1024 - 1) / 1024, 1024>>>(M, nf1, nf2, bin_size_x, bin_size_y, numbins[0],
                                                              numbins[1], d_binsize, d_kx, d_ky, d_sortidx, pirange);

    int n = numbins[0] * numbins[1];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts);
    thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);

    calc_inverse_of_global_sort_index_2d<<<(M + 1024 - 1) / 1024, 1024>>>(M, bin_size_x, bin_size_y, numbins[0],
                                                                          numbins[1], d_binstartpts, d_sortidx, d_kx,
                                                                          d_ky, d_idxnupts, pirange, nf1, nf2);

    calc_subprob_2d<<<(M + 1024 - 1) / 1024, 1024>>>(d_binsize, d_numsubprob, maxsubprobsize, numbins[0] * numbins[1]);

    d_ptr = thrust::device_pointer_cast(d_numsubprob);
    d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
    checkCudaErrors(cudaMemset(d_subprobstartpts, 0, sizeof(int)));

    int totalnumsubprob;
    checkCudaErrors(cudaMemcpy(&totalnumsubprob, &d_subprobstartpts[n], sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMalloc(&d_subprob_to_bin, totalnumsubprob * sizeof(int)));
    map_b_into_subprob_2d<<<(numbins[0] * numbins[1] + 1024 - 1) / 1024, 1024>>>(d_subprob_to_bin, d_subprobstartpts,
                                                                                 d_numsubprob, numbins[0] * numbins[1]);
    assert(d_subprob_to_bin != NULL);
    if (d_plan->subprob_to_bin != NULL)
        cudaFree(d_plan->subprob_to_bin);
    d_plan->subprob_to_bin = d_subprob_to_bin;
    assert(d_plan->subprob_to_bin != NULL);
    d_plan->totalnumsubprob = totalnumsubprob;

    return 0;
}

template <typename T>
int cuspread2d_subprob(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    T es_c = d_plan->spopts.ES_c;
    T es_beta = d_plan->spopts.ES_beta;
    int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

    // assume that bin_size_x > ns/2;
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    int numbins[2];
    numbins[0] = ceil((T)nf1 / bin_size_x);
    numbins[1] = ceil((T)nf2 / bin_size_y);

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    cuda_complex<T> *d_c = d_plan->c;
    cuda_complex<T> *d_fw = d_plan->fw;

    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_numsubprob = d_plan->numsubprob;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_idxnupts = d_plan->idxnupts;

    int totalnumsubprob = d_plan->totalnumsubprob;
    int *d_subprob_to_bin = d_plan->subprob_to_bin;

    int pirange = d_plan->spopts.pirange;

    T sigma = d_plan->opts.upsampfac;

    size_t sharedplanorysize =
        (bin_size_x + 2 * (int)ceil(ns / 2.0)) * (bin_size_y + 2 * (int)ceil(ns / 2.0)) * sizeof(cuda_complex<T>);
    if (sharedplanorysize > 49152) {
        std::cout << "error: not enough shared memory" << std::endl;
        return 1;
    }

    if (d_plan->opts.gpu_kerevalmeth) {
        for (int t = 0; t < blksize; t++) {
            spread_2d_subprob<T, 1><<<totalnumsubprob, 256, sharedplanorysize>>>(
                d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, es_c, es_beta, sigma, d_binstartpts,
                d_binsize, bin_size_x, bin_size_y, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize,
                numbins[0], numbins[1], d_idxnupts, pirange);
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            spread_2d_subprob<T, 0><<<totalnumsubprob, 256, sharedplanorysize>>>(
                d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, es_c, es_beta, sigma, d_binstartpts,
                d_binsize, bin_size_x, bin_size_y, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize,
                numbins[0], numbins[1], d_idxnupts, pirange);
        }
    }

    return 0;
}
template int cuspread2d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuspread2d<double>(cufinufft_plan_t<double> *d_plan, int blksize);
template int cuspread2d_subprob_prop<float>(int nf1, int nf2, int M, cufinufft_plan_t<float> *d_plan);
template int cuspread2d_subprob_prop<double>(int nf1, int nf2, int M, cufinufft_plan_t<double> *d_plan);
template int cuspread2d_nuptsdriven_prop<float>(int nf1, int nf2, int M, cufinufft_plan_t<float> *d_plan);
template int cuspread2d_nuptsdriven_prop<double>(int nf1, int nf2, int M, cufinufft_plan_t<double> *d_plan);

} // namespace spreadinterp
} // namespace cufinufft
