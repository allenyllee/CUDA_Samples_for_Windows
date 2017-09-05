/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Parallel reduction

    This sample shows how to perform a reduction operation on an array of values
    to produce a single value in a single kernel (as opposed to two or more
    kernel calls as shown in the "reduction" CUDA Sample).  Single-pass
    reduction requires global atomic instructions (Compute Capability 1.1 or
    later) and the __threadfence() intrinsic (CUDA 2.2 or later).

    Reductions are a very common computation in parallel algorithms.  Any time
    an array of values needs to be reduced to a single value using a binary
    associative operator, a reduction can be used.  Example applications include
    statistics computations such as mean and standard deviation, and image
    processing applications such as finding the total luminance of an
    image.

    This code performs sum reductions, but any associative operator such as
    min() or max() could also be used.

    It assumes the input size is a power of 2.

    COMMAND LINE ARGUMENTS

    "--n=<N>":         Specify the number of elements to reduce (default 33554432)
    "--threads=<N>":   Specify the number of threads per block (default 128)
    "--maxblocks=<N>": Specify the maximum number of thread blocks to launch (kernel 6 only, default 64)
    "--cpufinal":      Read back the per-block results and do final sum of block sums on CPU (default false)
    "--cputhresh=<N>": The threshold of number of blocks sums below which to perform a CPU final reduction (default 1)
    "--multipass":     Use a multipass reduction instead of a single-pass reduction

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>

const char *sSDKsample = "reductionMultiBlockCG";

#include <device_functions.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n/2 threads
    - only works for power-of-2 arrays

    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    See the CUDA SDK "reduction" sample for more information.
*/


__device__ __host__ bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

__device__ void reduceBlock(volatile float *sdata, float mySum, const unsigned int tid, cg::thread_block &cta, cg::thread_block_tile<32> &tile32)
{
    sdata[tid] = mySum;
    cg::sync(tile32);

    const int VEC = 32;
    float beta  = mySum;
    float temp;

    for (int i = VEC/2; i > 0; i>>=1)
    {
        if (tile32.thread_rank() < i)
        {
            temp      = sdata[tid+i];
            beta     += temp;
            sdata[tid]  = beta;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (cta.thread_rank() == 0) 
    {
        beta  = 0;
        for (int i = 0; i < blockDim.x; i += VEC) 
        {
            beta  += sdata[i];
        }
        sdata[0] = beta;
    }
    cg::sync(cta);
}

__device__ void reduceBlocks(const float *g_idata, float *g_odata, unsigned int n, bool nIsPow2, cg::thread_block &cta, cg::thread_block_tile<32> &tile32)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;
    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockDim.x < n)
            mySum += g_idata[i+blockDim.x];

        i += gridSize;
    }

    // do reduction in shared mem
    reduceBlock(sdata, mySum, tid, cta, tile32);

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduceMultiPass(const float *g_idata, float *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    bool nIsPow2 = isPow2(n);

    reduceBlocks(g_idata, g_odata, n, nIsPow2, cta, tile32);
}

// This reduction kernel reduces an arbitrary size array in a single kernel invocation
//
// For more details on the reduction algorithm (notably the multi-pass approach), see
// the "reduction" sample in the CUDA SDK.
extern "C" __global__ void reduceSinglePassMultiBlockCG(const float *g_idata, float *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    bool nIsPow2 = isPow2(n);
    //
    // PHASE 1: Process all inputs assigned to this block
    //
    reduceBlocks(g_idata, g_odata, n, nIsPow2, cta, tile32);

    //
    // PHASE 2: First block will process all partial sums 
    //
    cg::grid_group grid = cg::this_grid();
    cg::sync(grid);

    if (blockIdx.x == 0)
    {
        const unsigned int tid = threadIdx.x;
        int i = tid;
        float mySum = 0;
        extern float __shared__ smem[];

        while (i < gridDim.x)
        {
            mySum += g_odata[i];
            i += blockDim.x;
        }

        reduceBlock(smem, mySum, tid, cta, tile32);

        if (tid==0)
        {
            g_odata[0] = smem[0];
         }
    }
}


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    // choose which of the optimized versions of reduction to launch
    reduceMultiPass<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
}

void call_reduceSinglePassMultiBlockCG(int size, int threads, int numBlocks, float *d_idata, float *d_odata)
{

    int smemSize = threads * sizeof(float);
    void **kernelArgs = NULL;

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);

    kernelArgs = (void**)malloc(3 * sizeof(*kernelArgs));
    kernelArgs[0] = malloc(sizeof(d_idata));
    memcpy(kernelArgs[0], &d_idata, sizeof(d_idata));
    kernelArgs[1] = malloc(sizeof(d_odata));
    memcpy(kernelArgs[1], &d_odata, sizeof(d_odata));
    kernelArgs[2] = malloc(sizeof(size));
    memcpy(kernelArgs[2], &size, sizeof(size));

    cudaLaunchCooperativeKernel((void*)reduceSinglePassMultiBlockCG, dimGrid, dimBlock, kernelArgs, smemSize, NULL);

    free(kernelArgs[0]);
    free(kernelArgs[1]);
    free(kernelArgs[2]);
    free(kernelArgs);
}


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

    printf("%s Starting...\n\n", sSDKsample);

    dev = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    if (deviceProp.major < 6)
    {
        printf("\nreductionMultiBlockCG requires GPU with compute capability 6.0, i.e. Pascal or higher\nWaiving the run\n");
        exit(EXIT_WAIVED);
    }

    if (!deviceProp.cooperativeLaunch)
    {
        printf("\nSelected GPU does not support Cooperative Kernel Launch, Waiving the run\n");
        exit(EXIT_WAIVED);
    }

    bool bTestResult = false;

    bTestResult = runTest(argc, argv);

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template<class T>
T reduceCPU(T *data, int size)
{
    T sum = data[0];
    T c = (T)0.0;

    for (int i = 1; i < size; i++)
    {
        T y = data[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the reduction
// We set threads / block to the minimum of maxThreads and n/2.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    if (n == 1)
    {
        threads = 1;
        blocks = 1;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2(n / 2) : maxThreads;
        blocks = max(1, n / (threads * 2));
    }

    blocks = min(maxBlocks, blocks);
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
float benchmarkReduce(int  n,
                      int  numThreads,
                      int  numBlocks,
                      int  maxThreads,
                      int  maxBlocks,
                      int  testIterations,
                      bool multiPass,
                      bool cpuFinalReduction,
                      int  cpuFinalThreshold,
                      StopWatchInterface *timer,
                      float *h_odata,
                      float *d_idata,
                      float *d_odata)
{
    float gpu_result = 0;
    bool bNeedReadback = true;
    cudaError_t error;

    printf("\nLaunching %s kernel\n", multiPass ? "MultiPass" : "SinglePass Multi Block Cooperative Groups");

    for (int i = 0; i < testIterations; ++i)
    {
        gpu_result = 0;
        cudaDeviceSynchronize();
        sdkStartTimer(&timer);

        if (multiPass)
        {
            // execute the kernel
            reduce(n, numThreads, numBlocks, d_idata, d_odata);

            // check if kernel execution generated an error
            getLastCudaError("Kernel execution failed");

            if (cpuFinalReduction)
            {
                // sum partial sums from each block on CPU
                // copy result from device to host
                error = cudaMemcpy(h_odata, d_odata, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);
                checkCudaErrors(error);

                for (int i=0; i<numBlocks; i++)
                {
                    gpu_result += h_odata[i];
                }

                bNeedReadback = false;
            }
            else
            {
                // sum partial block sums on GPU
                int s=numBlocks;

                while (s > cpuFinalThreshold)
                {
                    int threads = 0, blocks = 0;
                    getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

                    reduce(s, threads, blocks, d_odata, d_odata);

                    s = s / (threads*2);
                }

                if (s > 1)
                {
                    // copy result from device to host
                    error = cudaMemcpy(h_odata, d_odata, s * sizeof(float), cudaMemcpyDeviceToHost);
                    checkCudaErrors(error);

                    for (int i=0; i < s; i++)
                    {
                        gpu_result += h_odata[i];
                    }

                    bNeedReadback = false;
                }
            }
        }
        else
        {
            getLastCudaError("Kernel execution failed");

            // execute the kernel
            call_reduceSinglePassMultiBlockCG(n, numThreads, numBlocks, d_idata, d_odata);

            // check if kernel execution generated an error
            getLastCudaError("Kernel execution failed");
        }

        cudaDeviceSynchronize();
        sdkStopTimer(&timer);
    }

    if (bNeedReadback)
    {
        // copy final sum from device to host
        error = cudaMemcpy(&gpu_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaErrors(error);
    }

    return gpu_result;
}

////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction test.
////////////////////////////////////////////////////////////////////////////////
bool
runTest(int argc, char **argv)
{
    int size = 1<<25;    // number of elements to reduce
    int maxThreads = 128;  // number of threads per block
    int maxBlocks = 64;
    bool cpuFinalReduction = false;
    int cpuFinalThreshold = 1;
    bool multipass = false;
    bool bTestResult = false;

    if (checkCmdLineFlag(argc, (const char **) argv, "n"))
    {
        size       = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "threads"))
    {
        maxThreads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "maxblocks"))
    {
        maxBlocks  = getCmdLineArgumentInt(argc, (const char **)argv, "maxblocks");
    }

    printf("%d elements\n", size);
    printf("%d threads (max)\n", maxThreads);

    cpuFinalReduction = checkCmdLineFlag(argc, (const char **) argv, "cpufinal");
    multipass         = checkCmdLineFlag(argc, (const char **) argv, "multipass");

    if (checkCmdLineFlag(argc, (const char **) argv, "cputhresh"))
    {
        cpuFinalThreshold = getCmdLineArgumentInt(argc, (const char **) argv, "cputhresh");
    }

    // create random input data on CPU
    unsigned int bytes = size * sizeof(float);

    float *h_idata = (float *) malloc(bytes);

    for (int i=0; i<size; i++)
    {
        // Keep the numbers small so we don't get truncation error in the sum
        h_idata[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }

    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

    cudaDeviceProp prop;
    int numBlocksPerSm = 0;
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassMultiBlockCG, numThreads, numThreads*sizeof(float)));
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    int numSms = prop.multiProcessorCount;

    if (numBlocks > numBlocksPerSm * numSms)
    {
        numBlocks = numBlocksPerSm * numSms;
    }

    if (numBlocks == 1)
    {
        cpuFinalThreshold = 1;
    }

    // allocate mem for the result on host side
    float *h_odata = (float *) malloc(numBlocks*sizeof(float));

    printf("%d blocks\n", numBlocks);

    // allocate device memory and data
    float *d_idata = NULL;
    float *d_odata = NULL;

    checkCudaErrors(cudaMalloc((void **) &d_idata, bytes));
    checkCudaErrors(cudaMalloc((void **) &d_odata, numBlocks*sizeof(float)));

        // copy data directly to device memory
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks*sizeof(float), cudaMemcpyHostToDevice));

    int testIterations = 100;

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);

    float gpu_result = 0;

    gpu_result = benchmarkReduce(size, numThreads, numBlocks, maxThreads, maxBlocks,
                                     testIterations, multipass, cpuFinalReduction,
                                     cpuFinalThreshold, timer, h_odata, d_idata, d_odata);

    float reduceTime = sdkGetAverageTimerValue(&timer);
    printf("Average time: %f ms\n", reduceTime);
    printf("Bandwidth:    %f GB/s\n\n", (size * sizeof(int)) / (reduceTime * 1.0e6));

    // compute reference solution
    float cpu_result = reduceCPU<float>(h_idata, size);

    printf("GPU result = %0.12f\n", gpu_result);
    printf("CPU result = %0.12f\n", cpu_result);

    double threshold = 1e-8 * size;
    double diff = abs((double)gpu_result - (double)cpu_result);
    bTestResult = (diff < threshold);

    // cleanup
    sdkDeleteTimer(&timer);

    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    return bTestResult;
}

