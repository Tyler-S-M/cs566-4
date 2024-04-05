#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define BLOCKS 1
#define THREADS 1024

__device__ int vector[THREADS * BLOCKS];
__device__ int last_run;

__global__ void fill(){

    for (int i = 0; i < 1024; i++)
        vector[i] = i * 2;

}

__global__ void read_last_run(float time){

    printf("Last Run Result: %d and took: %f milliseconds\n", last_run, time);
    last_run = -1;

}

__global__ void parallel_redution_warp(){

    __shared__ int sum;

    //get our idex (assume 1d grids only)
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int item = vector[idx];
    int warp_lane = idx - ((int)(threadIdx.x / 32) * 32);

    //use warp shuffle to add items
    for (int offset = 16; offset > 0; offset /= 2)
        item += __shfl_down_sync(0xffffffff, item, offset);

    //sync
    __syncthreads();
    
    //have each warp store their value
    if (warp_lane == 0){
        atomicAdd(&sum, item);
    }

    if (blockIdx.x == 0 && warp_lane == 0)
        last_run = sum;
    
    return;
}

__global__ void parallel_redution_block(){

    __shared__ volatile int shared_vector[THREADS];

    //get our idex (assume 1d grids only)
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);

    //sync
    __syncthreads();

    //load one element into shared memory as shown in slides
    shared_vector[idx] = vector[idx];

    //sync
    __syncthreads();

    //do reduction as shown in slides
    __syncwarp();
    for (int i = 1; i < blockDim.x; i *= 2){

        //more volta-needed sync
        __syncthreads();

        if (!(idx % (2*i)))
            shared_vector[idx] += shared_vector[idx + i];

        //more volta-needed sync
        __syncthreads();
    }

    //sync
    __syncthreads();

    //write
    if (idx == 0)
        last_run = shared_vector[0];

    return;
}

__global__ void parallel_redution_block_branchless(){

    __shared__ volatile int shared_vector[THREADS];

    //get our idex (assume 1d grids only)
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);

    //sync
    __syncthreads();

    //load one element into shared memory as shown in slides
    shared_vector[idx] = vector[idx];

    //sync
    __syncthreads();

    //do reduction as shown in slides
    for (int i = 1; i < blockDim.x; i *= 2){
        int index = threadIdx.x * i * 2;

        //more volta-needed sync
        __syncthreads();

        if (index < blockDim.x)
            shared_vector[index] += shared_vector[index + i];

        //more volta-needed sync
        __syncthreads();
    }

    //sync
    __syncthreads();

    //write
    if (idx == 0)
        last_run = shared_vector[0];

    return;
}

__global__ void parallel_redution_block_interleaved(){
    
    __shared__ volatile int shared_vector[THREADS];

    //get our idex (assume 1d grids only)
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);

    //sync
    __syncthreads();

    //load one element into shared memory as shown in slides
    shared_vector[idx] = vector[idx];

    //sync
    __syncthreads();

    //do reduction as shown in slides
    for (int i = blockDim.x/2; i > 0; i >>= 1){

        //more volta-needed sync
        __syncthreads();

        if (idx < i)
            shared_vector[idx] += shared_vector[idx + i];

        //more volta-needed sync
        __syncthreads();
    }

    //sync
    __syncthreads();

    //write
    if (idx == 0)
        last_run = shared_vector[0];

    return;
}



int main(){

    //timing stuff
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Warp Reduction
    /*******************************************************************/
    //fill vector with random values
    fill<<<1,1>>>();
    cudaDeviceSynchronize();

    //now reduce with warp
    cudaEventRecord(start);
    parallel_redution_warp<<<1, 1024>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    read_last_run<<<1,1>>>(milliseconds);
    cudaDeviceSynchronize();
    /*******************************************************************/

    // Block Reduction
    /*******************************************************************/
    //fill vector with random values
    fill<<<1,1>>>();
    cudaDeviceSynchronize();

    //now reduce with block
    cudaEventRecord(start);
    parallel_redution_block<<<1, 1024>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    read_last_run<<<1,1>>>(milliseconds);
    cudaDeviceSynchronize();
    /*******************************************************************/

    // Block Reduction Branchless
    /*******************************************************************/    
    //fill vector with random values
    fill<<<1,1>>>();
    cudaDeviceSynchronize();

    //now reduce with block
    cudaEventRecord(start);
    parallel_redution_block_branchless<<<1, 1024>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    read_last_run<<<1,1>>>(milliseconds);
    cudaDeviceSynchronize();
    /*******************************************************************/

    // Block Reduction Interleaved
    /*******************************************************************/    
    //fill vector with random values
    fill<<<1,1>>>();
    cudaDeviceSynchronize();

    //now reduce with block
    cudaEventRecord(start);
    parallel_redution_block_interleaved<<<1, 1024>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    read_last_run<<<1,1>>>(milliseconds);
    cudaDeviceSynchronize();
    /*******************************************************************/

}
