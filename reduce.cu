#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define BLOCKS 1
#define THREADS 1024

__device__ int vector[THREADS * BLOCKS];
__device__ int last_run;
__shared__ int reduced_vector[(THREADS * BLOCKS)/32];

__global__ void fill(){

    for (int i = 0; i < 1024; i++)
        vector[i] = i * 2;

}

__global__ void read_last_run(){

    printf("Last Run Result: %d\n", last_run);
    last_run = -1;

}

__global__ void parallel_redution(){

    //get our idex (assume 1d grids only)
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int item = vector[idx];
    int warp_lane = idx - ((int)(threadIdx.x / 32) * 32);

    //use warp shuffle to add items
    for (int offset = 16; offset > 0; offset /= 2)
        item += __shfl_down_sync(0xffffffff, item, offset);

    //have each warp store their value
    if (warp_lane == 0){
        __syncthreads();
        reduced_vector[idx/32] = item;
    }
        
        
    //make the first block's first warp do one more reduction
    if (blockIdx.x == 0 && idx < 32){

        //refetch
        item = reduced_vector[warp_lane];

        //use warp shuffle to add items
        for (int offset = 16; offset > 0; offset /= 2)
            item += __shfl_down_sync(0xffffffff, item, offset);

        //write
        if (warp_lane == 0){
            last_run = item;
        }
    }
    
    return;
}

__global__ void parallel_redution_interleaved(){
    return;
}

__global__ void parallel_redution_branchless(){
    return;
}

int main(){

    //fill vector with random values
    fill<<<1,1>>>();
    cudaDeviceSynchronize();

    //now reduce naively
    parallel_redution<<<1, 1024>>>();
    read_last_run<<<1,1>>>();
    cudaDeviceSynchronize();

    //now reduce naively
    parallel_redution_interleaved<<<1, 1024>>>();
    cudaDeviceSynchronize();

    //now reduce naively
    parallel_redution_branchless<<<1, 1024>>>();
    cudaDeviceSynchronize();

}
