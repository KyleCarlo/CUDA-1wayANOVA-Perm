#include <stdio.h>
#include <cuda_runtime.h>

__host__ __device__ int lcg_random(unsigned int seed) {
    seed = (1103515245U * (seed) + 12345U) & 0x7fffffffU;
    return static_cast<int>(seed);
}

// GPU kernel (writes integers)
__global__ void generateRandom(int *out, int n, unsigned int seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        unsigned int localSeed = seed;  // same start for all
        for (int i = 0; i <= id; i++) { // advance same number of times
            lcg_random(localSeed);
        }
        out[id] = (int)localSeed; // store integer
    }
}

int main() {
    const int N = 10;
    unsigned int seed = 1234;
    int cpu_vals[N];
    int gpu_vals[N];
    int *d_vals;

    // --- CPU version ---
    unsigned int cpu_seed = seed;
    for (int i = 0; i < N; i++) {
        cpu_vals[i] = lcg_random(&cpu_seed);
    }

    // --- GPU version ---
    cudaMalloc(&d_vals, N * sizeof(int));
    generateRandom<<<1, N>>>(d_vals, N, seed);
    cudaMemcpy(gpu_vals, d_vals, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_vals);

    // --- Compare ---
    printf("Index\tCPU\t\tGPU\n");
    for (int i = 0; i < N; i++) {
        printf("%d\t%d\t%d\n", i, cpu_vals[i], gpu_vals[i]);
    }

    return 0;
}