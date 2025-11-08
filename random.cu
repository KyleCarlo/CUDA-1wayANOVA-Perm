#include <stdio.h>
#include <cuda_runtime.h>

__host__ __device__ unsigned int lcg_random(unsigned int *seed) {
    *seed = (1103515245U * (*seed) + 12345U) & 0x7fffffffU;
    return *seed;
}

// GPU kernel
__global__ void generateRandom(float *out, int n, unsigned int seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        unsigned int localSeed = seed;  // same start for all
        for (int i = 0; i <= id; i++) { // advance same number of times
            lcg_random(&localSeed);
        }
        out[id] = (float)localSeed / 0x7fffffffU;
    }
}

int main() {
    const int N = 10;
    unsigned int seed = 1234;
    float cpu_vals[N];
    float gpu_vals[N];
    float *d_vals;

    // --- CPU version ---
    unsigned int cpu_seed = seed;
    for (int i = 0; i < N; i++) {
        unsigned int r = lcg_random(&cpu_seed);
        cpu_vals[i] = (float)r / 0x7fffffffU;
    }

    // --- GPU version ---
    cudaMalloc(&d_vals, N * sizeof(float));
    generateRandom<<<1, N>>>(d_vals, N, seed);
    cudaMemcpy(gpu_vals, d_vals, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_vals);

    // --- Compare ---
    printf("Index\tCPU\t\t\tGPU\n");
    for (int i = 0; i < N; i++) {
        printf("%d\t%f\t%f\n", i, cpu_vals[i], gpu_vals[i]);
    }

    return 0;
}
