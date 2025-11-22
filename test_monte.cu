%%writefile cuda_permutation_anova_1.cu
// WORKING VERSION
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LINE 1024

/* Linear Congruential Generator (LCG) */
__device__ unsigned int lcg_random(unsigned int seed) {
    return (1103515245U * (seed) + 12345U) & 0x7fffffffU;
}

/* Fisher–Yates Shuffling Algorithm */
__device__ void permute(size_t *array, size_t N, unsigned int seed, size_t *result) {
    for (size_t i = 0; i < N; i++) {
        result[i] = array[i];
    }
    for (size_t i = N - 1; i > 0; i--) {
        size_t j = lcg_random(seed) % (i + 1);
        size_t temp = result[i];
        result[i] = result[j];
        result[j] = temp;
    }
}

/* One Way ANOVA */
__device__ double OneWayAnova(size_t N, int k, size_t *n_i, size_t *group, double *feature){
    
    double group_ave[100];
    for (int i = 0; i < k; i++) {
        group_ave[i] = 0.0;
    }
        
    double average = 0.0;
    for (int i = 0; i < N; i++) {
        group_ave[group[i]] += feature[i];
        average += feature[i];
    }
    average /= N;
    for (int i = 0; i < k; i++) {
        group_ave[i] /= n_i[i];
    }

    /* SUM OF SQUARED ERROR (SSE) */
    double SSE = 0.0;
    double temp;
    for (int i = 0; i < N; i++) {
        temp = feature[i] - group_ave[group[i]];
        SSE += temp*temp;
    }

    /* SSR (SUM OF SQUARED RESIDUALS) */
    double SSR = 0.0;
    for (int i = 0; i < k; i++) {
        temp = group_ave[i] - average;
        SSR += n_i[i] * (temp * temp);
    }

    /* F-statistic */
    return (SSR/(k-1))/(SSE/(N-k));
}

__device__ void gpu_permute(size_t *array, size_t perm_count, size_t N, size_t *perm_array) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = thread_id; i < perm_count; i += stride) {
        if (i != 0) {
            permute(array, N, i-1, &perm_array[i * N]);
        } else {
            for (int j = 0; j < N; j++) {
                perm_array[j] = array[j];
            }
        }
    }
}

__device__ void gpu_anova(size_t *perm_array, size_t N, int k, size_t perm_count, double *feature, size_t *n_i, double *F_dist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int perm_idx = idx; perm_idx < perm_count; perm_idx += stride) {
        size_t *current_group = &perm_array[perm_idx * N];
        F_dist[perm_idx] = OneWayAnova(N, k, n_i, current_group, feature);
    }
}

__global__ void gpu_permute_and_anova(size_t *array, size_t perm_count, size_t N, int k, 
    double *feature, size_t *n_i, double *F_dist) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    size_t local_perm[N];

    for (int i = thread_id; i < perm_count; i += stride) {
        if (i == 0) {
            F_dist[0] = OneWayAnova(N, k, n_i, array, feature);
        } else {
            permute(array, N, i-1, local_perm);
            F_dist[i] = OneWayAnova(N, k, n_i, local_perm, feature);
        }
    }
}

int main() {
    size_t perm_count;
    size_t N;
    size_t k;
    size_t counter = 10;

    printf("Number of Rows: ");
    scanf("%zu", &N);
    printf("Number of Groups: ");
    scanf("%zu", &k);
    printf("Number of Permutations: ");
    scanf("%zu", &perm_count);

    // Get GPU device
    int device = -1;
    cudaGetDevice(&device);

    // Memory allocation
    double *feature;
    size_t *group;
    size_t *n_i;
    size_t *perm_array;
    double *F_dist;

    cudaMallocManaged(&feature, N * sizeof(double));
    cudaMallocManaged(&group, N * sizeof(size_t));
    cudaMallocManaged(&n_i, k * sizeof(size_t));
    cudaMallocManaged(&perm_array, N * perm_count * sizeof(size_t));
    cudaMallocManaged(&F_dist, perm_count * sizeof(double));

    // Initialize n_i to zero
    memset(n_i, 0, k * sizeof(size_t));

    // MEMORY ADVISE: Set up for input data (feature, group, n_i)
    cudaMemAdvise(feature, N * sizeof(double), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(group, N * sizeof(size_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(n_i, k * sizeof(size_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);


    // Prefetch data to CPU memory
    cudaMemPrefetchAsync(feature, N * sizeof(double), cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(group, N * sizeof(size_t), cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(n_i, k * sizeof(size_t), cudaCpuDeviceId, NULL);


    // READ DATA FROM FILE
    FILE *fp = fopen("dataset.csv", "r");
    if (fp == NULL){
        perror("Error opening file");
        return 1;
    }

    char line[MAX_LINE];
    size_t i = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (i >= N) break;

        line[strcspn(line, "\n")] = 0;

        char *token = strtok(line, ",");
        int j = 0;
        while (token != NULL) {
            if (j == 0)
                feature[i] = atof(token);
            else {
                group[i] = atoi(token);
                if (group[i] >= k){
                    perror("Error group count");
                    fclose(fp);
                    return 1;
                }
                n_i[group[i]] += 1;
            }
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(fp);
    
    // PREFETCH: Move input data to GPU before computation
    cudaMemPrefetchAsync(feature, N * sizeof(double), device, NULL);
    cudaMemPrefetchAsync(group, N * sizeof(size_t), device, NULL);
    cudaMemPrefetchAsync(n_i, k * sizeof(size_t), device, NULL);
    
    // Prefetch output arrays to GPU
    cudaMemPrefetchAsync(perm_array, N * perm_count * sizeof(size_t), device, NULL);
    cudaMemPrefetchAsync(F_dist, perm_count * sizeof(double), device, NULL);

    // Wait for prefetch to complete
    cudaDeviceSynchronize();

    // Number of Threads and Blocks
    size_t numThreads = 256;
    size_t numBlocks = (perm_count + numThreads - 1) / numThreads;

    
    printf("\n Generating Permutations and Computing F-statistic\n");
    printf("Launching kernel with %zu blocks and %zu threads per block\n", numBlocks, numThreads);
    
    for (size_t c = 0; c < counter; c++){
        gpu_permute_and_anova<<<numBlocks, numThreads>>>(
            group, perm_count, N, k, feature, n_i, F_dist);
    }
    cudaDeviceSynchronize();

    // PREFETCH: Move results back to CPU for printing
    cudaMemPrefetchAsync(perm_array, N * perm_count * sizeof(size_t), cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(F_dist, perm_count * sizeof(double), cudaCpuDeviceId, NULL);

    // PRINT RESULTS
    printf("\nDEBUG: First 5 permutations\n");
    for (int i = 0; i < 5; i++) {
        printf("GPU Permutation %d: ", i + 1);
        for (int j = 0; j < N && j < 20; j++) {  // Limit output for readability
            printf("%zu ", perm_array[i * N + j]);
        }
        if (N > 20) printf("...");
        printf("\n");
    }

    printf("\n Printing Results\n");
    printf("First 5 F-statistics:\n");
    for (int i = 0; i < 5; i++) {
        printf("F_dist[%d]: %lf\n", i, F_dist[i]);
    }

    // Calculate p-value
    size_t extreme_count = 0;
    for (size_t i = 1; i < perm_count; i++) {
        if (F_dist[i] >= F_dist[0]) {
            extreme_count++;
        }
    }
    double p_value = (double)extreme_count / (double)perm_count;
    printf("\nOriginal F-statistic: %lf\n", F_dist[0]);
    printf("Extreme count: %zu out of %zu permutations\n", extreme_count, perm_count);
    printf("p-value: %lf\n", p_value);
    

    // Free memory
    cudaFree(feature);
    cudaFree(group);
    cudaFree(n_i);
    cudaFree(perm_array);
    cudaFree(F_dist);

    return 0;
}