%%writefile cuda_permutation_anova.cu

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
        size_t j = lcg_random(seed) % (i + 1);  // pick random index [0, i]
        size_t temp = result[i];
        result[i] = result[j];
        result[j] = temp;
    }
}

/* One Way ANOVA */
__host__ __device__ double OneWayAnova(size_t N, int k, size_t *n_i, size_t *group, double *feature){
    /* AVERAGE & GROUP AVERAGE */
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

__global__ void gpu_permute(size_t *array, size_t perm_count, size_t N, size_t *perm_array) {
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

// GPU One-Way ANOVA
__global__ void gpu_anova(size_t *perm_array, size_t N, int k, size_t perm_count, double *feature, size_t *n_i, double *F_dist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int perm_idx = idx; perm_idx < perm_count; perm_idx += stride) {
        size_t *current_group = &perm_array[perm_idx * N];

        // Compute F-statistic score
        F_dist[perm_idx] = OneWayAnova(N, k, n_i, current_group, feature);
    }
}

int main() {
    size_t perm_count;
    size_t N;   // number of rows
    size_t k;   // number of groups
    size_t counter = 1;

    /* GET THE NUMBER OF ROWS */
    printf("Number of Rows: ");
    scanf("%zu", &N);
    printf("Number of Groups: ");
    scanf("%zu", &k);
    printf("Number of Permutations: ");
    scanf("%zu", &perm_count);

    // Memory alloc
    double *feature; // INPUT
    size_t *group; // INPUT
    size_t *n_i; // input number of samples per group
    size_t *perm_array; // Output for permutations
    double *F_dist; // Output for F-statistic

    // Number of Threads and Blocks
    size_t numThreads = 1024;
    size_t numBlocks = (perm_count + numThreads-1) / numThreads;


    // Memory allocation
    cudaMallocManaged(&feature, N * sizeof(double));
    cudaMallocManaged(&group, N * sizeof(size_t));
    cudaMallocManaged(&n_i, k * sizeof(size_t));
    cudaMallocManaged(&perm_array, N * perm_count * sizeof(size_t));
    cudaMallocManaged(&F_dist, perm_count * sizeof(double));

    // Get GPU device ID
    int device = -1;
    cudaGetDevice(&device);

    // Memory advice (set preferred location and read-mostly for CPU data)
    cudaMemAdvise(feature, N * sizeof(double), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(feature, N * sizeof(double), cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
    cudaMemAdvise(group, N * sizeof(size_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(group, N * sizeof(size_t), cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
    cudaMemAdvise(n_i, k * sizeof(size_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(n_i, k * sizeof(size_t), cudaMemAdviseSetReadMostly, cudaCpuDeviceId);

    // Prefetch data to CPU memory (initial data population happens on the CPU)
    cudaMemPrefetchAsync(feature, N * sizeof(double), cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(group, N * sizeof(size_t), cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(n_i, k * sizeof(size_t), cudaCpuDeviceId, NULL);

    // Read the dataset (CPU-side operation)
    FILE *fp = fopen("dataset.csv", "r");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }

    char line[MAX_LINE];
    size_t i = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (i >= N) break;  // prevent overflow

        line[strcspn(line, "\n")] = 0;

        char *token = strtok(line, ",");
        int j = 0;
        while (token != NULL) {
            if (j == 0)
                feature[i] = atof(token); // convert to float and save
            else {
                group[i] = atoi(token); // convert to int and save
                if (group[i] >= k) {
                    perror("Error group count");
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

    // Prefetch data to GPU memory (ready for GPU computation)
    cudaMemPrefetchAsync(feature, N * sizeof(double), device, NULL);
    cudaMemPrefetchAsync(group, N * sizeof(size_t), device, NULL);
    cudaMemPrefetchAsync(n_i, k * sizeof(size_t), device, NULL);

    // Memory advice for GPU computation
    cudaMemAdvise(perm_array, N * perm_count * sizeof(size_t), cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(perm_array, N * perm_count * sizeof(size_t), cudaMemAdviseSetReadMostly, device);

    // Prefetch output arrays to GPU memory
    cudaMemPrefetchAsync(perm_array, N * perm_count * sizeof(size_t), device, NULL);
    cudaMemPrefetchAsync(F_dist, perm_count * sizeof(double), device, NULL);

    /* GPU PERMUTATION */
    printf("\nSTEP 1: Generating Permutations\n");
    printf("Launching kernel with %zu blocks and %zu threads per block\n", numBlocks, numThreads);
    for (size_t c = 0; c < counter; c++) {
        gpu_permute<<<numBlocks, numThreads>>>(group, perm_count, N, perm_array);
    }
    cudaDeviceSynchronize();

    // Prefetch permutation array back to CPU for debugging (optional)
    cudaMemPrefetchAsync(perm_array, N * perm_count * sizeof(size_t), cudaCpuDeviceId, NULL);

    /* GPU ANOVA */
    printf("\nSTEP 2: Computing F-statistic\n");
    printf("Launching kernel with %zu blocks and %zu threads per block\n", numBlocks, numThreads);
    for (size_t c = 0; c < counter; c++) {
        gpu_anova<<<numBlocks, numThreads>>>(perm_array, N, k, perm_count, feature, n_i, F_dist);
    }
    cudaDeviceSynchronize();

    // Prefetch F-statistic results back to CPU for analysis
    cudaMemPrefetchAsync(F_dist, perm_count * sizeof(double), cudaCpuDeviceId, NULL);

    // Debugging: Print first 5 permutations and F-statistics
    for (int i = 0; i < 5; i++) {
        printf("GPU Permutation %d: ", i + 1);
        for (int j = 0; j < N; j++) {
            printf("%zu ", perm_array[i * N + j]);
        }
        printf("\n");
    }
    printf("\nSTEP 3: Printing Results\n");
    for (int i = 0; i < 5; i++) {
        printf("F_dist %d: %lf\n", i, F_dist[i]);
    }

    // Compute p-value
    size_t extreme_count = 0;
    double p_value = 0.0;
    for (size_t i = 1; i < perm_count; i++) {
        if (F_dist[i] >= F_dist[0]) {
            extreme_count++;
        }
    }
    p_value = (double)extreme_count / perm_count;
    printf("p-value: %lf\n", p_value);


    /* Free memory */
    cudaFree(feature);
    cudaFree(group);
    cudaFree(n_i);
    cudaFree(perm_array);
    cudaFree(F_dist);

    return 0;
}

