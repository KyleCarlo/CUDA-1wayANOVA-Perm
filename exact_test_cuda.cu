#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 1024
#define MAX_GROUPS 10

// Factorial for small numbers (up to ~20)
__device__ unsigned long long factorial(int n) {
    unsigned long long result = 1;
    for (int i = 2; i <= n; i++)
        result *= i;
    return result;
}

// Multinomial coefficient on device
__device__ unsigned long long multinomial(int total, int *counts, int n_keys) {
    unsigned long long result = factorial(total);
    for (int i = 0; i < n_keys; i++) {
        result /= d_factorial(counts[i]);
    }
    return result;
}

// Generate permutation from rank
__device__ void rank_to_permutation(int *keys, int *group_counts, int k, int N, unsigned long long rank, int *perm 
) {
    int group_counts_copy[MAX_GROUPS];
    
    for (int i = 0; i < n_keys; i++) {
        group_counts_copy[i] = group_counts[i];
    }
    
    int total = n;
    
    for (int pos = 0; pos < N; pos++) {
        for (int i = 0; i < k; i++) {
            if (group_counts_copy[i] == 0)
                continue;
            
            group_counts_copy[i]--;
            unsigned long long num = multinomial(total - 1, group_counts_copy, n_keys);
            
            if (rank < num) {
                perm[pos] = keys[i];
                total--;
                break;
            } else {
                rank -= num;
                group_counts_copy[i]++;
            }
        }
    }
}

// One-Way ANOVA calculation on device
__device__ double one_way_anova(int N, int k, int *group_counts, int *group, double *feature) {
    double group_ave[MAX_GROUPS] = {0.0};
    
    // Calculate group means and overall mean
    double average = 0.0;
    for (int i = 0; i < N; i++) {
        group_ave[group[i]] += feature[i];
        average += feature[i];
    }
    average /= N;
    
    for (int i = 0; i < k; i++) {
        if (group_counts[i] > 0)
            group_ave[i] /= group_counts[i];
    }
    
    // Calculate SSE (within-group variation)
    double SSE = 0.0;
    for (int i = 0; i < N; i++) {
        double temp = feature[i] - group_ave[group[i]];
        SSE += temp * temp;
    }
    
    // Calculate SSR (between-group variation)
    double SSR = 0.0;
    for (int i = 0; i < k; i++) {
        double temp = group_ave[i] - average;
        SSR += group_counts[i] * (temp * temp);
    }
    
    // F-statistic
    return (SSR / (k - 1)) / (SSE / (N - k));
}

__global__ void permutation_test_gpu(int N, int k, int *keys, int *group_counts, double *features, unsigned long long total_perms, double *F_dist
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int perm_idx = idx; perm_idx < perm_count; perm_idx += stride) {
        unsigned long long rank = perm_idx;
        
        if (rank >= total_perms)
            return;
        
        // Allocate local array for the permutation
        int perm[N];
        
        // Generate permutation for this rank 
        rank_to_permutation(keys, group_counts, k, N, rank, perm);
        
        // Calculate F-statistic for this permutation
        double F_stat = one_way_anova(N, k, group_counts, perm, features);
        
        F_dist[rank] = F_stat;
    }
}

/* Functions that ran on cpu */

// Binomial coefficient for permutation count
unsigned long long binom(int n, int k) {
    if (k > n) return 0;
    if (k > n - k) k = n - k;
    
    unsigned long long result = 1;
    for (int i = 1; i <= k; i++) {
        result = result * (n - k + i) / i;
    }
    return result;
}

// Calculate total permutations (multinomial coefficient)
unsigned long long get_perm_count(int total_elements, int *repeats, int k) {
    unsigned long long result = 1;
    int remaining = total_elements;
    
    for (int i = 0; i < k; i++) {
        int ni = repeats[i];
        unsigned long long c = binom(remaining, ni);
        result *= c;
        remaining -= ni;
    }
    return result;
}

int main() {
    int N, k;
    `
    printf("Number of Rows: ");
    scanf("%d", &N);
    printf("Number of Groups: ");
    scanf("%d", &k);

    // Get GPU device
    int device = -1;
    cudaGetDevice(&device);

    double *feature;
    int *group;
    int *group_counts;
    int *keys;
    double *F_dist;

    cudaMallocManaged(&feature, N * sizeof(double));
    cudaMallocManaged(&group, N * sizeof(int));
    cudaMallocManaged(&group_counts, k * sizeof(int));
    cudaMallocManaged(&keys, k * sizeof(int));
    cudaMallocManaged(&F_dist, total_perms * sizeof(double));
    
    // Initialize group_counts to zero
    memset(group_counts, 0, k * sizeof(int));

    cudaMemAdvise(feature, N * sizeof(double), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(group, N * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(group_counts, k * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(keys, k * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

    cudaMemPrefetchAsync(feature, N * sizeof(double), cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(group, N * sizeof(int), cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(group_counts, k * sizeof(int), cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(keys, k * sizeof(int), cudaCpuDeviceId, NULL);
    
    
    FILE *fp = fopen("dataset.csv", "r");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }
    
    char line[MAX_LINE];
    int i = 0;
    
    while (fgets(line, sizeof(line), fp) && i < N) {
        line[strcspn(line, "\n")] = 0;
        char *token = strtok(line, ",");
        int j = 0;
        
        while (token != NULL) {
            if (j == 0)
                h_feature[i] = atof(token);
            else {
                h_group[i] = atoi(token);
                if (h_group[i] >= k) {
                    fprintf(stderr, "Error: group index out of range\n");
                    return 1;
                }
                h_group_counts[h_group[i]]++;
            }
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(fp);
    
    // Create keys array [0, 1, 2, ..., k-1] to map group indices to group labels
    for (int i = 0; i < k; i++) {
        keys[i] = i;
    }
    
    // Calculate total permutations
    unsigned long long total_perms = get_perm_count(N, h_group_counts, k);
    printf("Total permutations: %llu\n", total_perms);

    cudaMemPrefetchAsync(keys, k * sizeof(int), device, NULL);
    cudaMemPrefetchAsync(group_counts, k * sizeof(int), device, NULL);
    cudaMemPrefetchAsync(feature, N * sizeof(double), device, NULL);
    cudaMemPrefetchAsync(F_dist, total_perms * sizeof(double), device, NULL);
    
    // Launch kernel
    size_t numThreads = 256;
    size_t numBlocks = (total_perms + numThreads - 1) / numThreads;
    
    printf("Launching %d blocks with %d threads per block\n", numBlocks, numThreads);
    
    
    permutation_test_gpu<<<num_blocks, numThreads>>>(
        N, k, d_keys, d_group_counts, d_feature, total_perms, d_F_dist
    );
    
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(F_dist, total_perms * sizeof(double), cudaCpuDeviceId, NULL);
    
    printf("\nFirst 5 F-statistics:\n");
    for (int i = 0; i < 5 && i < total_perms; i++) {
        printf("F_dist[%d]: %.6f\n", i, F_dist[i]);
    }
    
    printf("\nLast 5 F-statistics:\n");
    int start_idx = (total_perms > 5) ? total_perms - 5 : 0;
    for (unsigned long long i = start_idx; i < total_perms; i++) {
        printf("F_dist[%llu]: %.6f\n", i, F_dist[i]);
    }
    
    // Calculate p-value
    double original_F = F_dist[0];
    unsigned long long extreme_count = 0;
    
    for (unsigned long long i = 0; i < total_perms; i++) {
        if (F_dist[i] >= original_F) {
            extreme_count++;
        }
    }
    
    double p_value = (double)extreme_count / total_perms;
    printf("\nOriginal F-statistic: %.6f\n", original_F);
    printf("Extreme count: %llu\n", extreme_count);
    printf("P-value: %.6f\n", p_value);
    
    // Cleanup
    cudaFree(feature);
    cudaFree(group);
    cudaFree(group_counts);
    cudaFree(keys);

    
    return 0;
}