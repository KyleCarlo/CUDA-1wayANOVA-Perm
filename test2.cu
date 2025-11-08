#include <stdio.h>
#include <stdlib.h>

#define MAX_LINE 1024

/* Linear Congruential Generator (LCG) */
__host__ __device__ unsigned int lcg_random(unsigned int seed) {
    return (1103515245U * (seed) + 12345U) & 0x7fffffffU;
}

/* Fisher–Yates Shuffling Algorithm */
__host__ __device__ void permute(size_t *array, size_t N, unsigned int seed, size_t *result) {
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
double OneWayAnova(size_t N, int k, size_t *n_i, size_t *group, double *feature){
    /* AVERAGE & GROUP AVERAGE */
    double group_ave[k];
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
        // FOR CHECKING ONLY
        if (i == perm_count - 1) {
            printf("GPU Permutation %d: ", i+1);
            for (int j = 0; j < N; j++) {
                printf("%llu ", perm_array[i * N + j]);
            }
            printf("\n");
        }
    }
}

int main() {
    size_t perm_count;
    size_t N;   // number of rows
    size_t k;   // number of groups

    /* GET THE NUMBER OF ROWS */
    printf("Number of Rows: ");
    scanf("%zu", &N);
    printf("Number of Groups: ");
    scanf("%zu", &k);
    printf("Number of Permutations: ");
    scanf("%zu", &perm_count);

    double *feature = (double*) malloc(N * sizeof(double));
    size_t *group = (size_t*) malloc(N * sizeof(size_t));
    size_t *temp_group = (size_t*) malloc(N * sizeof(size_t));
    size_t *n_i = (size_t*) calloc(k, sizeof(size_t));
    double *F_dist = (double*) malloc(perm_count * sizeof(double));

    /* READ THE DATA */
    FILE *fp = fopen("dataset.csv", "r");
    if (fp == NULL){
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
                if (group[i] >= k){
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

    /* CPU PERMUTATION */
    for (size_t i = 0; i < perm_count; i++) {
        printf("CPU Permutation %zu: ", i+1);
        for (size_t j = 0; j < N; j++) {
            if (i == 0)
                printf("%zu ", group[j]);
            else
                printf("%zu ", temp_group[j]);
        }
        printf("\n");
        // if (i == 0)
        //     F_dist[i] = OneWayAnova(N, k, n_i, group, feature);
        // else 
        //     F_dist[i] = OneWayAnova(N, k, n_i, temp_group, feature);
        // printf("::: F: %lf\n", F_dist[i]);
        permute(group, N, i, temp_group);
    }

    /* GPU PERMUTATION */
    size_t *perm_array;
    cudaMalloc(&perm_array, N * perm_count * sizeof(size_t));
    size_t *gpu_group;
    cudaMalloc(&gpu_group, N * sizeof(size_t));
    cudaMemcpy(gpu_group, group, N * sizeof(size_t), cudaMemcpyHostToDevice);
    gpu_permute<<< 1, 2 >>>(gpu_group, perm_count, N, perm_array);
    cudaDeviceSynchronize();

    /* free memory */
    free(feature);
    free(group);
    free(n_i);
    free(F_dist);

    return 0;
}