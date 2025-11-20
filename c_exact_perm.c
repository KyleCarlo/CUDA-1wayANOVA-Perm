
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LINE 1024

/* UTILITIES FOR 128-BIT INTEGER EXACT PRINTING */
typedef unsigned __int128 u128;

// Print u128 EXACTLY
void print_u128(u128 x) {
    if (x == 0) { printf("0"); return; }
    char buf[128];
    int p = 0;
    while (x > 0) {
        buf[p++] = '0' + (x % 10);
        x /= 10;
    }
    while (p--) putchar(buf[p]);
}


/* EXACT BINOMIAL & MULTINOMIAL FUNCTIONS */
// Compute C(n,k) exactly using 128-bit integers
u128 binom_u128(u128 n, u128 k) {
    if (k > n) return 0;
    if (k > n - k) k = n - k;

    u128 result = 1;
    for (u128 i = 1; i <= k; i++) {
        result = result * (n - k + i) / i;
    }
    return result;
}

/* EXACT multinomial coefficient using sequential binomial method */
u128 getCountPerm_u128(int total_elements, size_t *repeats, int k) {
    u128 result = 1;
    int remaining = total_elements;

    for (int i = 0; i < k; i++) {
        int ni = repeats[i];
        u128 c = binom_u128(remaining, ni);
        result *= c;
        remaining -= ni;
    }
    return result;
}

// helper function for swapping values
void Exchange(int* data, int a, int b) {
    int temp = data[a];
    data[a] = data[b];
    data[b] = temp;
}

/* PERMUTATION GENERATOR */
int permute(int a[], int n) {
    int l, j;
    for (j = --n; j > 0 && a[j-1] >= a[j]; --j) { ; }
    if (j == 0) return 0;
    for (l = n; a[j-1] >= a[l]; --l) { ; }
    Exchange(a, j-1, l);
    while (j < n) { Exchange(a, j++, n--); }
    return 1;
}

/* ONE WAY ANALYSIS OF VARIANCE */
double OneWayAnova(size_t N, int k, size_t *n_i, int *group, double *feature){
    double *group_ave = calloc(k, sizeof(double));

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

    free(group_ave);

    /* F-statistic */
    return (SSR/(k-1))/(SSE/(N-k));
}

int main() {
    size_t N;
    int k;
    clock_t start, end;
    size_t counter = 10;

    printf("Number of Rows: ");
    scanf("%zu", &N);
    printf("Number of Groups: ");
    scanf("%d", &k);

    double *feature = malloc(N * sizeof(double));
    int *group = malloc(N * sizeof(int));
    int *group_copy = malloc(N * sizeof(int));
    size_t *group_duplicates = calloc(k, sizeof(size_t));

    FILE *fp = fopen("dataset.csv", "r");
    if (fp == NULL) {
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
                    return 1;
                }
                group_duplicates[group[i]] += 1;
            }
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(fp);

    memcpy(group_copy, group, N * sizeof(int));

    /* EXACT PERMUTATION COUNT USING 128-BIT INTEGER */
    u128 perm_count = getCountPerm_u128(N, group_duplicates, k);

    // Can over flow for large n
    double *F_dist = malloc(perm_count * sizeof(double));

    // Execution time start here: CPU Permutation
    /* CPU PERMUTATION */
    double elapse = 0.0f, 
           time_taken;

    /* PERMUTATION TEST */
    for (int c=0; c<counter; c++){
        start = clock();
        for (i = 0; i < perm_count; i++){
            // compute One Way ANOVA
            F_dist[i] = OneWayAnova(N, k, group_duplicates, group, feature);

            // This block is for debugging only
            // printf("%zu: ", i);
            // for (int j = 0; j < N; j++){
            //     printf("%d ", group[j]);
            // }
            // printf("; F = %lf\n", F_dist[i]);

            // change grouping assignment
            permute(group, N);
        }
        end = clock();
        time_taken = ((double)(end-start))*1E3/CLOCKS_PER_SEC;
        elapse = elapse + time_taken;
        memcpy(group, group_copy, N * sizeof(int));
    }
    printf("\nFunction (in C) average time for %lu loops is %f milliseconds to execute an array size ", counter, elapse/counter);
    print_u128(perm_count);
    printf("\n");

    printf("\nPrinting First 5 and Last 5 Results\n");
    for (int i = 0; i < 5; i++) {
        printf ("F_dist %d: %lf\n", i, F_dist[i]);
    }
    printf("=================================\n");
    for (int i = perm_count-5; i < perm_count; i++) {
        printf ("F_dist %d: %lf\n", i, F_dist[i]);
    } 

    size_t extreme_count = 0;
    for (size_t i = 1; i < perm_count; i++) {
        if (F_dist[i] >= F_dist[0]) {
            extreme_count++;
       }
    }

    // Calculating the p-value for the permutation test
    double p_value = (double)extreme_count/perm_count;
    printf("\nNull F: %lf\n", F_dist[0]);
    printf ("Extreme Count: %lu\n", extreme_count);
    p_value = (double)extreme_count / perm_count;
    printf("p-value: %lf\n", p_value);

    // free the allocated memory
    free(feature);
    free(group);
    free(group_duplicates);
    free(F_dist);

    return 0;
}
