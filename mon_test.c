#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* PERMUTATION GENERATOR */
void Exchange(int* data, int a, int b) {
    int temp = data[a];
    data[a] = data[b];
    data[b] = temp;
}

int genPermutation(int a[], int n) {
    int l, j;
    for (j = --n; j > 0 && a[j-1] >= a[j]; --j) { ; }
    if (j == 0) return 0;
    for (l = n; a[j-1] >= a[l]; --l) { ; }
    Exchange(a, j-1, l);
    while (j < n) { Exchange(a, j++, n--); }
    return 1;
}

double OneWayAnova(size_t N, int k, size_t *n_i, int *group, double *feature){

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

int main() {
    size_t n;
    int k;

    printf("Number of Rows: ");
    scanf("%zu", &n);
    printf("Number of Groups: ");
    scanf("%d", &k);

    double *feature = malloc(n * sizeof(double));
    int *group = malloc(n * sizeof(int));
    size_t *group_duplicates = calloc(k, sizeof(size_t));

    FILE *fp = fopen("dataset_1.csv", "r");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }

    char line[MAX_LINE];
    size_t i = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (i >= n) break;

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

    /* EXACT PERMUTATION COUNT USING 128-BIT INTEGER */
    u128 n_perms = getCountPerm_u128(n, group_duplicates, k);

    printf("\nTotal distinct permutations = ");
    print_u128(n_perms);
    printf("\n\n");

    // Can over flow for large n
    int *group_perms = malloc(n * n_perms * sizeof(int));

    for (i = 0; i < n_perms; i++){
        printf("%zu: ", i);
        for (int j = 0; j < n; j++){
            printf("%d ", group[j]);
            group_perms[(i*n) + j] = group[j];
        }
        printf("\n");
        genPermutation(group, n);
    }

    /* ONE WAY ANOVA */
    printf("F=%lf\ns", OneWayAnova(n, k, group_duplicates, group, feature));
    free(feature);
    free(group);
    free(group_duplicates);

    return 0;
}