#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 1024

long double factorial(long double num){
    long double answer = 1;
    for (int i = 2; i <= num; i++) {
        answer *= i;
    }

    return answer;
}

long double getCountPerm(int total_elements, size_t *repeats, int k) {
    long double divisor = 1;
    for (int i = 0; i < k; i++) {
        divisor *= factorial(repeats[i]);
    }
    long double answer = factorial(total_elements)/divisor;
    return answer;
}

void Exchange(int* data, int a, int b) {
  int temp = data[a];
  data[a]=data[b];
  data[b]=temp;
}

//generates all permutations of initially sorted array `a` with `n` elements
//returns 0 when no more permutations exist
int genPermutation(int a[], int n) 
{
  int l,j;
  for (j = --n; j > 0 && a[j-1] >= a[j]; --j) { ; }
  if (j == 0) return 0; 
  for (l = n; a[j-1] >= a[l]; --l) { ; }
  Exchange(a, j-1, l);
  while (j < n) { Exchange(a, j++ ,n--); }
  return 1;
}

// Fisher–Yates shuffle
// PERMUTATION IF WE WANT MONTE CARLO
// void randomPermutation(int *array, size_t n) {
//     for (size_t i = n - 1; i > 0; i--) {
//         size_t j = rand() % (i + 1);  // pick random index [0, i]
//         int temp = array[i];
//         array[i] = array[j];
//         array[j] = temp;
//     }
// }

int main() {
    size_t n;   // number of rows
    int k;      // number of groups

    /* GET THE NUMBER OF ROWS */
    printf("Number of Rows: ");
    scanf("%zu", &n);
    printf("Number of Groups: ");
    scanf("%d", &k);

    double *feature = malloc(n * sizeof(double));
    int *group = malloc(n * sizeof(int));
    size_t *group_duplicates = calloc(k, sizeof(size_t));

    /* READ THE DATA */
    FILE *fp = fopen("dataset.csv", "r");
    if (fp == NULL){
        perror("Error opening file");
        return 1;
    }

    char line[MAX_LINE];

    // Read each line
    size_t i = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (i >= n) break;  // prevent overflow

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
                group_duplicates[group[i]] += 1;
            }
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(fp);

    /* PERMUTATE */
    long double n_perms = getCountPerm(n, group_duplicates, k);
    int *group_perms = malloc(n * n_perms * sizeof(int));

    for (i = 0; i <= n_perms; i++){
        printf("%zu: ", i);
        for (int j = 0; j < n; j++){
            printf("%d ", group[j]);
            group_perms[(i*n) + j] = group[j];
        }
        printf("\n");
        genPermutation(group, n);
    }

    printf("nperms: %Lf\n", n_perms);
    for (int j = 0; j < n_perms; j++){
        printf("Perm %d: ", j);
        for (int k = 0; k < n; k++){
            printf("%d ", group_perms[(j*n)+k]);
        }
        printf("\n");
    }
    free(feature);
    free(group);

    return 0;
}