# GPU-Accelerated Permutation Test for One-Way ANOVA  
## **Integrating Project — C vs CUDA Performance Evaluation**

### Members:
- Go, Daphne
- Lasala, Kyle
- Manlises, Monica

---

# 1. Overview

This project implements both ***exact*** and ***Monte Carlo permutation tests*** for the ***one-way ANOVA*** statistic, comparing:

- ***Serial C implementation***
- ***CUDA-accelerated GPU implementation (SIMT model)***

Permutation tests are widely used in fields such as neuroimaging, genomics, and experimental sciences, but become computationally expensive due to the factorial growth of possible permutations. This project benchmarks performance, evaluates memory limitations, and quantifies the accuracy of Monte Carlo approximations against exact permutation results.

### **Inputs to Project**
- Data matrix: $X = \{ x_{ij} \}$
- Group labels: $g_1, g_2, \ldots, g_N$
- Number of groups: $k$
- Number of permutations: $P$

### **Outputs**
- Exact null distribution of $F$ test statistic 
- $p$-value for the permutation test.
- Performance metrics: runtimes (serial vs CUDA), speedup factors, GPU utilization.
- Output accuracy: outputs are compared to the ground truth, and errors will be shown.


---

# 2. Sequential Algorithm (C Implementation)

The sequential permutation test computes the ANOVA statistic for each permutation one at a time.

### **Step 1 — Generate a permutation of labels**
3 ways:
- Lexicographic Next Permutation (for Exact Permutation Test)
    - This algorithm generates a permutation dependent from the previous permutation
- LCG + Fisher–Yates shuffle (for Monte Carlo Method)

### **Step 2 — Compute group means**
$$
\mu_i = \frac{1}{n_i} \sum_{j=1}^{n_i} x_{ij}
$$

### **Step 3 — Compute within-group sum of squares**
$$
SSE = \sum_{i=1}^k \sum_{j=1}^{n_i} (x_{ij} - \mu_i)^2
$$

### **Step 4 — Compute between-group sum of squares**
$$
SSR = \sum_{i=1}^k n_i (\mu_i - \mu)^2
$$

### **Step 5 — Compute ANOVA statistic**
$$
F = \frac{SSR/(k-1)}{SSE/(N-k)}
$$

### **Step 6 — Repeat Steps 2–5 for all $P$ permutations**

### **Step 7 — Compute final $p$-value**

---

# 3. Parallel Algorithm (CUDA Implementation)

### What was parallelized?
1. Creation of the permutations (Sequential Step 1)
    - ***Rank-based permutation*** (for Exact Permutation Test)
      - In contrast to Lexicographic Next, Rank-based is a parallelization method that allows the permutations to be independently created without relying on the previous permutation.
      - This essentially makes one permutation a GPU thread which will be able to complete the next steps independently
2. Looping across all Permutations (Sequential Step 6)
    - Instead of evaluating permutations inside a large for-loop, CUDA executes thousands of them in parallel utilizing the Single Instruction, Multiple Threads (SIMT) model where each permutation is a thread.

3. Calculation of $F$-Statistic (Sequential Steps 2-5)
    - Within each thread/permutation, there is another thread that computes for the $F$-Statistic since it involves loops for the summations.

### Key idea:
> ***Each permutation becomes an independent GPU thread.***

---

# 4. Sequential → Parallel Conversion

## 4.1 Sections Parallelized

| Sequential Step | Parallel CUDA Version |
|-----------------|------------------------|
| Generate all $P$ permutations | ***Mapped to GPU threads*** (1 thread = 1 permutation) |
| Loop over all $P$ permutations | Not needed (1 thread = 1 permutation) |
| Compute $\mu_i$ | Each thread computes means for its permutation |
| Compute $SSE$ | Thread-local loop inside kernel |
| Compute $SSR$ | Thread-local loop inside kernel |
| Compute $F$ | Each thread computes and stores its own result |

These are the ***core computational bottlenecks*** and are ideal for parallelism because permutations are independent.

---

## 4.2 CUDA Kernel Structure (Per Thread)

Each thread performs:

1. Identification of permutation index  
2. Generation or application of its assigned permutation  
3. Building of group assignments  
4. Computation of:
   - $n_i$ for all groups  
   - $\mu_i$  
   - $SSE$  
   - $SSR$  
   - $F$
5. Writing of result to global memory

### CUDA Optimizations:
- Unified memory  
- `cudaMemAdvise`  
- `cudaMemPrefetchAsync`  
- Grid-stride loops  
- Shared memory caching  

---

# 5. Components That Remain Sequential

| Component | Reason |
|----------|--------|
| Permutation generation | Lexicographic permutation is inherently sequential |
| $p$-value computation | Final reduction step, minimal cost |
| Input preprocessing | Lightweight, no benefit from parallelism |

These steps do not benefit meaningfully from GPU parallelization.

---

# 6. Why These Steps Were Parallelized

The total cost is:

$$
\text{Cost} = P \times O(N)
$$

Because each permutation is ***independent***:

- Ideal for data-parallel execution  
- GPU can handle tens of thousands of permutations simultaneously  
- Sequential algorithm becomes infeasible for large $P$  

Using CUDA reduces effective time to roughly:

$$
O\left(\frac{P}{\text{threads}}\right)
$$

which yields ***significant acceleration***.

---
# 7. Results: <!--TODO-->
## 7.1. Accuracy
### 7.1.1. Sequential Execution (C)
1.  ***Screenshot***

### 7.1.2. Sequential Execution (C)
1.  ***Screenshot***

## 7.2.  Comparative Table of Average Execution Time
<!--
    <table>
      <thead>

      <tr>
      <th colspan="2">Input Sizes</th>
      <th colspan="2">2<sup>20</sup></th>
      <th colspan="2">2<sup>26</sup></th>
      <th colspan="2">2<sup>28</sup></th>
      </tr>
      
      <tr>
      <th colspan="2">Mode</th>
      <th>Debug</th>
      <th>Release</th>
      <th>Debug</th>
      <th>Release</th>
      <th>Debug</th>
      <th>Release</th>
      </tr>

      </thead>
      <tbody>
      <tr>
      <th rowspan="4">Implementation</th>
      <th>C</th>
      <td>2.9940</td><td>1.5960</td>
      <td>184.6147</td><td>117.6321</td>
      <td>755.4991</td><td>433.1389</td>
      </tr>

      <tr>
      <th>x86-64</th>
      <td>2.1233</td><td>2.3041</td>
      <td>128.5715</td><td>136.3224</td>
      <td>526.4390</td><td>557.5431</td>
      </tr>
      <tr>
      <th>SIMD with XMM</th>
      <td>1.5424</td><td>1.4825</td>
      <td>95.7449</td><td>103.9433</td>
      <td>358.0829</td><td>455.4245</td>
      </tr>
      <tr>
      <th>SIMD with YMM</th>
      <td>1.3859</td><td>1.5112</td>
      <td>98.7861</td><td>102.8042</td>
      <td>351.4904</td><td>454.2371</td>
      </tr>
      </tbody>
    </table>
-->

## 7.3. Correctness Check (C, CUDA)
<!--
1. **Correctness Check with size 2<sup>20</sup>**

- ![Correctness at 2^20](images/2_20_correctness.png)

2. **Correctness Check with size 2<sup>26</sup>**

- ![Correctness at 2^26](images/2_26_correctness.png)

3. **Correctness Check with size 2<sup>28</sup>**

- ![Correctness at 2^28](images/2_28_correctness.png)

4. **Correctness Check for SIMD with XMM and Boundary Checks**
   - Size: 1027
     - ![xmm correctness at 1027](images/xmm_check_1027.png)
   - Size: 32773
     - ![xmm correctness at 32773](images/xmm_check_32773.png)
   - Size: 131079
     - ![xmm correctness at 131079](images/xmm_check_131079.png)
5. **Correctness Check for SIMD with YMM and Boundary Checks**
   - Size: 1027
     - ![ymm correctness at 1027](images/ymm_check_1027.png)
   - Size: 32773
     - ![ymm correctness at 32773](images/ymm_check_32773.png)
   - Size: 131079
     - ![ymm correctness at 131079](images/ymm_check_131079.png)
-->

## 8. Discussion
### 8.1. Analysis of Results <!--TODO-->
<!-- 
    Across all input sizes, the Release mode of the C program performed faster than the Debug mode by around 56 to 87%. On the other hand, the Release mode of the programs with assembly languages (regardless of use of registers) is typically slower or about the same as the Debug mode. Furthermore, the Debug mode of the C program is consistently the slowest among all programs. As opposed to the Release mode which has more inconsistencies. This may be because the C compiler does more checks during debugging. As such the following observations will be based on the Release mode.

    For all input sizes, the programs calling the kernels of the x86-64 assembly languge (without XMM or YMM registers) is the slowest which may be due to function call overheads dominating the short workload when comparing to the C version. The x86-64 SIMD versions are also faster compared to the x86-64 only version because of the parallelism provided by the XMM and YMM registers which reduces loop iterations.

    For smaller inputs (2<sup>20</sup>), SIMD versions (XMM/YMM) provide marginal speedups at 6-8% over C. For the medium input (2<sup>26</sup>), SIMD versions show clearer advantages (13-14%). For larger inputs (2<sup>28</sup>) on the other hand the SIMD versions performed slower to C by around 4%. While the SIMD versions should be faster than the C versions and the x86-64 version only due to its parallel nature, it may be slower than the C program for the largest input possibly due to the device's memory capacity.

    Overall, the results show that SIMD implementations generally outperform standard x86-64 assembly and slightly improve upon C for small to medium inputs. However, the advantage decreases at larger inputs, likely due to the device’s memory capacity.
-->

### 8.2. Problems Encountered and Solutions Made
1. Factorial Explosion
    - Exact permutation tests become infeasible due to:
        - $O(N!)$ memory scaling  
        - GPU thread count limits  
        - Extremely large permutation counts  
        - Practical limit: ***$N\leq15$***

2. Solution: Monte Carlo Approximation
    - Monte Carlo sampling enables:
        - Large datasets  
        - Adjustable accuracy  
        - Efficient GPU execution  

    - MAE is computed as:

$$
MAE = \frac{1}{K} \sum_{i=1}^K \left| \hat{p}_i - p_{\text{exact}, i} \right|
$$

3. Random Permutation Quality
    - Accurate permutation tests require unbiased sampling.  
Achieved through:
        - Deterministic LCG pseudo-random sequence  
        - Uniform Fisher–Yates shuffling

### 8.3. Uniqueness

   - Computing for the exact permutation test with Monte Carlo as most research is only done using Monte Carlo
   - Using ANOVA as the test statistic for the permutation test

### 8.4. Realizations
   - Trying bold approaches such as trying to get exact permutation results may lead to roadblocks, particularly with resource limitations. But, it is okay to use other well-known approaches like the Monte Carlo model and just simply compare their capabilities