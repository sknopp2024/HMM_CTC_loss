# Thesis: Interpreting the Connectionist Temporal Classification Loss as the Likelihood of a Hidden Markov Model and Benchmarking Parallel Implementations that Use TensorFlow

This repository contains the Python code developed for the thesis.  
The work is organized into the following directories:

- `benchmark_HMM_code`
- `pHMM_code`
- `CTC_HMM_code`
- `stability_investigations_code`

Each directory contains the code related to a specific part of the thesis.

### File naming conventions

- Files starting with `_` contain helper **functions** or **modules**.
- Files starting with `hpc_` are intended to be run on an **HPC cluster**.


<br><br>


## CTC Loss Computation using CRF

Several implementations of the Conditional Random Fields (CRFs) are compared for computing the Connectionist Temporal Classification (CTC) loss.  
The performance is benchmarked against the TensorFlow baseline `tf.nn.ctc_loss()` (referred to as **TensorFlow CTC Loss Function**), which serves as a reference.

All variants implement the same core logic based on the recurrence equation for CRFs with CTC alignment.  
They differ in how the forward matrix **F** is computed — specifically in their level of parallelism and use of TensorFlow's capabilities.

---

#### 1. Nested For Loops

This is the most basic implementation. The forward matrix **F** is computed step-by-step over time steps **t** and label positions **s**.  
Python `for` loops are used for both dimensions.  
This variant is referred to as **Nested For Loops**.

---

#### 2. TensorFlow Scan

Here, the explicit loops from **Nested For Loops** are replaced by a **`tf.scan()`** operation, enabling better performance on GPUs.  
The computation is embedded in the TensorFlow graph, allowing for improved efficiency via GPU acceleration and reduced overhead.

---

#### 3. Vectorised For Loops

This version vectorises the loop over the label positions **s**, so only a single loop over time steps **t** remains.  
Each row **Fₜ** of the forward matrix is computed as a whole in one step.  
This improves computational efficiency and simplifies the implementation.

---

#### 4. Vectorised TensorFlow Scan

This version combines the vectorisation from **Vectorised For Loops** with the TensorFlow `tf.scan()` operation from **TensorFlow Scan**.  
The goal is to achieve both high performance and GPU efficiency using fully graph-based execution.

---

#### Numerical Stability

As in other probabilistic models, the multiplication of many probabilities can cause numerical underflow.  
All implementations operate in **log-space** and apply the **log-sum-exp trick** to ensure numerical stability throughout the forward computation.




<br> <br>

## Benchmarking Hidden Markov Models 

Sequential and parallel implementations of the forward algorithm were analyzed and compared in terms of performance.

### Sequential Implementations of the Forward Algorithm

This section describes two sequential implementations of the forward algorithm and their mathematical formulations.


####  1. Nested For Loops

In this implementation, the values αₜ(q) (see recurrence formula) are computed sequentially.  
The **sparsity** of the transition matrix **A** is exploited: summands with **aᵢq = 0** are skipped.  
The Python implementation ([source](https://www.python.org)) uses multiple `for` loops, including iterations over **t** (time steps) and **q** (states).  
Therefore, this version is referred to as **Nested For Loops**.

---

#### 2. Matrix Multiplication 
This is a vectorized implementation of the forward algorithm. Let the one-hot encoded input sequence be **û ∈ {0, 1}^{L × |Σ|}**, and **Bᵀ** the transposed emission matrix. Then define:

**E = û × Bᵀ**

where E ∈ [0, 1]^{L × K} contains emission probabilities:  
**Eₜq = P(Uₜ = uₜ | Qₜ = s_q)**

The forward probabilities α are computed as:

- **α₁ = π × E₁**  
- **αₜ = (αₜ₋₁ × A) ∘ Eₜ**, for t = 2,…,L

Here, × is matrix multiplication and ∘ is element-wise multiplication.

This implementation is referred to as **Matrix Multiplication** and is mathematically equivalent to the nested loop version.


---

### Parallel Implementations of the Forward Algorithm

#### 1. Associative Scan

In this version, the forward probability matrix **α** is computed using a **parallel scan procedure**, specifically the *all-prefix-sums operation* (also known as **associative scan**). This enables computations to be performed in parallel across time steps **t**, which improves runtime performance.

**Definition – All-Prefix-Sums Operation:**  
Given a sequence **n = (n₁, n₂, ..., nₘ)** and an associative binary operator **∘**, the all-prefix-sums operation produces:

> **(n₁, n₁ ∘ n₂, n₁ ∘ n₂ ∘ n₃, ..., n₁ ∘ n₂ ∘ ... ∘ nₘ)**

(see Sarkka, Blelloch, 1990)

---

#### 2. Associative Scan STM

This variant is based on the **Associative Scan** described above, but additionally leverages the **sparsity of the transition matrix A**.

Only those summands in the matrix definitions **Mₜᵢq = aᵢq × b_{q, uₜ}** and **M₁ᵢq = πᵢ × b_{q, u₁}** are considered for which **aᵢq ≠ 0**.  
To achieve this, the implementation uses a compact representation of **A**, storing only the non-zero entries via `tf.sparse.SparseTensor`.

By skipping zero entries, the number of operations is significantly reduced.  
As a result, this approach can reduce both **runtime** and **GPU memory consumption**, especially when **A** is highly sparse.

This version is referred to as **Associative Scan STM (Sparse Transition Matrix)**.


---


### Definition of HMMs

A *Hidden Markov Model* consists of:

- an alphabet Σ = {v₁, ..., v₍|Σ|₎}
- a set of hidden states 𝒮 = {s₁, s₂, ..., sₖ}
- an observation sequence U = (U₁, U₂, ..., Uₗ)
- a hidden state sequence Q = (Q₁, Q₂, ..., Qₗ) with each Qₜ ∈ 𝒮, for t = 1, ..., L

The model includes:

- a transition matrix A ∈ ℝᵏˣᵏ, where:  
  **aᵢⱼ = P(Qₜ = sⱼ | Qₜ₋₁ = sᵢ)**, ∀ i,j = 1,...,K

- an emission matrix B ∈ ℝᵏˣ|Σ|, where:  
  **bᵢⱼ = P(Uₜ = vⱼ | Qₜ = sᵢ)**, ∀ i = 1,...,K and j = 1,...,|Σ|

- an initial distribution π ∈ ℝ¹ˣᵏ, where:  
  **πᵢ = P(Q₁ = sᵢ)** for i = 1,...,K

Additionally, HMMs satisfy the *first-order Markov property*:

> P(Qₜ = sⱼ | Qₜ₋₁ = sⱼ, Qₜ₋₂ = sₖ, ...) = P(Qₜ = sⱼ | Qₜ₋₁ = sⱼ)  
> for all sⱼ, sᵢ, sₖ ∈ 𝒮

This means that each state only depends on the previous one.

**Note:** Matrices A and B are stochastic, i.e., each row sums to 1 and all entries are non-negative.

### The Forward Algorithm

The *forward algorithm* recursively computes the forward probabilities **α** for an observation sequence **u**, given an HMM with parameters λ = (A, B, π).  
It is defined as:

**Forward variable:**

> αₜ(q) = P(U₁ = u₁, ..., Uₜ = uₜ, Qₜ = s_q | λ)  
> for t = 1,...,L and q = 1,...,K

This gives the probability of observing the first t symbols and being in state s_q at time t.

**Initialization:**

> α₁(q) = π_q × b_{q, u₁}  
> where b_{q, u₁} is the emission probability of u₁ in state s_q

**Recursion step:**

> αₜ₊₁(q) = (∑ᵢ=1ᵏ αₜ(i) × aᵢq) × b_{q, uₜ₊₁}  
> for t = 1,...,L−1 and q = 1,...,K

**Final probability of the observation sequence:**

> P(u | λ) = ∑ᵢ=1ᵏ αₗ(i)

**Runtime complexity:**  
The forward algorithm has a runtime of **𝒪(K² × L)**.

 *For further details, see Rabiner (1989).*



