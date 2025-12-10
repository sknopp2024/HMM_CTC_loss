# Interpreting the Connectionist Temporal Classification Loss as the Likelihood of a Hidden Markov Model and Benchmarking Parallel Implementations that Use TensorFlow

This repository contains the Python code developed for the thesis.  
The work is organized into the following directories:

- `benchmark_HMMs` – Implementation and evaluation of classical Hidden Markov Models (HMMs) used in genome annotation. Different computational strategies are compared to optimize runtime and GPU resource usage.
- `pHMM` - Parallelized implementation of Profile Hidden Markov Models (pHMMs).
- `CTC_HMM` – Development of a method for efficient computation of the Connectionist Temporal Classification (CTC) loss based on a probabilistic model. 

Each directory contains the code and results related to a specific part of the thesis.


### Documentation (`/docs`)

The `/docs` directory contains supplementary materials for this repository:

-  abstract.
-  additional mathematical background and implementation-related information for each directory of the repository.

These documents provide deeper insight into the theoretical and technical foundations of the project.


### Requirements

This project requires Python 3.10.12 and the packages listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
