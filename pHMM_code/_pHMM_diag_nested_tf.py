#Sara Knopp, last modified on 11.08.2025

# Compute entries of the forward probability matrix of a pHMM diagonally using 
# for loops

import tensorflow as tf

# Uncomment and adjust the following lines if you want to run this script standalone
# (i.e., not imported by another script):
# Change the working directory to the project directory
#import os
#homefolder = "..."
#project_dir = os.path.join(homefolder, "Programs", "pHMM_code")
#os.chdir(project_dir)

from _funcs_pHMM_tf import _create_distance_array
from _pHMM_diag_vectorised_tf import get_diagonal_indices

#%%
@tf.function(reduce_retracing = True)
def pHMM_diag_nested_for(A, B, pi, u, U):
    """
    This function computes the forward probabilities (alpha) in log space for a 
    profile Hidden Markov Model (pHMM). The entries of alpha are calculated
    diagonally.
    
    Args:
        A (tf.Tensor): Transition matrix (K, K), dtype=tf.float32
        B (tf.Tensor): Emission matrix (K, H), dtype=tf.float32
        pi (tf.Tensor): Initial state distribution (batchsize, K), dtype=tf.float32
        u (tf.Tensor): Input sequence indices (batchsize, input_length), 
                       dtype=tf.int32.
        U (tf.Tensor): One-hot encoded input sequence (batchsize, input_length, H), 
                       dtype=tf.float32.
        
    Returns:
        log_alpha (tf.Tensor): Log forward probability matrix 
                               (batchsize, K, input_length), dtype=tf.float32.
    Notes:
        - H: Alphabet size
        - L: Input length
        - K: Number of states, K = 3 * input_length + 1
        - length of a diagonal: input_length - 1
    """
    input_length = tf.shape(u)[1]  # Length of input sequence
    K = 3*input_length+1  # Number of states 
    batchsize = tf.shape(u)[0]  # Batch size
    diag_length = input_length - 1  # Length of a diagonal of alpha 

    # Masks all non-zero transitions in A to identify allowed predecessor states.
    pred_mask = tf.not_equal(A, 0)
    
    # Create duration array (K,), d = 0 for Delete states, otherwise d = 1
    d = _create_distance_array(input_length) 
    
    # Convert transition and emission matrix and initial distribution to log space
    epsilon = 1e-6
    A_log = tf.math.log(tf.maximum(A, epsilon))  # Keep A differentiable
    B_log = tf.where(B > 0, tf.math.log(B), -1e6)  
    pi_log = tf.where(pi > 0, tf.math.log(pi), -1e6) 

    # Precompute the emission probabilities for each time step 
    UB_log = tf.matmul(U, tf.transpose(B_log))  
    # (batch_size x L x H) @ (H x K) -> (batch_size x L x K)
    
    # Initialisation of forward prb matrix, (batchsize x K x L)
    log_alpha = tf.fill([batchsize, K, input_length], -1e6)

    # Compute Updates (broadcasted): shape [batchsize, K]
    updates = pi_log[None, :] + UB_log[:, 0, :]
    updates = tf.reshape(updates, [-1])
    
    # Create indices, shape (batchsize * K, 3)
    indices = tf.stack(tf.meshgrid(tf.range(batchsize),tf.range(K),
                                   tf.constant([0]), indexing='ij'), axis=-1)
    indices = tf.reshape(indices, [-1, 3])
    # Update log_alpha
    log_alpha = tf.tensor_scatter_nd_update(log_alpha, indices, updates)
    
    for g in tf.range(0, input_length + K-2):  # Loop over all diagonals

        if g < diag_length: # First diagonals do not have full length
            # Initialize current diagonal with -inf entries
            diagonal = tf.fill([batchsize, g+1], tf.float32.min)
        else:
            diagonal = tf.fill([batchsize, diag_length], tf.float32.min)
        
        for i in tf.range(0, diag_length):  # Loop over all entries of a diagonal
            q = g - i  # Row index
            t = i + 1  # Column index
            
            # Calculate only elements inside alpha
            if (q >= 0 and q < K and t < input_length):
                # Predecessors of state q
                pred_indices = tf.cast(tf.where(pred_mask[:, q])[:, 0], tf.int32)
                num_pred = tf.shape(pred_indices)[0]
                    
                # Create a temporary Variable to store interim results
                temp_values = tf.fill([batchsize, num_pred], tf.float32.min)
                
                for j in tf.range(num_pred):  # Loop over all predecessors of q
                    pred = pred_indices[j]  # Index of the current predecessor
                    
                    # Calculation for current predecessor
                    updates_temp = A_log[pred, q] + log_alpha[:, pred, int(t-d[q])]
                    scatter_indices_temp = tf.stack([tf.range(batchsize), tf.fill([batchsize], j)], axis=1)  # Indizes für das Update
                    temp_values = tf.tensor_scatter_nd_update(temp_values,
                                        scatter_indices_temp, updates_temp)
                  
                # Apply log-sum-exp trick: log(sum(exp(x - max(x))) + max(x))
                updates = tf.reduce_logsumexp(temp_values, axis = 1) + tf.gather(B_log[q], u[:, t])
                scatter_indices = tf.stack([tf.range(batchsize), tf.fill([batchsize], i)], axis=1)  # Indizes für das Update
                diagonal = tf.tensor_scatter_nd_update(diagonal,
                                    scatter_indices, updates)
                
        if g >= K:  # Last diagonals do not have full length either, remove not necessary entries
            diagonal = diagonal[:, (g-K+1):]
            
        # Get indices of diagonal elements for update of log_alpha
        diag_indices = get_diagonal_indices(start_row = g, diag_length = diag_length, 
                                            num_rows = K, batchsize = batchsize)
        
        # Update log_alpha with current diagonal
        log_alpha = tf.tensor_scatter_nd_update(log_alpha, diag_indices, 
                                                tf.reshape(diagonal, [-1]))
    return log_alpha


#%%
#small example
'''
from _funcs_pHMM_tf import (
    _create_transition_matrix, _create_emission_prb_matrix, 
    _create_initial_distribution)

tf.random.set_seed(28102024) 

batchsize = 2
H = 20  # Alphabet size
input_length = [3, 10, 50] # List of different sequence lengths

# Generate input sequences with batch dimension, elements of dim (batchsize x input_length)
input_batch_list = [
    tf.random.uniform(shape=(batchsize, L), maxval=H, dtype=tf.int32)
    for L in input_length]

# One-hot encode the input sequences, elements of dim (batchsize x input_length x H)
one_hot_batch_list = [tf.one_hot(u, H) for u in input_batch_list] 

i = 0

A = _create_transition_matrix(input_length[i])
B = _create_emission_prb_matrix(input_length[i], H)
pi = _create_initial_distribution(input_length[i])
u = input_batch_list[i]
U = one_hot_batch_list[i]

alpha_diag_for = pHMM_diag_nested_for(A, B, pi, u, U)
print(alpha_diag_for)
tf.argmax(alpha_diag_for, axis=1)

last_log_alpha = alpha_for_tf[:, :, -1]  # shape: (batchsize, K)
log_likelihood = tf.reduce_logsumexp(last_log_alpha, axis=1)  # shape: (batchsize,)
print(-log_likelihood)
#%%
# Apply the algorithm and compute gradient
with tf.GradientTape() as tape:
    tape.watch(A)
    tape.watch(B)
    tape.watch(pi)  # Watch pi, if not gradient is None
    
    alpha_log = pHMM_diag_nested_for(A, B, pi, u, U)
    final_log_prob = tf.reduce_logsumexp(alpha_log[:, :, -1], axis=1)  
    loss = -tf.reduce_mean(final_log_prob)  # Negative Log-Likelihood Loss

# Calculate gradient
grads = tape.gradient(loss, [A, B, pi])
print(grads)
'''
