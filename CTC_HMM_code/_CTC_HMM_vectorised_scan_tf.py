#Sara Knopp, last modified on 30.07.2025

# Improve vectorised Version of CTC_HMM with scan by writing a function for
# calculating UB stepwise to save memory

import tensorflow as tf

# Uncomment and adjust the following lines if you want to run this script standalone
# (i.e., not imported by another script):
# Change the working directory to the project directory
#import os
#homefolder = "..."
#project_dir = os.path.join(homefolder, "Programs", "CTC_HMM_code")
#os.chdir(project_dir)

# Import utility functions
from _funcs_ctc_HMM_tf import (
    _build_sparse_transition_matrix,
    _create_UB_slice,
    _create_initial_distribution,
    check_min_num_frames,
    _label_expansion,
)

#%%
@tf.function(reduce_retracing=True)
def CTC_HMM_vectorised_scan(labels, logits, num_frames, blank):
    """
    Compute the CTC (Connectionist Temporal Classification) loss.

    Args:
        labels (tensor): Input sequences, where each label is an integer
                         (batchsize x label_length).
        logits (tensor): Logits from the neural network 
                         (batchsize x num_labels x num_frames). 
        num_frames (int): Number of frames in the input sequence.
        blank (int): The index representing the blank token in the set of labels.

    Returns:
        tensor: The CTC loss for each batch.

    Notes:
        - num_labels: Number of possible labels (including the blank token)
    """
    batchsize = tf.shape(labels)[0]

    # Ensure that there are enough frames to cover the label sequences
    penalty = check_min_num_frames(labels, batchsize, num_frames)
    #if num_frames too small -> penalty > 0
    tf.debugging.assert_less_equal(penalty, 0.0, 
    message = "The number of frames is too small to complete the given label sequence.")

    # Expand the labels to include the blank token
    expanded_labels = _label_expansion(labels, blank) #(batchsize x 1 x z)
    z = len(expanded_labels[0])

    # Initial distribution (1 x z), pi = (1, 1, 0, ..., 0)
    pi = _create_initial_distribution(z)
    pi_log = tf.where(pi > 0, tf.math.log(pi), -1e6)
    
    # Normalise logits with softmax (batchsize x num_frames x label_length)
    logits_log = tf.nn.log_softmax(logits)

    # Create a aparse transition matrix A (batchsize x z)
    A = _build_sparse_transition_matrix(expanded_labels, batchsize, z)
    
    # Calculate alpha
    subdiagonal_1_padding = tf.fill((batchsize, 1), -1e6)
    subdiagonal_2_padding = tf.fill((batchsize, 2), -1e6)
    
    # Define step function for tf.scan 
    def step_fn(F_prev, t):
        # Contributions from subdiagonal 1. Pad the first entry of F_prev.
        subdiagonal_1 = tf.concat([subdiagonal_1_padding, F_prev[:, :-1]], axis=1)
        
        # Contributions from subdiagonal 2
        subdiagonal_2 = tf.concat([subdiagonal_2_padding, F_prev[:, :-2] + A[:, 2:]], axis=1)
        
        # Combine contributions in log-space
        F_current = tf.reduce_logsumexp(
            [F_prev, subdiagonal_1, subdiagonal_2],  # F_t-1 + A
            axis=0  # Combine over the 3 contributions
        ) + _create_UB_slice(logits_log[:, t], expanded_labels)  # Add UB for the current time step

        return F_current
    
    # Initial F value (log-space)
    F_init = pi_log + _create_UB_slice(logits_log[:, 0], expanded_labels)  # Shape: (batch_size, max_length)
    
    # Use tf.scan to compute alpha over time
    time_steps = tf.range(1, num_frames)  # Time steps: [1, 2, ..., num_frames - 1]
    alpha = tf.scan(
        fn=step_fn,
        elems=time_steps,  # Iterate over time indices
        initializer=F_init
    )
    
    # Concatenate the initial step and the scanned results
    F = tf.concat([tf.expand_dims(F_init, axis=0), alpha], axis=0)  # (max_time, batch_size, max_length)
    
    # Transpose to desired shape (batch_size, max_length, max_time)
    F = tf.transpose(F, perm=[1, 2, 0])

    # Compute the CTC loss for each batch
    ctc_loss = -tf.math.reduce_logsumexp(tf.stack([F[:, z - 1, num_frames - 1], 
                                                   F[:, z - 2, num_frames - 1]], axis=0), axis=0)

    return ctc_loss  

#%%
#small example
'''
# Set random seed for reproducibility
tf.random.set_seed(28102024) 

H = 4  # Alphabet size (number of possible nucleotides: A, C, G, T)
blank = 0  # Blank index
num_labels = 5  # Number of labels [A, C, G, T, e]
batchsize = 4  # Batch size
num_frames = 50  # Number of Frames
label_length = 20  # Length of input sequences

# Create a list of input sequences of different length
labels = tf.random.uniform([batchsize, label_length],
                           minval = 1, maxval = H+1, dtype = tf.int32)
# Create list of logits
logits_single = tf.random.uniform([num_frames, num_labels])
logits = tf.tile(tf.expand_dims(logits_single, axis=0), [batchsize, 1, 1])

print(CTC_HMM_vectorised_scan(labels, logits, num_frames, blank))
#tf.Tensor([39.25209  37.818047 40.20242  39.07262 ])

#%%    
# Calculate gradient of logits and num_frames 

with tf.GradientTape() as tape:
    tape.watch(logits)  # Watch logits, if not gradient is None

    loss = CTC_HMM_vectorised_scan(labels, logits, num_frames, blank)

# Calculate gradient
grads = tape.gradient(loss, logits)
print(grads)
'''