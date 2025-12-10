#Sara Knopp, last modified on 07.08.2025

# CTC_HMM not vectorised version, scan

import tensorflow as tf

# Uncomment and adjust the following lines if you want to run this script standalone
# (i.e., not imported by another script):
# Change the working directory to the project directory
#import os
#homefolder = "..."
#project_dir = os.path.join(homefolder, "Programs", "CTC_HMM_code")
#os.chdir(project_dir)

from _funcs_ctc_HMM_tf import (
    _create_A,
    _create_UB,
    _create_initial_distribution,
    check_min_num_frames,
    _label_expansion,
)

#%%

@tf.function(reduce_retracing=True)
def calculate_F_t(F_t_1, t, z, A, UB, thresh_list_t):
    """
    Computes the F[t, s] values for all states s at time t in the log-space.

    Args:
        F_t_1 (tensor): The previously computed F values at time t-1 (1D tensor of shape (z,))
        t (int): The current time step
        z (int): The number of states
        A (tensor): The transition matrix for the states (tensor of shape (z,))
        UB (tensor): The logits matrix for each frame (tensor of shape (num_frames, z))
        thresh_list_t (list): A list of thresholds for each state s. 
                              If t < thresh_list_t[s], the computation is skipped.

    Returns:
        tensor: A tensor of the computed F[t, s] values for all states s at time t.
    """
    # Calculate log-sum-exp for F_t for all s
    def compute_s(s):
        if t < thresh_list_t[s]:
            return F_t_1[:, s]  # No calculations done
            
        # Calculate log(Sum(F_t_1 * A[s]) * UB[t, s])
        # Apply the log-sum-exp trick
        return tf.reduce_logsumexp(F_t_1 + tf.cast(A[:, s], dtype = tf.float32), axis=1) + UB[:, t, s]

    # Calculate F_t for all s
    F_t_values = tf.transpose(tf.map_fn(lambda s: compute_s(s), tf.range(z),
        fn_output_signature=tf.float32))
    
    return F_t_values

@tf.function
def compute_thresh_list_t(z, num_frames, input_length):
    """
    Computes a list of time step thresholds for each label position.

    These thresholds define the earliest possible time step `t` 
    at which a non-zero value can be assigned to the corresponding 
    label position in the forward variable matrix F.

    Args:
        z (int): Number of label positions.
        num_frames (int): Total number of time steps (frames).
        input_length (tf.Tensor): Length of the original input sequence.

    Returns:
        tf.Tensor: A 1D tensor of shape (z,) containing threshold values 
                   for each label position.

    Notes:
        - Time steps `t` below the computed threshold will be masked out (F = 0).
    """
    thresh_list_t = tf.TensorArray(dtype=tf.int32, size=z)
    
    input_length = tf.cast(input_length, tf.float64)
    
    for s in tf.range(z):
        # Compute the raw threshold for label position s
        value = tf.math.floor(
            tf.cast(input_length + (s - z - 1) / 2, tf.float32)
        )
        value = tf.cast(value, tf.int32)

        # t must be minimum 1, since the first column of F is already defined 
        value_clipped = tf.maximum(value, 1)

        thresh_list_t = thresh_list_t.write(s, value_clipped)

    return thresh_list_t.stack()

@tf.function(reduce_retracing=True)
def CTC_HMM_sequential_scan(labels, logits, num_frames, blank):
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
        - labels must not contain blanks
    """
    batchsize, input_length = tf.shape(labels)[0], tf.shape(labels)[1]

    # Ensure that there are enough frames to cover the label sequences
    penalty = check_min_num_frames(labels, batchsize, num_frames)
    #if num_frames too small -> penalty > 0
    tf.debugging.assert_less_equal(penalty, 0.0, 
    message = "The number of frames is too small to complete the given label sequence.")
    
    # Expand the labels to include the blank token
    expanded_labels = _label_expansion(labels, blank)  # (batchsize x 1 x z)
    z = len(expanded_labels[0])
    
    # Initial distribution (1 x z), pi = (1, 1, 0, ..., 0)
    pi = _create_initial_distribution(z)
    pi_log = tf.where(pi > 0, tf.math.log(pi), -1e6)
    
    # Construct the UB (logits matrix for each frame) (batchsize x num_frames x z)
    UB = _create_UB(expanded_labels, logits)
    UB_log = tf.where(UB > 0, tf.math.log(UB), -1e6)   
    
    # Create the transition matrix A (batchsize x z x z)
    A = _create_A(expanded_labels, blank)
    A_log = tf.where(A > 0, tf.math.log(A), -1e6)
    
    # Determine the threshold for t, t column index of F
    # For t lower than this thresholds: F = 0
    thresh_list_t = compute_thresh_list_t(z, num_frames, input_length)
    
    # Initialize the first column of F
    init = pi_log + UB_log[:, 0]
    
    # Perform the forward pass through the frames
    F_scan = tf.scan(lambda F_t_1, t: calculate_F_t(F_t_1, t, z, A_log, UB_log, 
                                                    thresh_list_t),
                tf.range(1, num_frames), initializer = init)
    
    # Add the first column back to the complete F matrix
    F_complete = tf.concat([init[tf.newaxis, :], F_scan], axis=0)

    # Update F[b, 1, :] with the calculated forward pass results
    F = tf.transpose(F_complete, perm=[1, 2, 0])  # Update the entire slice of F  
    
    # Compute the CTC loss for each batch
    ctc_loss = -tf.math.reduce_logsumexp(tf.stack([F[:, z - 1, num_frames - 1], 
                                                   F[:, z - 2, num_frames - 1]], axis=0), axis=0)
    return ctc_loss  

#%%
#small example data
'''
# Set random seed for reproducibility
tf.random.set_seed(28102024)

H = 4  # Alphabet size (number of possible nucleotides: A, C, G, T)
blank = 0
num_labels = 5  # Number of labels [A, C, G, T, e]
batchsize = 4  # Batch size
num_frames = 50  
label_length = 20  # Length of input sequences

# Create a list of input sequences of different length
labels = tf.random.uniform([batchsize, label_length],
                           minval = 1, maxval = H+1, dtype = tf.int32)
# Create list of logits
logits_single = tf.random.uniform([num_frames, num_labels])
logits = tf.tile(tf.expand_dims(logits_single, axis=0), [batchsize, 1, 1])

print(CTC_HMM_sequential_scan(labels, logits, num_frames, blank))
#[39.252087 37.818047 40.20242  39.072617]

#%%    
# Calculate gradient of logits and num_frames 

with tf.GradientTape() as tape:
    tape.watch(logits)  # Watch logits, if not gradient is None
    
    loss = CTC_HMM_sequential_scan(labels, logits, num_frames, blank)

# Calculate gradient
grads = tape.gradient(loss, logits)
print(grads)
'''