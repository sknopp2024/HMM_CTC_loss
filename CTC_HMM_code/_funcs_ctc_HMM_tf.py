#Sara Knopp, last modified on 29.07.2025

# Utility functions for CTC_HMMs

import tensorflow as tf


def check_min_num_frames(labels, batchsize, num_frames):  
    """
    Calculate the minimum number of frames required for the labels and 
    check if the given number of frames num_frames is sufficient.

    Args:
        labels (tensor): Input sequences, where each label is an integer
                         (batchsize x label_length), dtype=int64.
        batchsize (int): The number of batches.
        num_frames (tensor/int): Number of frames in the input sequence, dtpe = int32.

    Returns:
        float: Warning if num_frames is insufficient.
                Otherwise, returns None.
    """
    # Compare neighbors: equal runs
    equal_neighbors = tf.equal(labels[:, 1:], labels[:, :-1])  # (B, L-1)
    repeats = tf.reduce_sum(tf.cast(equal_neighbors, tf.int32), axis=1)  # (B)
    
    label_lengths = tf.shape(labels)[1] 
    min_frames_needed_per_batch = repeats + label_lengths  # (B)

    # Get the smallest required number over the batch
    min_required = tf.reduce_min(min_frames_needed_per_batch)  # scalar

    def warn_and_return():
        tf.print("Warning: The number of frames is too small to complete the given sequence.")
        return 1.0

    return tf.cond(num_frames < min_required, warn_and_return, lambda: tf.constant(0.0))


def _label_expansion(labels, blank):  
    """
    Expands a label sequence by inserting a blank token between each label 
    and at the beginning and end of the sequence for each sequence in the batch.

    Args:
        labels (tensor): Input sequences, where each label is an integer
                         (batchsize x label_length), dtype=int64.
        blank (int): The index representing the blank token in the set of labels.

    Returns:
        tf.Tensor: Expanded label sequence of shape (batchsize x z),
        dtype=int64. z = 2*label_length + 1
    
    Notes:
        - label [A, C] -> expanded label [blank, A, blank, C, blank]
    """
    def expand_single_sequence(label):
        label_length = tf.shape(label)[0]
        z = 2 * label_length + 1   # Length of expanded label

        expanded_label = tf.TensorArray(dtype=label.dtype, size=z)
        expanded_label = expanded_label.write(0, tf.cast(blank, label.dtype))  # Start with blank

        def cond(i, expanded_label):
            return i < label_length

        def body(i, expanded_label):
            expanded_label = expanded_label.write(2 * i + 1, label[i])
            expanded_label = expanded_label.write(2 * i + 2, tf.cast(blank, label.dtype))
            return i + 1, expanded_label

        _, expanded_label_final = tf.while_loop(cond, body, [0, expanded_label])
        return expanded_label_final.stack()
    
    return tf.map_fn(
        expand_single_sequence,
        labels,
        fn_output_signature=tf.TensorSpec(shape=(None,), dtype=labels.dtype)
    )


def _create_initial_distribution(z):  
    """
    Creates initial distribution pi of length z.

    Args:
        z (int): length of expanded_label, z = 2*label_length+1

    Returns:
        pi (tensor): initial distribution pi = (1, 1, 0, ..., 0) of length z,
                     dtype=tf.float32
    """
    pi = tf.tensor_scatter_nd_update(
        tf.zeros([z], dtype=tf.float32),
        indices=tf.constant([[0], [1]], dtype=tf.int32),  # positions to change
        updates=tf.constant([1.0, 1.0], dtype=tf.float32)  # set to one
    )
    return pi


def _create_UB(expanded_labels, logits):  
    """
    Creates the UB matrix containing the logits for the expanded label sequence.

    Args:
        expanded_labels (tensor): Expanded label sequence 
                                 (batchsize x z).
        logits (tensor): Logits from the neural network 
                        (batchsize x num_labels x num_frames), dtype=float32.
    Returns:
        tensor: UB matrix containing the logits for the expanded label sequence, 
                (batchsize x num_frames x z), dtype=float32.
                
    Notes:
        - The softmax of the logits is computed before gathering the logits for 
          the labels in expanded_labels.
        - The UB matrix is constructed by selecting columns from the softmaxed 
          logits based on the indices in the expanded label sequence.
    """
    UB_norm = tf.nn.softmax(logits, axis=-1)  # Shape: (batchsize x T x num_labels)

    # Get shape values
    num_frames = tf.shape(logits)[1]

    # Expand to (batch_size, 1, z) → then tile to (batch_size, num_frames, z)
    expanded_labels_tiled = tf.tile(tf.expand_dims(expanded_labels, axis=1), [1, num_frames, 1])

    # Use tf.gather with batch_dims=2 to get the right logits
    UB = tf.gather(UB_norm, expanded_labels_tiled, axis=2, batch_dims=2)  # Shape: (batch_size, num_frames, z)

    return UB


def _create_UB_slice(logits_t, expanded_labels):
    """
    Generates the t-th row of the UB matrix, containing the logits for the 
    expanded label sequence at a specific time step.
    This way system memory is saved.

    Args:
        logits_t (tensor): Logits from the neural network at time step t, 
                            with shape (num_labels x 1).
        expanded_labels (tensor): Expanded label sequence 
                                  (batchsize x 1 x 2*label_length+1).

    Returns:
        tensor: A tensor of shape (batchsize x 1 x z) containing the 
                selected logits corresponding to the expanded label sequence 
                at time step t.
    
    Notes:
        - The UB matrix is constructed by selecting columns from the softmaxed 
          logits based on the indices in the expanded label sequence.
        - z: length of the expanded label sequence.
    """
    # Gather the softmaxed logits corresponding to each expanded label for each frame
    UB_slice = tf.gather(logits_t, expanded_labels, axis=1, batch_dims=1)  #(batch_size x z)
    
    return UB_slice


def _create_A(expanded_labels, blank):  
    """
    Creates the transition matrix A for the expanded label sequence.

    Args:
        expanded_labels (tensor): Expanded label sequence (batchsize x 1 x z)
        blank (int): The index representing the blank token in the set of labels.

    Returns:
        tf.Tensor: Transition matrix A (batch_size, z, z), with allowed transitions 
                   marked as 1, dtype=float32.
    Notes:
        - z = 2*label_length+1, length of extended label.
        - The matrix A is built based on the label sequence and the blank token:
          - If the current label is the blank or the same as the label two positions earlier, 
            the matrix reflects transitions between the same and previous state.
          - Otherwise, it includes transitions between the current, previous, 
            and two positions earlier states.
            
    """
    batch_size = tf.shape(expanded_labels)[0]
    z = tf.shape(expanded_labels)[1]

    A = tf.zeros((batch_size, z, z), dtype=tf.float32)
    
    # Step 1: Set up transitions A[b, 0, 0] = A[b, 1, 0] = A[b, 1, 1] = 1
    positions = tf.constant([[0, 0], [1, 1], [1, 0]], dtype=tf.int32)  # (3, 2)

    # Positions for all batches
    positions = tf.tile(tf.expand_dims(positions, 0), [batch_size, 1, 1])  # (batchsize, 3, 2)
    
    # Indices of the batches
    batch_indices = tf.range(batch_size, dtype=tf.int32)
    batch_indices = tf.reshape(batch_indices, (-1, 1, 1))  # (batchsize, 1, 1)
    
    # combine batch indices with positions
    indices = tf.concat([tf.tile(batch_indices, [1, 3, 1]), positions], axis=2)  # (batch, 3, 3)
    indices = tf.reshape(indices, (-1, 3))  # (3 * batch, 3)
    
    # values of positions that are supposed to be updated
    values = tf.ones((tf.shape(indices)[0],), dtype=tf.float32)
    
    # Wende die Updates an
    A = tf.tensor_scatter_nd_update(A, indices, values)
    
    # Step 2: make sequence 2, ..., z
    i_range = tf.range(2, z)
    i_tile = tf.tile(tf.reshape(i_range, (1, -1)), [batch_size, 1])  # (batch, z-2)

    # Gather the labels at positions i = 2 to z-1
    labels_i = tf.gather(expanded_labels, i_range, axis=1)  # (batch, z-2)
    # Gather the labels at positions i-2 (two time steps before each i)
    labels_i_m2 = tf.gather(expanded_labels, i_range - 2, axis=1)  # (batch, z-2)

    # mask_case1: (label == blank) or (label == label[i-2])
    case1 = tf.logical_or(
        tf.equal(labels_i, blank),
        tf.equal(labels_i, labels_i_m2)
    )

    # Batch indices for i dimension
    batch_idx = tf.tile(tf.reshape(batch_indices, (-1, 1)), [1, z - 2])  # (batch, z-2)

    # Indices for (i, i), (i, i-1), (i, i-2)
    idx_ii   = tf.stack([batch_idx, i_tile, i_tile], axis=-1)
    idx_im1  = tf.stack([batch_idx, i_tile, i_tile - 1], axis=-1)
    idx_im2  = tf.stack([batch_idx, i_tile, i_tile - 2], axis=-1)

    # Fill 1 on diagonal and left of diagonal A[i, i] = A[i-1, i] = 1 
    A = tf.tensor_scatter_nd_update(A, tf.reshape(idx_ii, (-1, 3)), tf.ones([batch_size * (z - 2)]))
    A = tf.tensor_scatter_nd_update(A, tf.reshape(idx_im1, (-1, 3)), tf.ones([batch_size * (z - 2)]))

    # Only add (i, i-2) for case2
    mask_case2 = tf.logical_not(case1)
    
    # Indices of positions that need to be set to one, (batch, row, col)
    idx_case2_masked = tf.boolean_mask(idx_im2, mask_case2)  
    A = tf.tensor_scatter_nd_update(A, idx_case2_masked, tf.ones([tf.shape(idx_case2_masked)[0]]))

    return A


def _build_sparse_transition_matrix(expanded_labels, batch_size, max_length):
    """
    Constructs a sparse transition mask that allows label-skipping transitions
    (i.e., s → s+2) only when the intermediate symbol is a blank and the labels
    at positions s and s+2 are different.

    This follows the standard CTC rule that disallows skip transitions between
    repeated labels (e.g., A - blank - A), to prevent collapsing multiple
    identical labels into one.

    Args:
        expanded_labels (tf.Tensor): Expanded label sequences with inserted blanks,
                                     shape (batch_size, max_length), dtype=int32.
        batch_size (int): Number of sequences in the batch.
        max_length (int): Length of the expanded label sequences (z).

    Returns:
        tf.Tensor: Sparse transition matrix A of shape (batch_size, max_length),
                   where each element contains:
                   - 0.0 (log(1.0)) if a skip transition is allowed,
                   - -1e6 (≈ log(0)) otherwise.
    Notes:
        - The first two positions (s=0,1) are padded with log(1.0) = 0.0 by default,
          although they are not used for skip transitions.
        - example: label = [blank, A, blank, A, blank, C, blank]
          A = [0.0, 0.0, -1e6, -1e6, -1e6, 0.0, -1e6]
    """
    comparison = tf.not_equal(expanded_labels[:, :-2], expanded_labels[:, 2:])
    A = tf.where(comparison, tf.fill([batch_size, max_length - 2], tf.math.log(1.0)), -1e6)
    padding = tf.fill([batch_size, 2], 0.0)
    A = tf.concat([padding, A], axis=1) 
    return A
