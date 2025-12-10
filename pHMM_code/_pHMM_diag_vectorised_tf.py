#Sara Knopp, last modified on 12.08.2025

# Compute entries of the probability matrix of a pHMM diagonally, 
# each diagonal in one step (vecorised)

import tensorflow as tf

#%%

@tf.function(reduce_retracing = True)
def get_diag_position(pos_list, idx):
    """
    Selects a tensor from a list of tensors based on the given index.

    Args:
        pos_list (list of tf.Tensor): A list containing tensors representing 
             diagonal positions (e.g., for insert, match, and delete states).
        idx (tf.Tensor): A scalar tensor (int32) representing the index of the 
                  desired position in pos_list. Expected values are 0, 1, or 2.

    Returns:
        tf.Tensor: The tensor from pos_list corresponding to the given index.
    """
    return tf.switch_case(idx, branch_fns=[
        lambda: pos_list[0],
        lambda: pos_list[1],
        lambda: pos_list[2]
    ])

@tf.function(reduce_retracing = True)
def extract_transition_prbs(A, state, diag_pos, g):
    """
    Extract the relevant transition probabilities for a given state type 
    (match, insert, or delete) from the precomputed sparse transition matrix A. 

    Args:
        A (tf.Tensor): Sparse transition matrix for a specific state type
        state (str): One of 'match', 'insert', or 'delete'
        diag_pos (tf.Tensor): Tensor containing diagonal positions of the given 
                              state type in diagonal g
        g (int): Number of current diagonal in alpha

    Returns:
        tf.Tensor: Extracted transition probabilities for the corresponding states, 
                   shape (block_length, len(diag_pos)), dtype=tf.float32
    Notes:
        - block_length is 3 for match states, 2 for insert and delete states
    """
    # Determine in which row of alpha the respective states are
    state_rows = g - diag_pos
    
    # Determine from which block of A the probabilities should be taken
    block_index = tf.math.floordiv(state_rows, 3)
    
    # Length of a block, 3 for match, 2 for insert and delete
    block_length = 3 if state == 'match' else 2

    # Start and end row of the relevant blocks
    start_row = block_index * block_length
    end_row = start_row + block_length

    def slice_block(start, end):
        return tf.strided_slice(A, [start, 0], [end, tf.shape(A)[1]])

    # Extract relevant probabilities from A
    blocks = tf.map_fn(
        lambda x: slice_block(x[0], x[1]),
        (start_row, end_row),
        fn_output_signature=tf.TensorSpec(shape=(None, A.shape[1]), dtype=A.dtype)
    )
    # len(diag_pos) x block_length x 1
    
    blocks_squeezed = tf.squeeze(blocks, axis = -1) # len(diag_pos) x block_length

    result = tf.transpose(blocks_squeezed)
    
    return result

@tf.function(reduce_retracing = True)
def extract_log_alpha_insert(alpha, diag_pos, g):
    """
    Gather log-probabilities from the forward prb matrix alpha at specific 
    positions depending on the considered insert states on the diagonal number 
    g in alpha.

    Args:
        alpha (tf.Tensor): Forward probability matrix (alpha), (batch_size, K, L)
        diag_pos (tf.Tensor): Tensor containing diagonal positions of the insert 
                              states in diagonal g, (num_states,)
        g (int): Number of current diagonal in alpha

    Returns:
        tf.Tensor: Gathered log-probabilities for the specified insert states,
                   (batch_size, 2, num_states), dtype=tf.float32

    Notes:
        - To calculate a position [i, j] in alpha that corresponds to an insert 
          state the entries at [i, j-1] and [i-2, j-1] in alpha are needed
        - B: batch size, num_states: number of insert states in diagonal number g
    """
    batch_size = tf.shape(alpha)[0]  # Batch size
    num_states = tf.shape(diag_pos)[0]  # Number of insert states in diagonal number g

    # Row indices for the needed entries of alpha
    g_i_0 = tf.maximum(g - diag_pos, 0)
    g_i_1 = tf.maximum(g - diag_pos - 2, 0)
    row_indices = tf.stack([g_i_0, g_i_1], axis=1)  # (num_states, 2)

    # Insert batch dimension (B, num_states, 2)
    row_indices = tf.tile(row_indices[None, :, :], [batch_size, 1, 1])
    column_indices = tf.tile(diag_pos[None, :, None], [batch_size, 1, 2])
    batch_indices = tf.tile(tf.range(batch_size)[:, None, None], [1, num_states, 2])

    # Indices of needed entries, (B, num_states, 2, 3)
    gather_indices = tf.stack([batch_indices, row_indices, column_indices], axis=-1)
    
    # Gather entries of alpha
    gathered = tf.gather_nd(alpha, gather_indices)  # (B, num_states, 2)
    result = tf.transpose(gathered, perm=[0, 2, 1])
    
    return result

@tf.function(reduce_retracing = True)
def extract_log_alpha_match(log_alpha, diag_pos, g):
    """
    Gather log-probabilities from the forward prb matrix alpha at specific 
    positions depending on the considered match states on the diagonal number 
    g in alpha.

    Args:
        alpha (tf.Tensor): Forward probability matrix (alpha), (batch_size, K, L)
        diag_pos (tf.Tensor): Tensor containing diagonal positions of the match 
                              states in diagonal g, (num_states,)
        g (int): Number of current diagonal in alpha

    Returns:
        tf.Tensor: Gathered log-probabilities for the specified match states,
                   (batch_size, 2, num_states), dtype=tf.float32

    Notes:
        - To calculate a position [i, j] in alpha that corresponds to a match 
          state the entries at [i-1, j-1], [i-2, j-1] and [i-3, j-1] in alpha 
          are needed
        - B: batch size, num_states: number of match states in diagonal number g
    """
    batch_size = tf.shape(log_alpha)[0]  # Batch size
    num_states = tf.shape(diag_pos)[0]  # Number of match states in diagonal number g

    # Row indices for the needed entries of alpha
    # g - diag_pos is the row in alpha of the respective states
    g_i_0 = tf.maximum(g - diag_pos - 3, 0)  
    g_i_1 = tf.maximum(g - diag_pos - 2, 0)
    g_i_2 = tf.maximum(g - diag_pos - 1, 0)
    time_indices = tf.stack([g_i_0, g_i_1, g_i_2], axis=1)  # (num_states, 3)

    # Insert batch dimension (B, num_states, 3)
    time_indices = tf.tile(time_indices[None, :, :], [batch_size, 1, 1])  
    state_indices = tf.tile(diag_pos[None, :, None], [batch_size, 1, 3])  
    batch_indices = tf.tile(tf.range(batch_size)[:, None, None], [1, num_states, 3]) 

    # Indices of needed entries, (B, num_states, 3, 3)
    gather_indices = tf.stack([batch_indices, time_indices, state_indices], axis=-1)  

    # Gather entries of alpha
    gathered = tf.gather_nd(log_alpha, gather_indices)  # (B, num_states, 3)
    result = tf.transpose(gathered, perm=[0, 2, 1])
    
    return result

@tf.function(reduce_retracing = True)
def extract_log_alpha_delete(log_alpha, diag_pos, g):
    """
    Gather log-probabilities from the forward prb matrix alpha at specific 
    positions depending on the considered delete states on the diagonal number 
    g in alpha.

    Args:
        alpha (tf.Tensor): Forward probability matrix (alpha), (batch_size, K, L)
        diag_pos (tf.Tensor): Tensor containing diagonal positions of the delete 
                              states in diagonal g, (num_states,)
        g (int): Number of current diagonal in alpha

    Returns:
        tf.Tensor: Gathered log-probabilities for the specified delete states,
                   (batch_size, 2, num_states), dtype=tf.float32

    Notes:
        - To calculate a position [i, j] in alpha that corresponds to a delete 
          state the entries at [i-3, j] and [i-4, j] in alpha are needed
        - B: batch size, num_states: number of delete states in diagonal number g
    """
    batch_size = tf.shape(log_alpha)[0]  # Batch size
    num_states = tf.shape(diag_pos)[0]  # Number of delete states in diagonal number g

    # Row indices for the needed entries of alpha
    # g - diag_pos is the row in alpha of the respective states
    g_i_0 = tf.maximum(g - diag_pos - 3, 0)
    g_i_1 = tf.maximum(g - diag_pos - 4, 0)
    time_indices = tf.stack([g_i_0, g_i_1], axis=1)  # (num_states, 3)

    # Insert batch dimension (B, num_states, 3)
    time_indices = tf.tile(time_indices[None, :, :], [batch_size, 1, 1])
    state_indices = tf.tile(diag_pos[None, :, None]+1, [batch_size, 1, 2])
    batch_indices = tf.tile(tf.range(batch_size)[:, None, None], [1, num_states, 2])

    # Indices of needed entries, (B, num_states, 2, 3)
    gather_indices = tf.stack([batch_indices, time_indices, state_indices], axis=-1)

    # Gather entries of alpha
    gathered = tf.gather_nd(log_alpha, gather_indices)  # (B, num_states, 2)
    result = tf.transpose(gathered, perm=[0, 2, 1])
    
    return result    
    
@tf.function(reduce_retracing = True)
def extract_emission_prbs(B, u, diag_pos, g):
    """
    Extract the relevant emission probabilities for a given state type 
    (match, insert, or delete) from the precomputed sparse transition matrix A. 

    Args:
        A (tf.Tensor): Emission matrix of an pHMM (K, H)
        u (tf.Tensor): Input sequence (batchsize, input_length), dtype=tf.int32
        diag_pos (tf.Tensor): Tensor containing diagonal positions of a 
                              state type in diagonal g
        g (int): Number of current diagonal in alpha

    Returns:
        tf.Tensor: Extracted emission probabilities for the considered states, 
                   (batchsize, len(diag_pos)), dtype=tf.float32
    Notes:
        - To calculate a position [i, j] in alpha the entry at [i, u[diag_pos]]
          in emission matrix B is needed
        - K: Number of states
        - H: Alphabet size
    """
    batchsize = tf.shape(u)[0]
    
    # Row indices 
    rows = g - diag_pos  # g - diag_pos is the row in alpha of the respective states
    rows = tf.tile(rows[None, :], [batchsize, 1]) # (batchsize, len(diag_pos))
    
    # Column indices: u[diag_pos]
    cols = tf.gather(u, diag_pos+1, axis = 1)
    
    # Indices of needed entries
    indices = tf.stack([rows, cols], axis=-1)
    
    # Gather entries of emission matrix B, (batchsize x len(diag_pos))
    B_filtered = tf.gather_nd(B, indices)
    
    return B_filtered

@tf.function(reduce_retracing = True)
def make_diag_positions(start, diag_length, step=3):
    """
    Generate a sequence of positions in a diagonal. The positions are spaced 
    evenly with a configurable step size (default 3), starting from a given 
    offset.
    
    Args:
        start (int): Starting index for the diagonal positions
        diag_length (int): Maximum allowed index (exclusive upper bound)
        step (int, optional): Step size between positions (default: 3)
    
    Returns:
        tf.Tensor: 1D tensor of type tf.int32 containing the diagonal positions
    
    Notes:
        - The output includes values: start, start + step, start + 2*step, ..., 
          up to diag_length - 1
    """
    # Maximum length of the final vector with position indices
    max_len = (diag_length - 1 - start) // step + 1
    return tf.range(start, start + max_len * step, step, dtype=tf.int32)

@tf.function(reduce_retracing = True)
def combine_diag_elements(value_index_pairs, default_value=-float('inf')):
    """
    Combines (values, indices)-pairs in a single tensor with order of values
    as given in indices.

    Args:
        value_index_pairs: List of tupels (values, indices)
            - values: tensor with values (batch_size, num_values)
            - indices: tensor with position of values (num_values,)
        default_value: Default for unset positions

    Returns:
        tf.Tensor: tensor  (batch_size, total_cols)
    """
    batch_size = tf.shape(value_index_pairs[0][0])[0]  # Batch size

    all_values = []
    all_indices = []

    for values, col_indices in value_index_pairs:
        num_cols = tf.shape(values)[1]

        # Batch-indizes: (batch_size, num_cols)
        batch_ids = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, num_cols])
        col_ids = tf.tile(tf.expand_dims(col_indices, 0), [batch_size, 1])

        # Combine Indices: (batch_size * num_cols, 2)
        combined_indices = tf.stack([batch_ids, col_ids], axis=-1)
        combined_indices = tf.reshape(combined_indices, [-1, 2])

        all_indices.append(combined_indices)
        all_values.append(tf.reshape(values, [-1]))

    final_indices = tf.concat(all_indices, axis=0)
    final_values = tf.concat(all_values, axis=0)

    max_col_index = tf.reduce_max(final_indices[:, 1]) + 1
    output_shape = [batch_size, max_col_index]

    # Combine values
    output = tf.scatter_nd(indices=final_indices, updates=final_values, shape=output_shape)
    
    #Remove zero columns (arise if diag_idx does not start with 0)
    mask = tf.reduce_any(tf.not_equal(output, 0.0), axis=0)
    output = tf.boolean_mask(output, mask, axis=1)

    return output

@tf.function(reduce_retracing = True)
def get_diagonal_indices(start_row, diag_length, num_rows, batchsize):
    """
    Generates a list of indices representing the positions along a diagonal in 
    a matrix, starting at a given row (start_row) and in the second column. The 
    diagonal runs from the bottom-left to the top-right of the matrix.

    Args:
        start_row (int): row in which diagonal begins.
        diag_length (int): length of the diagonal.
        num_rows (int): first dimension of the matrix.

    Returns:
        tf.Tensor: Tensor (N, 2), wobei jede Zeile ein gültiges (row, col)-Paar ist.
    """
    # Maximum possible length of the diagonal
    max_len = tf.minimum(start_row + 1, diag_length) 
    offsets = tf.range(max_len)

    row_indices = start_row - offsets
    col_indices = 1 + offsets

    # Combine row and column indices
    indices = tf.stack([row_indices, col_indices], axis=1)

    # Remove indices that are not inside the matrix 
    valid_mask = row_indices < num_rows
    indices = tf.boolean_mask(indices, valid_mask)
    
    # Add batch indices
    # Determine number of updates
    num_updates = tf.shape(indices)[0]
    
    # batch_indices: (batchsize,)
    batch_indices = tf.range(batchsize)
    
    # Indices for each batch, (num_updates, batchsize, 2)
    indices_tiled = tf.tile(indices[:, tf.newaxis, :], [1, batchsize, 1])
    
    # Batch-indices, (num_updates, batchsize, 1)
    batch_indices_tiled = tf.tile(batch_indices[tf.newaxis, :, tf.newaxis], [num_updates, 1, 1])
    
    # Combine: (num_updates, batchsize, 3)
    full_indices = tf.concat([batch_indices_tiled, indices_tiled], axis=-1)
    
    # Change order
    full_indices = tf.reshape(tf.transpose(full_indices, [1, 0, 2]), [-1, 3])

    return full_indices


@tf.function(reduce_retracing = True)
def pHMM_diag_vectorised(A_insert_col, A_match_col, A_delete_col, B, pi, u, U):
    """
    This function computes the forward probabilities (alpha) in log space for a 
    profile Hidden Markov Model (pHMM). The entries of alpha are calculated
    diagonally.
    
    Args:
        A_insert_col (tf.Tensor): Transition vector (1, K-L+1), dtype=tf.float32
        A_match_col (tf.Tensor): Transition vector (1, K-1), dtype=tf.float32
        A_delete_col (tf.Tensor): Transition vector (1, K-L-1), dtype=tf.float32
        B (tf.Tensor): Emission matrix (K, H), dtype=tf.float32
        pi (tf.Tensor): Initial state distribution (batchsize, K), dtype=tf.float32
        u (tf.Tensor): Input sequence (batchsize, input_length), dtype=tf.int32
        U (tf.Tensor): One-hot encoded input sequence (batchsize, input_length, H), 
                       dtype=tf.float32
    Returns:
        tf.Tensor: Forward probability matrix (alpha) (batchsize, K, input_length)
                   dtype=tf.float32
    Notes:
        - H: Alphabet size
        - K: Number of states, K = 3 * input_length + 1
        - L: input length
        - length of a diagonal: input_length - 1
    """
    input_length = tf.shape(u)[1]  # Length of input sequence
    K = 3*input_length+1  # Number of states 
    batchsize = tf.shape(u)[0]  # Batch size
    diag_length = input_length - 1  # Length of a diagonal of alpha 

    # Convert transition and emission matrix and initial distribution to log space
    A_insert_log, A_match_log, A_delete_log = [tf.where(A_ > 0, tf.math.log(A_), -1e6) 
                                  for A_ in (A_insert_col, A_match_col, A_delete_col)] 
    pi_log = tf.where(pi > 0, tf.math.log(pi), -1e6) 
    B_log = tf.where(B > 0, tf.math.log(B), -1e6)  
    
    # Precompute the emission probabilities for each time step 
    UB_log = tf.matmul(U, tf.transpose(B_log))  
    # (batch_size x L x H) @ (H x K) -> (batch_size x # log-space x K)
    
    # Initialization, (batchsize x K x L)
    log_alpha = tf.fill([batchsize, K, input_length], -1e6)

    # Calculate first column of log_alpha
    updates = pi_log[None, :] + UB_log[:, 0, :]
    updates = tf.reshape(updates, [-1])
    
    # Indizes erzeugen: shape [batchsize * K, 3]
    indices = tf.stack(tf.meshgrid(tf.range(batchsize),
                                   tf.range(K),
                                   tf.constant([0]), indexing='ij'), axis=-1)
    indices = tf.reshape(indices, [-1, 3])
    # Update log_alpha for t = 1
    log_alpha = tf.tensor_scatter_nd_update(log_alpha, indices, updates)
    
    # Determine the positions of the different state types in the diagonal
    diag_pos_complete = [
        make_diag_positions(0, diag_length),  # (0, 3, 6, ...)
        make_diag_positions(1, diag_length),  # (1, 4, 7, ...)
        make_diag_positions(2, diag_length),  # (2, 5, 8, ...)
    ]
    
    # Create object for correct mapping of diagonal positions with states
    # index = (insert, match, delete)
    offset_map = tf.constant([
                    [0, 2, 1],  # offset = 0
                    [1, 0, 2],  # offset = 1
                    [2, 1, 0]   # offset = 2
                    ], dtype=tf.int32)
    
    for g in tf.range(0, input_length + K-2): # Loop over all diagonals 

        # First diagonals do not have full length
        if g < diag_length: # Remove positions of the diagonals, that are not inside alpha
            diag_pos = [tf.boolean_mask(t, t < 1+g) for t in diag_pos_complete]
        elif g >= K:  # Last diagonals do not have full length either, remove
            diag_pos = [tf.boolean_mask(t, t > g-K) for t in diag_pos_complete]
        else:
            diag_pos = diag_pos_complete
            
        # Determine offset
        offset = g % 3
        state_indices = offset_map[offset]  # Tensor with [i_idx, m_idx, d_idx]
        
        # Determine the order in which state types (insert, match, delete) appear 
        # in the current diagonal
        i_idx = state_indices[0] 
        m_idx = state_indices[1]
        d_idx = state_indices[2]
        
        # Select the appropriate positions for each state type from the diagonal
        diag_pos_insert = get_diag_position(diag_pos, i_idx) # indices of insert states in diag g
        diag_pos_match  = get_diag_position(diag_pos, m_idx)
        diag_pos_delete = get_diag_position(diag_pos, d_idx)

        # Calculate entries for insert state positions in diagonal g
        if tf.shape(diag_pos_insert)[0] == 0:
            diag_elem_insert = tf.zeros([batchsize, 0], dtype=tf.float32)
        else:
            # Get needed entries of A
            A_insert = extract_transition_prbs(A_insert_log, 'insert', diag_pos_insert, g)
            
            # Get needed entries of B
            B_insert = extract_emission_prbs(B_log, u, diag_pos_insert, g)  
            
            # Get needed entries of alpha
            alpha_insert = extract_log_alpha_insert(log_alpha, diag_pos_insert, g)
            
            # Calculate all insert state entries of diagonal g
            diag_elem_insert = tf.reduce_logsumexp(A_insert + alpha_insert, axis = 1) + B_insert
            # batchsize x len(diag_pos_insert)
            
        # Calculate entries for match state positions in diagonal g
        if tf.shape(diag_pos_match)[0] == 0:
            diag_elem_match = tf.zeros([batchsize, 0], dtype=tf.float32)
        else:
            # Get needed entries of A
            A_match = extract_transition_prbs(A_match_log, 'match', diag_pos_match, g)
            
            # Get needed entries of B
            B_match = extract_emission_prbs(B_log, u, diag_pos_match, g)
            
            # Get needed entries of alpha
            alpha_match = extract_log_alpha_match(log_alpha, diag_pos_match, g)
            
            # Calculate all match state entries of diagonal g
            diag_elem_match = tf.reduce_logsumexp(A_match + alpha_match, axis = 1) + B_match
        
        # Calculate entries for delete state positions in diagonal g
        if tf.shape(diag_pos_delete)[0] == 0:
            diag_elem_delete = tf.zeros([batchsize, 0], dtype=tf.float32)
        else:
            # Get needed entries of A
            A_delete = extract_transition_prbs(A_delete_log, 'delete', diag_pos_delete, g)
            
            # Get needed entries of B
            B_delete = extract_emission_prbs(B_log, u, diag_pos_delete, g)
            
            # Get needed entries of alpha
            alpha_delete = extract_log_alpha_delete(log_alpha, diag_pos_delete, g)
            
            # Calculate all delete state entries of diagonal g
            diag_elem_delete = tf.reduce_logsumexp(A_delete + alpha_delete, axis = 1) + B_delete
    
        # Put elements of the diagonal together in respective order
        diagonal = combine_diag_elements([
            (diag_elem_insert, diag_pos_insert),
            (diag_elem_match, diag_pos_match),
            (diag_elem_delete, diag_pos_delete)])
        diagonal = tf.reshape(diagonal, [-1])
        
        # Get indices of the diagonal
        diag_indices = get_diagonal_indices(start_row = g, diag_length = diag_length, 
                                            num_rows = K, batchsize = batchsize)

        # Insert the calculated diagonal in log_alpha
        log_alpha = tf.tensor_scatter_nd_update(log_alpha, diag_indices, 
                                                diagonal)
        # Insert each diagonal immediately after its calculation, as it is 
        # required for the following calculations
    return log_alpha

#%%
#small example
'''
# Uncomment and adjust the following lines if you want to run this script standalone
# (i.e., not imported by another script):
# Change the working directory to the project directory
#import os
#homefolder = "..."
#project_dir = os.path.join(homefolder, "Programs", "pHMM_code")
#os.chdir(project_dir)

from _funcs_pHMM_tf import (
    _create_sparse_transition_matrix,
    _create_emission_prb_matrix, 
    _create_initial_distribution)

tf.random.set_seed(28102024) 

batchsize = 2
H = 20  # Alphabet size
input_length = [3, 10, 25] # List of different sequence lengths

# Generate input sequences with batch dimension, elements of dim (batchsize x input_length)
input_batch_list = [
    tf.random.uniform(shape=(batchsize, L), maxval=H, dtype=tf.int32)
    for L in input_length]

# One-hot encode the input sequences, elements of dim (batchsize x input_length x H)
one_hot_batch_list = [tf.one_hot(u, H) for u in input_batch_list] 

i = 0
K = input_length[i]*3+1
A_insert_col, A_match_col, A_delete_col = _create_sparse_transition_matrix(K)

B = _create_emission_prb_matrix(input_length[i], H)
pi = _create_initial_distribution(input_length[i])
u = input_batch_list[i]
U = one_hot_batch_list[i]

alpha_diag_vec_tf = pHMM_diag_vectorised(A_insert_col, A_match_col, A_delete_col, B, pi, u, U)
print(alpha_diag_vec_tf)
tf.argmax(alpha_diag_vec_tf, axis=1)

last_log_alpha = alpha_diag_vec_tf[:, :, -1]  # shape: (batchsize, K)
log_likelihood = tf.reduce_logsumexp(last_log_alpha, axis=1)  # shape: (batchsize,)
print(-log_likelihood)
#%%
# Apply the algorithm and compute gradient
with tf.GradientTape() as tape:
    tape.watch(A_insert_col)
    tape.watch(A_match_col) 
    tape.watch(A_delete_col)
    tape.watch(B)
    tape.watch(pi)  # Watch pi, if not gradient is None
    
    #alpha_log = pHMM_diag_vectorised(u, U, alphabet_size = H)
    alpha_log = pHMM_diag_vectorised(A_insert_col, A_match_col, A_delete_col, B, pi, u, U)  # (batch x K x L)
    final_log_prob = tf.reduce_logsumexp(alpha_log[:, :, -1], axis=1)  
    loss = -tf.reduce_mean(final_log_prob)  # Negative Log-Likelihood Loss

# Calculate gradient
grads = tape.gradient(loss, [A_insert_col, A_match_col, A_delete_col, B, pi])
print(grads)
'''
