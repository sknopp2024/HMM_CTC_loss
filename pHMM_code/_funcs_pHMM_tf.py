# Sara Knopp, last modified on 09.05.2025

# Utility functions for profile HMMs

import tensorflow as tf


def _create_sparse_transition_matrix(K):  
    """
    Create sparse transition matrices for the three types of states of a pHMM. 
    To be more precise, the sparse matrices are only column vectors, whereby 
    the entries for each type of state from the KxK transition matrix are put
    together columnwise.

    Args:
        K (int): Number of states of the pHMM
        
    Returns:
        A_insert (tf.Tensor): Transition probabilities for insert states (1, K-L+1), 
                              dtype=tf.float32
        A_match (tf.Tensor): Transition probabilities for match states (1, K-1), 
                             dtype=tf.float32
        A_delete (tf.Tensor): Transition probabilities for delete states (1, K-L-1), 
                              dtype=tf.float32
    Notes:
        - K: number of states
        - L: input length
    """
    # number of repetitions of the middle part
    num_middle_blocks = tf.math.floordiv(K + 2 - 6, 3)

    # Create transition column for insert states
    # Create parts of the column
    start = tf.constant([0.0, 0.07], dtype=tf.float32)
    middle_block = tf.constant([0.07, 0.07], dtype=tf.float32)
    middle = tf.tile(middle_block, [num_middle_blocks])
    end = tf.constant([1.0, 1.0], dtype=tf.float32)

    # Combine blocks
    values = tf.concat([start, middle, end], axis=0)  
    A_insert = tf.expand_dims(values, axis=1)  # Expand dimension
    
    # Create transition column for match states
    # Create parts of the column
    start = tf.constant([0.0, 0.0, 0.93], dtype=tf.float32)
    middle_block = tf.constant([0.9, 0.97, 0.93], dtype=tf.float32)
    middle = tf.tile(middle_block, [num_middle_blocks+1])
    
    # Combine blocks
    values = tf.concat([start, middle], axis=0)  
    A_match = tf.expand_dims(values, axis=1)  # Expand dimension
    
    # Create transition column for delete states
    # Create parts of the column
    start = tf.constant([0.0, 0.0], dtype=tf.float32)
    middle_block = tf.constant([0.03, 0.03], dtype=tf.float32)
    middle = tf.tile(middle_block, [num_middle_blocks+1])
    
    # Combine blocks
    values = tf.concat([start, middle], axis=0)  
    A_delete = tf.expand_dims(values, axis=1)  # Expand dimension
      
    return A_insert, A_match, A_delete

def _create_transition_matrix(input_length):  
    """
    Creates the transition matrix for a Profile HMM (pHMM) given an input 
    sequence length L.

    Args:
        input_length (int): The length L of the input sequence for which the 
        transition matrix is created.

    Returns:
        tf.Tensor: A transition (K, K) matrix containing the transition 
                    probabilities between different states (dtype=tf.float32).

    Notes:
        - K: number of states, K = input_lengt*3 + 1
        - The matrix consists of K states:
          I_0, M_1, D_1, I_1, ..., M_j, D_j, I_j, ..., M_L, D_L, I_L
    """
    K = input_length * 3 + 1  # Number of states
    
    # Initialise transition matrix    
    A = tf.zeros((K, K), dtype=tf.float32)

    # State-Indices
    match_idx  = tf.range(1, K - 3, delta=3, dtype=tf.int32)
    insert_idx = tf.range(0, K - 3, delta=3, dtype=tf.int32)
    delete_idx = tf.range(2, K - 3, delta=3, dtype=tf.int32)

    # Hilfsfunktion für gezieltes Setzen von Einträgen
    def scatter_update(mat, rows, cols, values):
        idx = tf.stack([rows, cols], axis=1)
        updates = tf.scatter_nd(idx, values, shape=tf.shape(mat))
        return mat + updates

    # Match-Transitions
    A = scatter_update(A, match_idx, match_idx + 3, tf.fill(tf.shape(match_idx), 0.9))  # Match → next Match
    A = scatter_update(A, match_idx, match_idx + 2, tf.fill(tf.shape(match_idx), 0.07)) # Match → Insert
    A = scatter_update(A, match_idx, match_idx + 4, tf.fill(tf.shape(match_idx), 0.03)) # Match → Delete

    # Insert-Transitions
    A = scatter_update(A, insert_idx, insert_idx + 1, tf.fill(tf.shape(insert_idx), 0.93)) # Insert → Match
    A = scatter_update(A, insert_idx, insert_idx,     tf.fill(tf.shape(insert_idx), 0.07)) # Insert → Insert

    # Delete-Transitions
    A = scatter_update(A, delete_idx, delete_idx + 2, tf.fill(tf.shape(delete_idx), 0.97)) # Delete → Match
    A = scatter_update(A, delete_idx, delete_idx + 3, tf.fill(tf.shape(delete_idx), 0.03)) # Delete → Delete

    # Last transitions
    A = scatter_update(A, tf.constant([K - 3]), tf.constant([K - 1]), tf.constant([1.0])) # Last Match → Insertion
    A = scatter_update(A, tf.constant([K - 1]), tf.constant([K - 1]), tf.constant([1.0])) # Last insert → itself

    return A

def _create_initial_distribution(input_length): 
    """
    Creates an initial distribution pi for a model based on a given input length.
    
    Args:
        input_length (int): length of the input sequence 
        
    Returns:
        tf.Tensor: the initial distribution pi of length K (dtype = float32) 
    
    Notes:
        - K: Number of states (3*input_length+1)
        The distribution is set such that:
        - 90% probability of starting in the first match state
        - 7% probability of starting in the first insert state
        - 3% probability of starting in the first delete state
    """
    pi = tf.tensor_scatter_nd_update(
        tf.zeros([3 * input_length + 1], dtype=tf.float32),
        indices=tf.constant([[0], [1], [2]], dtype=tf.int32),  # positions to change
        updates=tf.constant([0.07, 0.9, 0.03], dtype=tf.float32)) 
    return pi

def _create_emission_prb_matrix(input_length, alphabet_size):  
    """
    Creates an emission prb matrix B with random emission prbsbased on a given 
    input length.
    
    Args:
        input_length (int): Length of the input sequence. 
        alphabet_size (int): Size of the alphabet. 
        
    Returns:
        tf.Tensor: A (K, H) emission prb matrix B (dtype = float32).
    
    Notes:
        - K: Number of states (3*input_length+1)
        - H: alphabet size
    """
    tf.random.set_seed(28102024)  # Set seed for reproducibility
    
    # Generate random emission probabilities (K x H) 
    B_logits = tf.random.uniform(shape=(input_length * 3 + 1, alphabet_size), 
                                 minval=0, maxval=100, dtype=tf.float32)/100
    B = tf.nn.softmax(B_logits, axis=1)
    
    return B

def _create_distance_array(input_length):  
    """
    Creates a distance array d based on a given input length.
    
    Args:
        input_length (int): length of the input sequence 
        
    Returns:
        np.ndarray: the distance array of length K
    
    Notes:
        - K: Number of states (3*input_length+1)
        - Distance for match and insert states is 1, for delete states 0
    """
    K = input_length * 3 + 1  # Number of states

    # Initialise tensor with ones
    d = tf.ones((K,), dtype=tf.int32)

    # Indices of delete states
    indices_to_zero = tf.range(2, K, delta=3)

    # Create list of integer arrays, 1 for match/insert, 0 for delete state
    d = tf.tensor_scatter_nd_update(
        d,
        indices=tf.expand_dims(indices_to_zero, axis=1), 
        updates=tf.zeros_like(indices_to_zero, dtype=tf.int32))
    
    return d