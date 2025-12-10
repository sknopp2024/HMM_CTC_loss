#Sara Knopp, last modified on 14.08.2025

# Generate data for CTC_HMMs

import tensorflow as tf

def generate_data_ctc(tf_ctc_loss_fct = False):
    '''
    Generates synthetic input data and model parameters for evaluating the 
    performance of CTC-Loss calculation.

    This function creates input sequences of varying lengths along with 
    corresponding logits. Optionally, it also prepares the data in a format 
    compatible with TensorFlow's CTC loss function (`tf.nn.ctc_loss`), 
    including batched logits and sequence length information.

    Args:
        tf_ctc_loss_fct (bool): 
            If True, the returned data includes additional formatting required 
            for use with TensorFlow's `tf.nn.ctc_loss` function, such as 
            batched logits and length tensors.

    Returns:
        If tf_ctc_loss_fct is False:
            - labels_list (list of tf.Tensor): List of integer sequences 
              representing target labels (one per input length). DIMENSION?
            - logits_batch (list of tf.Tensor): List of logits tensors with 
              batch dimension (shape: [batch_size, num_frames, num_labels]).
            - num_frames (list of int): List of frame counts corresponding to 
              each input sequence.
            - blank (int): Index of the blank symbol used for CTC loss.

        If tf_ctc_loss_fct is True:
            - labels_list (list of tf.Tensor): Same as above.
            - logits_batch (list of tf.Tensor): List of logits tensors with 
              batch dimension (shape: [batch_size, num_frames, num_labels]).
            - label_length_list (list of tf.Tensor): List of label lengths 
              (shape: [batch_size], dtype: int64).
            - logit_length (list of list of int): List of logits lengths per 
              sequence in the batch.
            - blank (int): Index of the blank symbol used for CTC loss.
    '''
    # Set random seed for reproducibility
    tf.random.set_seed(28102024)
    print('Start of data generation.')
    
    H = tf.constant(4, dtype=tf.int32)  # Alphabet size (number of nucleotides: A, C, G, T)
    blank = tf.constant(4, dtype=tf.int32)  # Blank index 
    num_labels = tf.constant(5, dtype=tf.int32)  # Number of labels [A, C, G, T, e]
    batchsize = tf.constant(8, dtype=tf.int32)  
    
    # Number of frames
    num_frames = tf.constant([1000, 2000, 5000, 10000, 20000], dtype=tf.int32) 
    # Length of input sequences
    label_length = tf.constant([100, 200, 500, 1000, 2000], dtype=tf.int32)  
    
    # Create a list of input sequences of different length
    labels_list = [tf.random.uniform([batchsize, length], minval = 0, maxval = H, 
                                     dtype=tf.int32) for length in label_length]
    # Create list of logits
    logits_list = [tf.random.uniform([frames, num_labels]) for frames in num_frames]
    logits_expanded = [tf.expand_dims(logits, axis = 0) for logits in logits_list]  
    logits_batch = [tf.repeat(logits, batchsize, axis = 0) for logits in logits_expanded]
    
    if tf_ctc_loss_fct:
        
        # Lenght of label
        label_length_list = [tf.fill([batchsize], tf.cast(length, tf.int32))
                             for length in label_length]
        
        logit_length_list = [tf.fill([batchsize], nf) for nf in num_frames] 
        # len(num_frames) x batchsize
    
        print('Data generation finished.')
        return labels_list, logits_batch, label_length_list, logit_length_list, blank
    
    print('Data generation finished.')
    return labels_list, logits_batch, num_frames, blank