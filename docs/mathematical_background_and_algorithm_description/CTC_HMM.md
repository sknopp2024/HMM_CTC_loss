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

