## Abstract 

Deep learning methods are widely used in many areas of bioinformatics. The rapid growth of available sequencing data calls for
models that can be executed efficiently and make optimal use of available computational resources. This thesis investigates how classical
sequence models can be implemented efficiently using state-of-the-art computing frameworks to meet the performance requirements of modern 
sequencing data analysis. 
<br>
Three research objectives related to sequence modelling in bioinformatics are addressed. Firstly, Hidden Markov Models (HMMs), widely used 
in genome annotation tools, are implemented using diverse computational strategies and evaluated for performance. Secondly, this study 
investigates whether profile Hidden Markov Models (pHMMs), which are frequently used in sequence alignment tools, can be parallelised. 
Thirdly, as the Connectionist Temporal Classification (CTC) loss is widely applied in nanopore sequencing basecallers, a novel method for 
its efficient computation is developed using a probabilistic model.
<br>
Realistic data was selected to ensure meaningful performance evaluation. The implementation was carried out in Python using TensorFlow to enable 
GPU acceleration and parallel computation. The developed methods were evaluated based on computation time, GPU utilisation, and GPU memory consumption.
<br>
Of all HMM implementations evaluated, the variant employing parallelisation along the time dimension achieved the best runtime performance, while 
maintaining memory usage suitable for practical applications. In addition, a parallelised pHMM was developed that achieved reduced runtime without 
incurring additional memory overhead compared to non-parallel implementations. Lastly, a Conditional Random Field capable of calculating the CTC loss 
was successfully designed. Its parallelised versions demonstrated significantly lower memory usage and shorter runtimes than non-parallelised approaches. 
<br>
The presented methods enable more efficient processing of sequence data and may contribute to the development of faster and more scalable genome annotation, 
sequence alignment and basecalling tools.
