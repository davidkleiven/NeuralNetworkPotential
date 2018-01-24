# NeuralNetworkPotential
Neural Network Pair Potential

# Caveats
 - Sometimes numpy uses multiple threads when the matricies become large enough
   (due to linking with OpenBLAS). Since the code is parallelized using
   MPI this may lead to lower performance.
   To limit the maximum number of threads set
   ```bash
   export OMP_NUM_THREADS=2
   ```
   if one want to limit the number of threads to 2.
