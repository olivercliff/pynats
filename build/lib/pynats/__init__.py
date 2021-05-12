# For some reason the JVM causes a segfault with OpenBLAS (numpy's linalg sovler). Need to halt multithreading before starting JVM:
import os
os.environ['OMP_NUM_THREADS'] = '1'