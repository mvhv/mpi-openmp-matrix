Distributed Memory Matrix Multiplication
----------------------------------------
A small experiment in distributed-memory multi-processing. Matrix-multiplication implemented serially in plain C and then parallelised using both OpenMP and MPI.

MPI is used for type defition and distributed-memory functionality across multiple physical nodes, with OpenMP providing the logical multi-threading functionality within each physical node.
