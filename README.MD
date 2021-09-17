### 15-418 Final Project: Distributed Machine Learning

![Final Report](pdfs/FinalReport.jpg)

Summary: \
For our project, we will implement distributed Machine Learning using Message Passing Interface and CUDA in efforts to speed up the training time of a linear regression model. Specifically, we will use MPI to enable parallel gradient computation and Ring-AllReduce for convergence. We will implement N-bounded delay Bulk Synchronous Parallel (BSP) Execution, as well as exploring and analyzing different approaches to the challenges it entails, such as consistency and synchronization. If time permits, we will be using CUDA in efforts to speedup the batch gradient computation.

compile: 

cmake --build .

Run sequential:
./lr_decentralized 1000 0.01 data/weatherAUS.csv out.txt Yes No 

Run decentralized distributed: 
mpirun -np [num_proc] ./lr_decentralized [num_epoch] [learning_rate] [infile] [outfile] [label1] [label2] 

example: mpirun -np 6 ./lr_decentralized 1000 0.01 data/weatherAUS.csv out.txt Yes No 

Run decentralized distributed: 
mpirun -np [num_proc] ./lr_centralized [num_epoch] [learning_rate] [N-bound] [infile] [outfile] [label1] [label2] 

example: mpirun -np 6 ./lr_decentralized 1000 0.01 5 data/weatherAUS.csv out.txt Yes No 