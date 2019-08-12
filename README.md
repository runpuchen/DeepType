#DeepType

Deep Learning Approach to Identifying Breast Cancer Subtypes Using High-Dimensional Genomic Data

#Code Organization
•	PreTrainingTuneALpha.py
•	TrainingTuneBeta.py
•	KmeansParaSelection.py
•	load_biology_data.py

#Requirements
•	python
•	tensorflow

#Implement and Activate Tensorflow Environment under Conda
##Implement:
conda create -n tensorflow_env tensorflow
##Activation:
conda activate tensorflow_env

#Use the software
##Data format 
filename.mat file
##Variables
Data: D*N numerical matrix. Each row is a gene, and each column is a sample. The genes should be ranked in the descending order by variances across samples.
targets: N*1 numerical vector. The ith element denotes the class that the ith sample belongs to.

#Set parameters
learning_rate: learning rate
num_pretrain: number of batches in pretraining process
num_train: number of training steps in each epoch of training process
num_train_epoch: number of epochs in training process
batch_size_pretrain: batch size in pretraining process
batch_size_train: batch size in training process
batch_size_test: number of test samples
num_groups: number of subtypes expected to detect
num_labels: number of classes
num_hidden_1: number of hidden nodes in the first hidden layer
num_hidden_2: number of hidden nodes in the second hidden layer
k_list: list of number of clusters
lambda1_list: list of sparsity penalty coefficient
lambda1_list: list of sparsity penalty coefficient
lambda2_list: list of K-means penalty coefficient
n_fold: number of folds



#Run in command line with required command arguments:

python PreTrainingTuneALpha.py

python TrainingTuneBeta.py



