**DeepType**

Deep Learning Approach to Identifying Breast Cancer Subtypes Using High-Dimensional Genomic Data 

**Code Organization**

This software contains these codes:

- DeepType.py
- data.py
- eval.py
- flags.py
- model.py
- training.py
- utils.py

**Requirements**

- python
- tensorflow

**Implement and Activate Tensorflow Environment under Conda**

- Implement:
conda create -n tensorflow_env tensorflow

- Activation:
conda activate tensorflow_env

**Use the software**

**1. Data format:** filename.mat file

**2. Variables:**

  Data: D*N numerical matrix. Each row is a gene, and each column is a sample. The genes should be ranked in the descending order by variances across samples.

  targets: N*1 numerical vector. The ith element denotes the class that the ith sample belongs to.

**3. Set parameters in flags.py:**

  NUM_GENES_1: the number of input genes. 

  NUM_CLUSTERS: the number of clusters K.

  NUM_HIDDEN: the number of hidden layers.

  NUM_NODES: numerical vector, the numbers of nodes in the hidden layers.

  NUM_CLASSES: the number of unique classes of samples.

  NUM_TRAIN_SIZE: the number of samples in the training set.

  NUM_VALIDATION_SIZE: the number of samples in the validation set.

  NUM_TEST_SIZE: the number of samples in the test set.

  NUM_SAMPLE_SIZE: the number of samples in the whole dataset.

  NUM_BATCH_SIZE: batch size.

  NUM_LEARNING_RATE: learning rate.

  NUM_SUPERVISED_BATCHES: the number of training steps in the supervised initialization.

  NUM_TRAIN_BATCHES: the number of training steps in each epoch.

  LAMBDA: sparsity penalty coefficient.

  ALPHA: K-means loss coefficient.

  DATA_DIR: Directory to put the training data.

  RESULT_DIR: Directory to put the results.

**4. Run the program**

  python DeepType.py
  
**5. Data available**

Due to the file size limit of Github, the breast cancer dataset is available at https://drive.google.com/file/d/1ao1zu3DS8GkYF-tHxpQ-1ev2psxXL-fx/view?usp=sharing





