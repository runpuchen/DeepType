from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from load_biology_data import *
import pickle
from Visualize import Visualize_TSNE_PCA, Transfer_TSNE_PCA
from KmeansParaSelection import SelectK
import os
import sys

num_gene = int(sys.argv[1])
print('number of genes: ', num_gene)
local_dir = os.getcwd()

def make_train_validation(data1, data2, targets, k, n_fold):
    [num_train_sample, _] = data1.shape
    num_block = int(num_train_sample/n_fold)
    id_val = range(num_block *k, num_block *(k+1))
    id_train = np.setdiff1d(range(num_train_sample), id_val)
    train_data1 = data1[id_train,:]
    train_data2 = data2[id_train,:]
    train_targets = targets[id_train,:]
    val_data1 = data1[id_val, :]
    val_data2 = data2[id_val, :]
    val_targets = targets[id_val,:]
    batch_size_val = num_block

    return train_data1, train_data2, train_targets, val_data1, val_data2, val_targets, batch_size_val


#################################################
#################################################
# Parameter setting
np.random.seed(0)
tf.set_random_seed(0)

# Training Parameters
learning_rate = 0.002

num_pretrain = 500


batch_size_pretrain = 512
batch_size_test = 131

num_groups =13 # brca, this can change. MNIST: 15
num_labels = 5 # brca. MNIST: 10
k_list = list([  9, 10, 11, 12, 13, 14, 15] )# this can change

lambda0 = 1
#lambda1_list = list([0.1])
lambda1_list = np.logspace(-2, -6, num=10, endpoint=True)

lambda1_list = np.concatenate((lambda1_list, [0.0]), axis=0)
lambda1_list = sorted(lambda1_list)
print(lambda1_list)
n_fold = 10


dropout = 0.2
graduate_threshold = 0.01

display_step = 100

# Network Parameters
num_hidden_1 = 1024 # 1st layer num features
num_hidden_2 = 512 # 2nd layer num features (the latent dim)
num_input = num_gene

show_choice = 0
normalize = 1
#################################################
#################################################


# Import biology data
filename = local_dir+ '/BRCA/2View/' + 'BRCA2View' + str(18000) + '.mat'
result_dir = local_dir+ '/BRCA/results/2View/' + str(num_gene) + '/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
print(result_dir)

data1_brca_train, data2_brca_train, targets_brca_train, data1_brca_test, data2_brca_test, targets_brca_test, index = \
    load_biology_data_2View(filename, batch_size_test, normalize, num_gene)


[num_train_sample, _] = data1_brca_train.shape # the total train dataset

data1_brca = np.concatenate((data1_brca_train, data1_brca_test), axis = 0)
data2_brca = np.concatenate((data2_brca_train, data2_brca_test), axis = 0)
targets_brca = np.concatenate((targets_brca_train, targets_brca_test), axis = 0)


# tf Graph input (only pictures)
X1 = tf.placeholder("float", [None, num_input])
X2 = tf.placeholder("float", [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_labels])
M = tf.placeholder(tf.float32, [None, num_hidden_2])
lambda1 = tf.placeholder(tf.float32)
lambda2 = tf.placeholder(tf.float32)

weights = {
    'encoder_h1_v1': tf.Variable(tf.random_normal([num_input, num_hidden_1], stddev=0.02)),
    'encoder_h1_v2': tf.Variable(tf.random_normal([num_input, num_hidden_1], stddev=0.02)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1*2, num_hidden_2], stddev=0.02)),
    'classify_h3': tf.Variable(tf.random_normal([num_hidden_2, num_labels],  stddev=0.02)),


}
biases = {
    'encoder_b1_v1': tf.Variable(tf.random_normal([num_hidden_1],  stddev=0.02)),
    'encoder_b1_v2': tf.Variable(tf.random_normal([num_hidden_1],  stddev=0.02)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2],  stddev=0.02)),
    'classify_b3': tf.Variable(tf.random_normal([num_labels],  stddev=0.02)),
}





# Building the encoder


def encoder(x1,x2):
    # Encoder Hidden layer with sigmoid activation #1

    layer_1_v1 = tf.nn.sigmoid(tf.add(tf.matmul(x1, weights['encoder_h1_v1']),
                                   biases['encoder_b1_v1']))
    layer_1_v2 = tf.nn.sigmoid(tf.add(tf.matmul(x2, weights['encoder_h1_v2']),
                                   biases['encoder_b1_v2']))

    layer_1_v1 = tf.maximum(layer_1_v1, 1.e-9)
    layer_1_v1 = tf.minimum(layer_1_v1, 1 - 1.e-9)
    layer_1_v2 = tf.maximum(layer_1_v2, 1.e-9)
    layer_1_v2 = tf.minimum(layer_1_v2, 1 - 1.e-9)
    # combine
    layer_1 = tf.concat([layer_1_v1, layer_1_v2], 1)

    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))


    layer_2 = tf.maximum(layer_2, 1.e-9)
    layer_2 = tf.minimum(layer_2, 1 - 1.e-9)
    return layer_2



# Classification layer
def classify(layer_2):
    y_est = tf.nn.softmax(tf.matmul(layer_2, weights['classify_h3']) + biases['classify_b3'])
    return y_est

def KmeansObj(hidden, M):

    diff = hidden - M
    return tf.reduce_mean(tf.reduce_sum(tf.pow(diff, 2), 1))

def classificationEvaluation(y_est, y_true):
    pred_temp = tf.equal(tf.argmax(y_est, 1), tf.argmax(y_true, 1))
    classification_accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    classification_error = 1 - classification_accuracy
    return classification_accuracy, classification_error


hidden = encoder(X1, X2)  # for infer
y_est = classify(hidden)

Kmeans_obj = KmeansObj(hidden, M)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_est + 1e-16), reduction_indices=[1]))

l1_regularizer = tf.contrib.layers.l1_regularizer(scale=1.0, scope=None)
regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, {weights['encoder_h1_v1'], weights['encoder_h1_v2']})

l21_1 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(weights['encoder_h1_v1'], 2), 1)), 0)

l21_2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(weights['encoder_h1_v2'], 2), 1)), 0)

#regularization_penalty = l21_1 + l21_2

loss_pretrain = cross_entropy +  lambda1*regularization_penalty  # loss function in pretrain


optimizer_pretrain = tf.train.AdamOptimizer(learning_rate).minimize(loss_pretrain)


classification_accuracy, classification_error = classificationEvaluation(y_est, Y)


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()


# Start Training
# Start a new TF session
cr_total = list()
error_total = list()
cr_mean = list()
error_mean = list()
cr_std = list()
error_std = list()


for lambda1_this in lambda1_list:
    cr_this = list()
    error_this = list()

    print(lambda1_this)
    for k in range(n_fold):
        print(k)
        train_data1, train_data2, train_targets, val_data1, val_data2, val_targets, batch_size_val = \
            make_train_validation(data1_brca_train, data2_brca_train, targets_brca_train, k, n_fold)

        with tf.Session() as sess:
            print('Begin pretraining')

        # Run the initializer
            sess.run(init)

        # Training
            i = 1
            while i <= num_pretrain:
                i += 1

        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        #batch_x, batch_y = mnist.train.next_batch(batch_size_pretrain) # load MNIST data
                batch_x1,batch_x2, batch_y = get_next_batch_2View(train_data1, train_data2, train_targets, batch_size_pretrain, i)

        # Run optimization op (backprop) and cost op (to get loss value)
                _, l, cr = sess.run([optimizer_pretrain, loss_pretrain, cross_entropy],
                                feed_dict={X1: batch_x1, X2: batch_x2,  Y: batch_y,  lambda1: lambda1_this})
        # Display logs per step
                if i % display_step == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f, cross entropy: %f' % (i, l, cr))

                    l, cr, error = sess.run([loss_pretrain, cross_entropy, classification_error],
                    feed_dict={X1: val_data1, X2: val_data2, Y: val_targets, lambda1: lambda1_this})

                    print('Test Loss: %f, cross entropy: %f, classification error: %f ' % (l, cr, error))



            l, cr, error, weights_output, biases_output= sess.run([loss_pretrain, cross_entropy, classification_error, weights, biases],
                    feed_dict={X1: val_data1, X2: val_data1,  Y: val_targets, lambda1: lambda1_this, lambda2: 0.0})
        # print the performance on validation set
            print('lambda1 = ' +str(lambda1_this) + ', Classification error on Test: %f \n' % ( error))



        cr_this.append(cr)
        error_this.append(error)


    cr_total.append(cr_this)
    error_total.append(error_this)
    cr_mean.append(np.mean(np.array(cr_this)))
    cr_std.append(np.std(np.array(cr_this)))
    error_mean.append(np.mean(np.array(error_this)))
    error_std.append(np.std(np.array(error_this)))



# select optimal parameter
# select the smallest error
error_min = min(error_mean)
_,id_min = min( (error_mean[i],i) for i in xrange(len(error_mean)) )

print(id_min)
std_min = error_std[id_min]
id_selected = (error_mean <= error_min + std_min).nonzero()
id_selected = id_selected[0]
print('id_selected:', id_selected)
lambda1_selected = [ lambda1_list[i] for i in id_selected]
lambda1_optimal = max(lambda1_selected)

sio.savemat(result_dir+'trainAlpha.mat', {'cr_total': cr_total, 'error_total': error_total, 'cr_mean': cr_mean,
                                          'cr_std': cr_std, 'error_mean': error_mean, 'error_std': error_std,
                                          'lambda1_optimal': lambda1_optimal, 'lambda1_list': lambda1_list})


# train the model using lambda1_optimal

with tf.Session() as sess:
    print('Parameter Selected, Begin pretraining')

    sess.run(init)

    # Training
    i = 1
    while i <= num_pretrain:
        i += 1
        batch_x1, batch_x2, batch_y = get_next_batch_2View(data1_brca_train, data2_brca_train, targets_brca_train, batch_size_pretrain, i)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l, cr = sess.run([optimizer_pretrain, loss_pretrain, cross_entropy],
                feed_dict={X1: batch_x1, X2: batch_x2,  Y: batch_y, lambda1: lambda1_optimal})

    weights_pretrain, bias_pretrain, hidden_pretrain = sess.run([weights, biases, hidden], feed_dict = {X1: data1_brca, X2: data1_brca})
    types = ('Basal', 'Her2+', 'LumA', 'LumB', 'Normal like', 'Normal') # visualize by original labels
    label =	np.nonzero(targets_brca == 1)[1]
    title = result_dir+'Trained_Whole'
    X_embedded_whole, X_PCA_whole = Transfer_TSNE_PCA(hidden_pretrain, 2, 3)
    print('here')
    Visualize_TSNE_PCA(X_embedded_whole, X_PCA_whole, types, label, title, 0)

    path_model = result_dir + 'tmp/model/'

    if not os.path.exists(path_model):
        os.makedirs(path_model)

    saver.save(sess, path_model+'pretrained_model')
path_mat = result_dir + 'tmp/mat/'

if not os.path.exists(path_mat):
    os.makedirs(path_mat)

sio.savemat(path_mat+'optimal.mat', {'cr_total': cr_total, 'error_total': error_total, 'lambda1_list': lambda1_list,'lambda1_optimal': lambda1_optimal,
                    'weights_pretrain': weights_pretrain, 'bias_pretrain': bias_pretrain, 'hidden_pretrain': hidden_pretrain,
                    'X_embedded_whole': X_embedded_whole, 'X_PCA_whole': X_PCA_whole, 'index': index, 'batch_size_test': batch_size_test})












