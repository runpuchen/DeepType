from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from load_biology_data import *
import pickle
from Visualize import Visualize_TSNE_PCA, Transfer_TSNE_PCA
from KmeansParaSelection import *
import os
import sys

key = int(sys.argv[1])
num_gene = 2000*key
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



np.random.seed(0)
tf.set_random_seed(0)

# Training Parameters
learning_rate = 0.002


num_train = 50
num_train_epoch = 100

batch_size_train = 500
batch_size_test = 131

num_groups =11 # brca, this can change. MNIST: 15
num_labels = 5 # brca. MNIST: 10
k_list = list([  9, 10, 11, 12, 13, 14, 15] )# this can change

lambda0 = 1.0




# lambda1 = pretrain_results['lambda1_optimal']

#lambda2 = 1e-4/lambda0 # penalty of sparsity
lambda2_list = np.logspace(0, -5, num=5, endpoint=True)
n_fold = 5

print(lambda2_list)

dropout = 0.2
r_bound = 1+0.1


display_step = num_train

# Network Parameters
num_hidden_1 = 1024 # 1st layer num features
num_hidden_2 = 512 # 2nd layer num features (the latent dim)
num_input = 2000 # MNIST data input (img shape: 28*28)

show_choice = 0
normalize = 1

method = 'silhouette'

# Import biology data

filename = local_dir+ '/BRCA/2View/' + 'BRCA2View' + str(18000) + '.mat'
result_dir = local_dir+ '/BRCA/results/2View/' + str(num_gene) + '/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

pretrained_file = result_dir + 'trainAlpha.mat'
content = sio.loadmat(pretrained_file)
lambda1_trained =content['lambda1_optimal']
lambda1_trained = lambda1_trained[0][0]
print('lambda1_trained: ', lambda1_trained)

#lambda1_trained = 0.07

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
regularization_penalty = lambda1*tf.contrib.layers.apply_regularization(l1_regularizer, {weights['encoder_h1_v1'], weights['encoder_h1_v2']})

l21_1 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(weights['encoder_h1_v1'], 2), 1)), 0)

l21_2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(weights['encoder_h1_v2'], 2), 1)), 0)

#regularization_penalty = l21_1 + l21_2

loss_train = cross_entropy + lambda2*Kmeans_obj + regularization_penalty

optimizer_train = tf.train.AdamOptimizer(learning_rate).minimize(loss_train)

classification_accuracy, classification_error = classificationEvaluation(y_est, Y)


init = tf.global_variables_initializer()
cr_total = list()
weights_total = list()
biases_total = list()
error_total = list()
pretrained_model_path = result_dir + 'tmp/model/'
pretrained_model_file = pretrained_model_path + 'pretrained_model.meta'

with tf.Session() as sess:
    sess.run(init)

    saver = tf.train.import_meta_graph(pretrained_model_file)
    saver.restore(sess, tf.train.latest_checkpoint(pretrained_model_path))
    print('Pretrained model restored')

    # infer the hidden

    print('Begin training...')
    total_M = np.zeros([num_train_sample, num_hidden_2])
    h = hidden.eval(feed_dict={X1: data1_brca_train, X2: data2_brca_train})
    [kmeans_pretrained, num_groups_initial, s] = SelectK(h, k_list, method, result_dir)
    print('Initial selected number of clusters: %d' %(num_groups_initial))
    # obtain the clustering results on pretraining manifold

N_group_list = range(num_groups_initial, num_groups_initial+5)
optimal_para_pair = dict.fromkeys(N_group_list, 0)
s_k = list()

for num_groups in N_group_list:
    print('num of groups:', num_groups)
    # given each num_groups, tune lambda2
    cr_total = list()
    error_total = list()
    cr_mean = list()
    error_mean = list()
    cr_std = list()
    error_std = list()

    for lambda2_this in lambda2_list:
        print(lambda2_this)
        cr_this = list()
        error_this = list()

        # given each num_groups and lambda2, perform 10-fold CR
        for k in range(n_fold):
            # make the 10-fold data
            print(k)
            train_data1, train_data2, train_targets, val_data1, val_data2, val_targets, batch_size_val = \
            make_train_validation(data1_brca_train, data2_brca_train, targets_brca_train, k, n_fold)

            with tf.Session() as sess:

            # load the pretraining
                sess.run(init)
                saver = tf.train.import_meta_graph(pretrained_model_file)
                saver.restore(sess, tf.train.latest_checkpoint(pretrained_model_path))
                print('Pretrained model restored')

                for train_epoch in range(num_train_epoch):

                    print('Epoch: %i \n' % (train_epoch))
                    # fix M and s, train W and b
                    # obtain M and s

                    h = hidden.eval(feed_dict={X1: data1_brca_train, X2: data2_brca_train})

                    kmeans = KMeans(n_clusters=num_groups, init = "k-means++", max_iter=50, tol=0.01).fit(h)
                    ass = kmeans.labels_
                    centers = kmeans.cluster_centers_
                    A = np.zeros([num_train_sample, num_groups])
                    for i_sample in range(0, num_train_sample):
                        A[i_sample, ass[i_sample]] = 1
                    total_M = np.dot(A, centers)

                    for i in range(1, num_train+1):
                        batch_x1,batch_x2, batch_y = \
                        get_next_batch_2View(train_data1, train_data2, train_targets, batch_size_train, i) # load BRCA data
                        batch_M, _ = get_next_batch(total_M, A, batch_size_train, i)

                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, l, p = sess.run([optimizer_train, loss_train, regularization_penalty],
                            feed_dict={X1: batch_x1, X2: batch_x2, Y: batch_y, M: batch_M, lambda2: lambda2_this, lambda1: lambda1_trained})
                        # Display logs per step
                        if i % display_step == 0 or i == 1:
                            print('Step %i: Minibatch Loss: %f penalty: %f' % (i, l, p))
                            h = hidden.eval(feed_dict={X1: val_data1, X2: val_data2})
                            # obtain the assignment on the h
                            ass_val = kmeans.predict(h)
                            A_val = np.zeros([batch_size_val, num_groups])
                            for i_sample in range(0, batch_size_val):
                                A_val[i_sample, ass_val[i_sample]] = 1
                            batch_M = np.dot(A_val, centers)

                            l , cr, error_on_test = sess.run([loss_train, cross_entropy, classification_error],
                            feed_dict={X1: val_data1, X2: val_data2, Y: val_targets, M: batch_M, lambda2: lambda2_this, lambda1: lambda1_trained})
                            print('Test Loss: %f, cross entropy: %f, classification error: %f \n' % (l, cr, error_on_test))

                h = hidden.eval(feed_dict={X1: val_data1, X2: val_data2})
                # obtain the assignment on the h
                ass_val = kmeans.predict(h)
                A_val = np.zeros([batch_size_val, num_groups])
                for i_sample in range(0, batch_size_val):
                    A_val[i_sample, ass_val[i_sample]] = 1
                batch_M = np.dot(A_val, centers)
                # training end, obtain the classification error
                l, cr, error, weights_output, biases_output= sess.run([loss_train, cross_entropy, classification_error, weights, biases],
                        feed_dict={X1: val_data1, X2: val_data1,  Y: val_targets,M: batch_M, lambda2: lambda2_this, lambda1: lambda1_trained})
            # print the performance on validation set
                print('Lambda2 is: %f, error on validation: %f \n' % (lambda2_this, error))

            cr_this.append(cr)
            error_this.append(error)


        cr_total.append(cr_this)
        error_total.append(error_this)
        cr_mean.append(np.mean(np.array(cr_this)))
        cr_std.append(np.std(np.array(cr_this)))
        error_mean.append(np.mean(np.array(error_this)))
        error_std.append(np.std(np.array(error_this)))

        # find the optimal lambda2 for this num_group

    mat_path = result_dir + 'tmp/mat/' + str(num_groups) + '/'
    error_optimal = min(x for x in error_mean)
    error_bound = error_optimal * r_bound
    id_selected = (error_mean <= error_bound).nonzero()
    id_selected = id_selected[0]
    lambda2_selected = [ lambda2_list[i] for i in id_selected]
    lambda2_optimal = max(lambda2_selected)
    optimal_para_pair[num_groups] = lambda2_optimal
    sio.savemat(mat_path + 'trainingLambda2.mat',
    {'cr_total': cr_total, 'error_total': error_total, 'cr_mean': cr_mean, 'error_mean': error_mean,
     'cr_std': cr_std, 'error_std':error_std, 'lambda2_list': lambda2_list, 'lambda2_optimal': lambda2_optimal})



        # train on the selected lambda2. Open a session again, train from pretraining results

    with tf.Session() as sess:
        # load from pretraining
        sess.run(init)
        saver = tf.train.import_meta_graph(pretrained_model_file)
        saver.restore(sess, tf.train.latest_checkpoint(pretrained_model_path))

        print('Optimal parameter selected, begin training...')
        total_M = np.zeros([num_train_sample, num_hidden_2])
        h = hidden.eval(feed_dict={X1: data1_brca_train, X2: data2_brca_train}) # infer the hidden
        s = 0
        kmeans = KMeans(n_clusters=num_groups, init = "k-means++", max_iter=50, tol=0.01).fit(h)

        for train_epoch in range(num_train_epoch):

            print('Epoch: %i \n' % (train_epoch))
            # fix M and s, train W and b
            # obtain M and s
            if train_epoch > 0:
                h = hidden.eval(feed_dict={X1: data1_brca_train, X2: data2_brca_train})

            kmeans = KMeans(n_clusters=num_groups, init = "k-means++", max_iter=50, tol=0.01).fit(h)
            ass = kmeans.labels_
            centers = kmeans.cluster_centers_
            A = np.zeros([num_train_sample, num_groups])
            for i_sample in range(0, num_train_sample):
                A[i_sample, ass[i_sample]] = 1
            total_M = np.dot(A, centers)

            for i in range(1, num_train+1):
                batch_x1,batch_x2, batch_y = \
                    get_next_batch_2View(data1_brca_train, data2_brca_train, targets_brca_train, batch_size_train, i) # load BRCA data
                batch_M, _ = get_next_batch(total_M, A, batch_size_train, i)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, l = sess.run([optimizer_train, loss_train],
                        feed_dict={X1: batch_x1, X2: batch_x1,Y: batch_y, M: batch_M, lambda2: lambda2_optimal, lambda1: lambda1_trained})


        weights_output, biases_output, hidden_output = sess.run([weights, biases, hidden],feed_dict={X1: data1_brca, X2: data2_brca,Y: targets_brca})
        ass_total = kmeans.predict(hidden_output)
        # save session
        sess_path = result_dir + 'tmp/model/' + str(num_groups)+ '/'
        if not os.path.exists(sess_path):
            os.makedirs(sess_path)

        saver.save(sess, sess_path+'trained_model')

        mat_path = result_dir + 'tmp/mat/' + str(num_groups) + '/'
        if not os.path.exists(mat_path):
            os.makedirs(mat_path)

        X_embedded_whole, X_PCA_whole = Transfer_TSNE_PCA(hidden_output, 2, 3)


        types = ('Basal', 'Her2+', 'LumA', 'LumB', 'Normal like', 'Normal') # visualize by original labels
        label =	np.nonzero(targets_brca == 1)[1]
        title = mat_path+'Trained_Whole'
'''

        Visualize_TSNE_PCA(X_embedded_whole, X_PCA_whole, types, label, title, show_choice)

        types = range(num_groups)

        title = mat_path+'Trained_Clustered_Whole'
        Visualize_TSNE_PCA(X_embedded_whole, X_PCA_whole, types, ass_total, title, show_choice)


        types = range(num_groups)
        title = mat_path+'Trained_Clustered_Whole'
        Visualize_TSNE_PCA(X_embedded_whole, X_PCA_whole, types, ass_total, title, show_choice)
'''
        sio.savemat(mat_path + 'encoder.mat', {'weights_output':weights_output, 'biases_output': biases_output, # save model parameters
        'silhouette': s, 'hidden_output': hidden_output,'centers': centers, 'index_total': index,
        'X_embedded_whole': X_embedded_whole, 'X_PCA_whole': X_PCA_whole, 'ass_total': ass_total})

    if method.lower() == 'bic':
        s_k.append(compute_bic(kmeans,hidden_output))
    elif method.lower() == 'silhouette':
        s_k.append(compute_silhouette(kmeans, hidden_output))
    elif method.lower() == 'gap':
        tmp, _, _ = gap.gap_statistic(hidden_output, refs=None, B=10, K= num_groups, N_init = 10)
        s_k.append(tmp)
    else:
        sys.exit('Undefined Method!')


sio.savemat(result_dir + 'parameter.mat', {'s_k': s_k, 'N_group_list': N_group_list})
#fig = plt.figure()
#plt.plot(N_group_list,s_k,'r-o')
#fig.savefig(result_dir + method, bbox_inches='tight')












