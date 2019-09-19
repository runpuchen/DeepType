""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import mpl_toolkits.mplot3d.axes3d as p3
import tensorflow as tf

from sklearn import cluster

import operator
from sklearn.metrics import silhouette_score
from model import *


def VisualizeHidden(embed_tsne, embed_pca, label, types, title):

    embed_tsne = (embed_tsne - min(embed_tsne[:,0]))/(max(embed_tsne[:,0]) - min(embed_tsne[:,0]))
    embed_pca = (embed_pca - min(embed_pca[:, 0]))/(max(embed_pca[:, 0]) - min(embed_pca[:, 0]))
    label_unique = np.unique(label)
    colors = cm.Paired(np.linspace(0, 1, len(label_unique)))
    fig = plt.figure()
    plot_time = 0
    for this_label, c, this_type in zip(label_unique, colors, types):
        id = (label == this_label).nonzero()
        plt.scatter(embed_tsne[id, 0], embed_tsne[id, 1], color = c, label= str(this_type))
        plot_time += 1
    lgd = plt.legend(loc = 'upper left',  scatterpoints=1)
    plt.xlim((-0.1,1))

    fig.savefig(title + '_TSNE.png', bbox_extra_artists=(lgd,), bbox_inches='tight')


    fig = plt.figure()
    ax = p3.Axes3D(fig)
    plot_time = 0
    for this_label, c, this_type in zip(label_unique, colors, types):
        id = (label == this_label).nonzero()
        ax.scatter(embed_pca[id, 0], embed_pca[id, 1],embed_pca[id, 2], color = c, label= str(this_type))
        plot_time += 1
    lgd = plt.legend(loc = 'upper left', bbox_to_anchor=(1.01, 1), scatterpoints=1)
    plt.xlim((-0.1,1))


    fig.savefig(title + '_PCA.png', bbox_extra_artists=(lgd,), bbox_inches='tight')



    return embed_tsne, embed_pca

def Transfer_TSNE_PCA(X, dim_tsne, dim_pca):
    X_embedded = TSNE(n_components=dim_tsne).fit_transform(X)
    pca = PCA(n_components=dim_pca)
    pca.fit(X)
    X_PCA = pca.transform(X)
    return X_embedded, X_PCA

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def cross_validation_param_selection(error_list, para_list, std_list):

    error_min = min(error_list)
    _,id_min = min( (error_list[i],i) for i in xrange(len(error_list)))

    print(id_min)
    std_min = std_list[id_min]
    #std_min = 0.05
    id_selected = (error_list <= error_min + std_min).nonzero()
    print(id_selected)
    id_selected = id_selected[0]
    print('id_selected:', id_selected)
    para_optimal = [ para_list[i] for i in id_selected]
    para_optimal = max(para_optimal)
    print(para_optimal)

    return para_optimal


def SelectK(X, k_list, method, dir):

    KMeans = [cluster.KMeans(n_clusters = k, init="k-means++").fit(X) for k in k_list]
    s = [compute_silhouette(kmeansi,X) for kmeansi in KMeans]
    index, value = max(enumerate(s), key=operator.itemgetter(1))
    kmeans_selected = KMeans[index]
    K_selected = k_list[index]
    Parameter = s

    return kmeans_selected, K_selected, Parameter




def compute_silhouette(kmeans, X):
    kmeans.fit(X)

    labels = kmeans.labels_

    silhouetteScore = silhouette_score(X, labels, metric='euclidean')

    return silhouetteScore



