from __future__ import division, print_function, absolute_import


from sklearn import cluster
from scipy.spatial import distance

import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.metrics import silhouette_score
import gap as gap
import sys



def SelectK(X, k_list, method, dir):
    if method.lower() == 'bic':
        KMeans = [cluster.KMeans(n_clusters = k, init="k-means++").fit(X) for k in k_list]

        BIC = [compute_bic(kmeansi,X) for kmeansi in KMeans]
        index, value = max(enumerate(BIC), key=operator.itemgetter(1))
        kmeans_selected = KMeans[index]
        K_selected = k_list[index]
        Parameter = BIC
        #fig = plt.figure()
        #plt.plot(k_list,BIC,'r-o')
        #fig.savefig(dir + 'BIC', bbox_inches='tight')

    elif method.lower() == 'silhouette':
        KMeans = [cluster.KMeans(n_clusters = k, init="k-means++").fit(X) for k in k_list]
        s = [compute_silhouette(kmeansi,X) for kmeansi in KMeans]
        index, value = max(enumerate(s), key=operator.itemgetter(1))
        kmeans_selected = KMeans[index]
        K_selected = k_list[index]
        Parameter = s
        #fig = plt.figure()
        #plt.plot(k_list,s,'r-o')
        #fig.savefig(dir + 'Average silhouette width', bbox_inches='tight')

    elif method.lower() == 'gap':
        gaps, s_k, K = gap.gap_statistic(X, refs=None, B=10, K= k_list, N_init = 10)
        bestKValue = gap.find_optimal_k(gaps, s_k, K)
        K_selected = bestKValue
        kmeans_selected = cluster.KMeans(n_clusters = K_selected, init="k-means++").fit(X)
        Parameter = gaps

#fig = plt.figure()
#       plt.plot(k_list,gaps,'r-o')
#       fig.savefig(dir + 'Gap statistic', bbox_inches='tight')

    else:
        sys.exit('Undefined Method!')


    return kmeans_selected, K_selected, Parameter



def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------

    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_

    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]],
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term


    return(BIC)


def compute_silhouette(kmeans, X):
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    silhouetteScore = silhouette_score(X, labels, metric='euclidean')

    return silhouetteScore









