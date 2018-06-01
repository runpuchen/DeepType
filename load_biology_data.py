import numpy as np
import scipy.io as sio

def load_biology_data(filename, n_test, normalize):
    matContents = sio.loadmat(filename)
    data = matContents['data']
    [n_dim, n_sample] = data.shape
    if normalize == 1:
        for i in range(n_dim):
            m1 = min(data[i,:])
            m2 = max(data[i,:])
            data[i,:] =( data[i,:] - m1)/(m2 - m1)
    targets = matContents['targets']
    index = np.random.permutation(n_sample)
    data = data[:, index]
    targets = targets[index]
    n_label = len(np.unique(targets))
    Y = np.zeros([n_sample, n_label])
    targets = targets - 1
    for target in np.unique(targets):
        id = (targets == target).nonzero()[0]
        Y[id, target] = 1
    X = data.T
    X_train = X[0:n_sample - n_test, :]
    Y_train = Y[0:n_sample - n_test, :]
    X_test = X[n_sample - n_test:n_sample, :]
    Y_test = Y[n_sample - n_test:n_sample, :]

    return X_train, Y_train, X_test, Y_test, index



def get_next_batch(X, Y, batch_size, i_batch):
    [n_sample, n_dim] = X.shape
    start_id = (i_batch-1) * batch_size
    id_batch = range(start_id, start_id+batch_size)
    id_batch = np.asarray(id_batch) % n_sample
    x_batch = X[id_batch, :]
    y_batch = Y[id_batch, :]
    return x_batch, y_batch


def load_biology_data_2View(filename, n_test, normalize, num_gene):
    matContents = sio.loadmat(filename)
    data1 = matContents['data1']
    data2 = matContents['data2']
    data1 = data1[range(num_gene), :]
    data2 = data2[range(num_gene), :]
    [n_dim1, n_sample] = data1.shape
    [n_dim2, _] = data2.shape
    if normalize == 1:
        for i in range(n_dim1):
            m1 = min(data1[i,:])
            m2 = max(data1[i,:])
            data1[i,:] =( data1[i,:] - m1)/(m2 - m1)
        for i in range(n_dim2):
            m1 = min(data2[i,:])
            m2 = max(data2[i,:])
            data2[i,:] =( data2[i,:] - m1)/(m2 - m1)





    targets = matContents['targets']

    index = np.random.permutation(n_sample)
    data1 = data1[:, index]
    data2 = data2[:, index]
    targets = targets[index]
    n_label = len(np.unique(targets))
    Y = np.zeros([n_sample, n_label])
    targets = targets - 1
    for target in np.unique(targets):
        id = (targets == target).nonzero()[0]
        Y[id, target] = 1
    X1 = data1.T
    X2 = data2.T
    X1_train = X1[0:n_sample - n_test, :]
    X2_train = X2[0:n_sample - n_test, :]
    Y_train = Y[0:n_sample - n_test, :]
    X1_test = X1[n_sample - n_test:n_sample, :]
    X2_test = X2[n_sample - n_test:n_sample, :]
    Y_test = Y[n_sample - n_test:n_sample, :]

    return X1_train, X2_train, Y_train, X1_test, X2_test, Y_test, index

def load_biology_data_3View(filename, n_test, normalize):
    matContents = sio.loadmat(filename)
    data1 = matContents['data1']
    data2 = matContents['data2']
    data3 = matContents['data3']
    [n_dim1, n_sample] = data1.shape
    [n_dim2, _] = data2.shape
    if normalize == 1:
        for i in range(n_dim1):
            m1 = min(data1[i,:])
            m2 = max(data1[i,:])
            data1[i,:] =( data1[i,:] - m1)/(m2 - m1)
        for i in range(n_dim2):
            m1 = min(data2[i,:])
            m2 = max(data2[i,:])
            data2[i,:] =( data2[i,:] - m1)/(m2 - m1)
    targets = matContents['targets']

    index = np.random.permutation(n_sample)
    data1 = data1[:, index]
    data2 = data2[:, index]
    data3 = data3[:, index]
    targets = targets[index]
    n_label = len(np.unique(targets))
    Y = np.zeros([n_sample, n_label])
    targets = targets - 1
    for target in np.unique(targets):
        id = (targets == target).nonzero()[0]
        Y[id, target] = 1
    X1 = data1.T
    X2 = data2.T
    X3 = data3.T
    X1_train = X1[0:n_sample - n_test, :]
    X2_train = X2[0:n_sample - n_test, :]
    X3_train = X3[0:n_sample - n_test, :]
    Y_train = Y[0:n_sample - n_test, :]
    X1_test = X1[n_sample - n_test:n_sample, :]
    X2_test = X2[n_sample - n_test:n_sample, :]
    X3_test = X3[n_sample - n_test:n_sample, :]
    Y_test = Y[n_sample - n_test:n_sample, :]

    return X1_train, X2_train, X3_train, Y_train, X1_test, X2_test, X3_test, Y_test, index



def get_next_batch_2View(X1,X2, Y, batch_size, i_batch):
    [n_sample, n_dim] = X1.shape
    start_id = (i_batch-1) * batch_size
    id_batch = range(start_id, start_id+batch_size)
    id_batch = np.asarray(id_batch) % n_sample
    x1_batch = X1[id_batch, :]
    x2_batch = X2[id_batch, :]
    y_batch = Y[id_batch, :]
    return x1_batch, x2_batch, y_batch

def get_next_batch_3View(X1,X2,X3, Y, batch_size, i_batch):
    [n_sample, n_dim] = X1.shape
    start_id = (i_batch-1) * batch_size
    id_batch = range(start_id, start_id+batch_size)
    id_batch = np.asarray(id_batch) % n_sample
    x1_batch = X1[id_batch, :]
    x2_batch = X2[id_batch, :]
    x3_batch = X3[id_batch, :]
    y_batch = Y[id_batch, :]
    return x1_batch, x2_batch, x3_batch, y_batch
