from sklearn.cluster import KMeans

from model import *
import tensorflow as tf
from data import *
from eval import loss_supervised, evaluation, \
                 loss_supervised_unsupervised, do_get_hidden, do_validation
from utils import *
from collections import deque
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'




def training(loss, learning_rate, loss_key=None):
    """

    :param loss: Loss tensor, from loss()
    :param learning_rate: The learning rate to use for gradient descent
    :param loss_key: int giving stage of pretraining so we can store
                loss summaries for each pretraining stage
    :return train_op: The Op for training.
    """
    loss_summaries = {}
    if loss_key is not None:
        # Add a scalar summary for the snapshot loss.
        loss_summaries[loss_key] = tf.summary.scalar(loss.op.name, loss)
    else:
        tf.summary.scalar(loss.op.name, loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize()
    train_op = optimizer(loss)


    return train_op


def main_supervised_1view(FLAGS):
    """
    Perform supervised training with sparsity penalty.
    :return acc: the accuracy on trainng set.
    :return ae_supervised: the trained autoencoder.
    :return sess: the current session.
    """
    print('Supervised training...')
    data = read_data_sets(FLAGS)

    # combine training and validation

    do_val = True

    sess, acc, ae_supervised = supervised_1view(data, FLAGS, do_val)


    # get the manifold
    data_whole = np.concatenate((data.train.data, data.validation.data, data.test.data), axis = 0)
    target_whole = np.concatenate((data.train.labels, data.validation.labels, data.test.labels), axis = 0)
    data_sets_whole = DataSet(data_whole, target_whole)

    if FLAGS.visualization:
        # supervised training ends, visualize
        with sess.graph.as_default():
            initialize_uninitialized(sess)
            last_layer_final_train = do_get_hidden(sess, ae_supervised,  data_sets_whole, FLAGS.num_hidden_layers, FLAGS)
        X_TSNE_trained, X_PCA_trained = Transfer_TSNE_PCA(last_layer_final_train, 2, 3)
        labels = np.nonzero(target_whole == 1)[1]
        types = np.unique(labels)
        title = FLAGS.results_dir + 'SupervisedTrained'
        VisualizeHidden(X_TSNE_trained, X_PCA_trained, labels, types, title)
    else: pass
    

    return acc, ae_supervised, sess

def supervised_1view(data, FLAGS, do_val = True):
    """
    :param data: input data
    :param do_val: whether do validation.
    :return sess: the current session.
    :return ae: the trained autoencoder.
    :return acc: the accuracy on validation/training batch.
    """
    if FLAGS.initialize == True:
        file_dir = FLAGS.data_dir + 'initialize_encoder.mat'
        matContents = sio.loadmat(file_dir)
        AE_initialize = matContents


    sess = tf.Session()
    acc_record = deque([])
    N_record = 5
    with sess.graph.as_default():
        # here we need to perform the parameter selection


        ae_shape = FLAGS.NN_dims_1

        ae = AutoEncoder(ae_shape)


        if FLAGS.initialize:
            ae._setup_variables(FLAGS.initialize, AE_initialize)

        input_pl = tf.placeholder(tf.float32, shape=(None, FLAGS.dimension), name = 'input_pl')

        logits = ae.supervised_net(input_pl, FLAGS.num_hidden_layers+1)

        labels_placeholder = tf.placeholder(tf.float32,
                                        shape=(None, FLAGS.num_classes), name = 'target_pl')
        alpha_placeholder = tf.placeholder(tf.float32, None, name = 'alpha_pl')

        loss, sp, cr = loss_supervised(logits, labels_placeholder, ae, alpha_placeholder)
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())

        for step in range(FLAGS.supervised_train_steps): # gradually increase alpha, for fast convergence
            alpha = 0. + FLAGS.alpha * (1 - np.exp(-(step+0.)/300.))
            if alpha >= FLAGS.alpha * (1.0-1e-2):
                alpha = FLAGS.alpha
            input_feed, target_feed = data.train.next_batch(FLAGS.batch_size)
            feed_dict_supervised = {input_pl: input_feed,
                                    labels_placeholder: target_feed, alpha_placeholder: alpha}


            accuracy, est = evaluation(logits, labels_placeholder)
            sess.run(train_op, feed_dict=feed_dict_supervised)


            acc_train, loss_value, penalty, cr_value, estimation = \
                sess.run([accuracy, loss, sp, cr, est], feed_dict=feed_dict_supervised)

            # Print training process.
            if (step+1) % FLAGS.display_steps == 0 or step+1 == FLAGS.supervised_train_steps or step == 0:
                print(alpha)
                output ='Train step '+str(step+1)+' minibatch loss: '+ str(loss_value) + ' penalty:' + str(penalty)+ \
                         ' accuracy: ' + str(acc_train)
                print(output)
                acc = acc_train
                if do_val:
                    if do_val:
                        acc_val, _ = do_validation(sess, ae, data.validation, FLAGS)
                else: acc_val = acc_train

                acc_record.append(acc_val)
                if len(acc_record) > N_record: acc_record.popleft()
                else: pass

                acc_val_show = max(acc_record)
                output = 'accuracy on validation: ' + str(acc_val_show)
                print(output)
                acc = acc_val


    return sess, acc, ae


def main_supervised_unsupervised_1view(ae, sess, FLAGS):
    """
    :param ae: the autoencoder to be trained.
    :param sess: the current session.
    :return ae_final: the final trained autoencoder.
    """

    print('Supervised training ends, begin supervised-unsupervised training...')
    data = read_data_sets(FLAGS)
    num_hidden_layers = FLAGS.num_hidden_layers
    with sess.graph.as_default():
        initialize_uninitialized(sess)
        last_layer_train = do_get_hidden(sess, ae,  data.train, num_hidden_layers, FLAGS)

        # perform Kmeans
        kmeans = KMeans(n_clusters=FLAGS.num_clusters, init='k-means++', max_iter=50, tol=0.01).fit(last_layer_train)
        # get the Kmeans, make a new sess
        do_val = True
    sess, acc_final, ae_final, kmeans = supervised_unsupervised_1view(sess, ae, kmeans, data, FLAGS, do_val)


    # supervised-unsupervised training ends, visualize

    print('Supervised-unsupervised training ends, visualize...')
    # visualize
    # get the hidden layer
    # combine train and validation and do the training
    data_whole = np.concatenate((data.train.data, data.validation.data), axis = 0)
    target_whole = np.concatenate((data.train.labels, data.validation.labels), axis = 0)
    data_set_train = DataSet(data_whole, target_whole)

    with sess.graph.as_default():
        initialize_uninitialized(sess)
        last_layer_final_train = do_get_hidden(sess, ae,  data_set_train, FLAGS.num_hidden_layers, FLAGS)
    labels = kmeans.predict(last_layer_final_train)
    title = FLAGS.results_dir + 'FinalTrainedClustered'
    types = np.unique(labels)
    X_TSNE_trained, X_PCA_trained = Transfer_TSNE_PCA(last_layer_final_train, 2, 3)
    VisualizeHidden(X_TSNE_trained, X_PCA_trained, labels, types, title)
    labels = np.nonzero(target_whole == 1)[1]
    types = np.unique(labels)
    title = FLAGS.results_dir + 'FinalTrained'
    VisualizeHidden(X_TSNE_trained, X_PCA_trained, labels, types, title)


    return sess, ae, kmeans, acc_final


def supervised_unsupervised_1view(sess, ae,  kmeans, data, FLAGS, do_validation_flag = True):
    '''
    :param ae: the autoencoder to be trained.
    :param kmeans: the kmeans object indicating the cluster centers and assignments.
    :param data: DataSet object.
    :param do_validation_flag: whether to do validation.
    :return sess: the current session.
    :return acc: the accuracy on validation set/training batch.
    :return ae: the trained autoencoder.
    :return kmeans: the trained kmeans object.
    '''

    num_train = data.train.num_examples

    num_hidden_layers = FLAGS.num_hidden_layers
    num_training = 100
    train_epochs = int(np.floor(FLAGS.train_steps/num_training))
    acc_record = deque([])
    N_record = 5


    with sess.graph.as_default():
        input_batch_pl = tf.placeholder(tf.float32, shape=(None, FLAGS.dimension),name='input_pl')
        input_center_pl = tf.placeholder(tf.float32, shape=(None, FLAGS.hidden_layer_dim),
                                             name='input_center')
        input_target_pl = tf.placeholder(tf.float32, shape=(None, FLAGS.num_classes),name='input_target')

        logits = ae.supervised_net(input_batch_pl, num_hidden_layers+1)
        accuracy, _ = evaluation(logits, input_target_pl)
        loss, kmeans_loss, sp, cr = loss_supervised_unsupervised(ae, logits, input_target_pl,
                            ae.supervised_net(input_batch_pl, FLAGS.num_hidden_layers),
                            input_center_pl, FLAGS)

        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

        initialize_uninitialized(sess)



        for epochs in range(0, train_epochs):
            print('\n' +'Training circle: ' + str(epochs))

            if epochs == 0:
                pass
            else:
                last_layer_train = do_get_hidden(sess, ae,  data.train, num_hidden_layers, FLAGS)
                kmeans = KMeans(n_clusters=FLAGS.num_clusters,init='k-means++', max_iter=50, tol=0.01).fit(last_layer_train)

            centers = kmeans.cluster_centers_

            # convert centers to center matrix
            assignment = kmeans.labels_


            ASS = np.zeros([num_train, FLAGS.num_clusters])


            for i in range(num_train):
                ASS[i, assignment[i]] = 1
            center_matrix = np.dot(ASS, centers)
            center_set = DataSet(center_matrix, ASS)
            center_set._index_in_epoch = data.train.start_index

            input_feed, target_feed = data.train.next_batch(FLAGS.batch_size)
            centers_feed, _ = center_set.next_batch(FLAGS.batch_size, UNSUPERVISED = True)
            feed_dict_combined = {input_batch_pl: input_feed, input_target_pl: target_feed, input_center_pl: centers_feed}


            for step in range(num_training):


                _, loss_value, kl, penalty,  acc_train = sess.run([train_op, loss, kmeans_loss, sp, accuracy],
                                         feed_dict=feed_dict_combined)

                if do_validation_flag:
                    acc_val, _ = do_validation(sess, ae, data.validation, FLAGS)
                else: acc_val = acc_train


                acc_record.append(acc_val)
                if len(acc_record) > N_record: acc_record.popleft()
                else: pass

                # Write the summaries and print an overview fairly often.
                if (step+1) % FLAGS.display_steps == 0 or step+1 == num_training or step == 0:
                    output = 'Train step ' + str(step+1) + ' minibatch loss: ' + str(loss_value) + \
                             ' penalty: ' + str(penalty) + ' accuracy: ' + str(acc_train) + ' Kmeans: ' + str(kl)
                    print(output)
                    # do validation
                    if do_validation_flag:
                        acc_val, _ = do_validation(sess, ae, data.validation, FLAGS)
                    else: acc_val = acc_train

                    acc_record.append(acc_val)
                    if len(acc_record) > N_record: acc_record.popleft()
                    else: pass

                    acc = acc_val
                    ass_val_show = max(acc_record)

                    output = 'accuracy on validation: ' + str(ass_val_show)
                    print(output)


    return sess, acc, ae, kmeans







