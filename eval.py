import tensorflow as tf
from data import *
from model import *
from utils import *
from sklearn.cluster import KMeans




def loss_supervised(logits_est, labels_true, ae,  alpha): # penalize could be exclusive Lasso

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels_true * tf.log(logits_est + 1e-16), reduction_indices=[1]))
    weight_1 = ae._w(1)
    l21 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(weight_1, 2), 1)))

    penalty = l21

    loss = tf.reduce_mean(cross_entropy) + penalty*alpha
    return loss, penalty, cross_entropy


def loss_supervised_unsupervised(ae, logits, labels, hidden, M, FLAGS):
    ls, penalty, cross_entropy = loss_supervised(logits, labels, ae, FLAGS.alpha)
    diff = hidden - M
    lk = tf.reduce_mean(tf.reduce_sum(tf.pow(diff, 2), 1), 0)

    loss = ls + FLAGS.beta * lk
    return loss, lk, penalty, cross_entropy



def evaluation(logits, labels):
    pred_temp = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    num = pred_temp.shape[0]
    correct = tf.reduce_mean(tf.cast(pred_temp, "float"))

    return correct, tf.argmax(logits, 1)


def do_get_hidden(sess,ae, data_set, n, FLAGS):
    [n_sample, _] = data_set._labels.shape

    input_train_pl = tf.placeholder(tf.float32, shape = (n_sample,FLAGS.dimension), name = 'input_train')

    last_layer_train = ae.supervised_net(input_train_pl, n) # infer the hiddens

    feed_dict = fill_feed_dict_ae_for_hidden(data_set, input_train_pl, FLAGS) # change

    hidden_layer = sess.run(last_layer_train, feed_dict = feed_dict)

    return hidden_layer


def do_validation(sess, ae, data_validation, FLAGS):


    [validation_size, validation_dim] = data_validation.data.shape
    validation_pl = tf.placeholder(tf.float32, shape=(validation_size, validation_dim), name='validation_pl')
    target_pl = tf.placeholder(tf.float32, shape=(validation_size, FLAGS.num_classes), name='target_pl')

    feed_dict = fill_feed_dict_ae_test(data_validation, validation_pl, target_pl, FLAGS)

    logits_validation = ae.supervised_net(validation_pl, FLAGS.num_hidden_layers+1)

    accuracy, targets = evaluation(logits_validation, target_pl)

    acc_val, target_prediction = sess.run([accuracy, targets], feed_dict = feed_dict)



    return acc_val, target_prediction


def do_get_hidden_mv(sess, AE_list, data_set, n, FLAGS):
    [n_sample, _] = data_set._labels.shape
    input_pl = [tf.placeholder(tf.float32, shape=(n_sample,FLAGS.dimension[v]))
                    for v in range(FLAGS.num_view)]



    last_layer_list = [AE_list[v].supervised_net(input_pl[v], FLAGS.num_hidden_layers)
                           for v in range(FLAGS.num_view)]

    last_layer_concat = tf.concat(last_layer_list, 1)

    hidden_layer = AE_list[FLAGS.num_view].supervised_net(last_layer_concat, 1)


    feed_dict = fill_feed_dict_ae_for_hidden(data_set, input_pl, FLAGS) # change

    hidden_layer = sess.run(hidden_layer, feed_dict = feed_dict)


    return hidden_layer

def do_inference_main(AE, sess, FLAGS):
    # data is data_whole
    data, index = read_data_sets(FLAGS, test = True)
    with sess.graph.as_default():
        initialize_uninitialized(sess)
        true_targets = data.labels
        manifold = do_get_hidden(sess, AE,  data, FLAGS.num_hidden_layers, FLAGS)
        acc, target_predicted = do_validation(sess, AE, data, FLAGS)
        kmeans = KMeans(n_clusters=FLAGS.num_clusters,init='k-means++', max_iter=50, tol=0.01).fit(manifold)
        assignments = kmeans.predict(manifold)

        title = FLAGS.results_dir + 'FinalTrainedClusteredFinal'
        types = np.unique(assignments)
        X_TSNE_trained, X_PCA_trained = Transfer_TSNE_PCA(manifold, 2, 3)
        VisualizeHidden(X_TSNE_trained, X_PCA_trained, assignments, types, title)
        labels = np.nonzero(true_targets == 1)[1]
        title = FLAGS.results_dir + 'FinalTrainedFinal'
        VisualizeHidden(X_TSNE_trained, X_PCA_trained, labels, types, title)

        AE_final = dict()

        for i in range(AE.num_hidden_layers + 1):
            w_name_i = 'w_' + str(i+1)
            w_name_in_ae_i = 'weights' + str(i+1)
            temp = sess.run(AE[w_name_in_ae_i])
            AE_final[w_name_i] = temp
            b_name_i = 'b_' + str(i+1)
            b_name_in_ae_i = 'biases' + str(i+1)
            temp = sess.run(AE[b_name_in_ae_i])
            AE_final[b_name_i] = temp


    return acc, target_predicted, assignments, manifold, index, AE_final, true_targets











