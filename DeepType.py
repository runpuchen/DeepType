"""
Runpu Chen 2019
DeepType.py

Usage:

    Run in command line with required command arguments:
    	python DeepType.py


"""



import tensorflow as tf
from training import *
from flags import set_flags
from eval import do_inference_main
import pickle
import os


if __name__ == '__main__':
    FLAGS = set_flags()
    np.random.seed(0)
    tf.set_random_seed(0)

    # Create folders
    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)


    # create autoencoder and perform training


    _, AE, sess= main_supervised_1view(FLAGS)

    _, _, _, _, _, AE_pretrain, _ = \
        do_inference_main(AE, sess, FLAGS)

    sio.savemat(FLAGS.results_dir+'encoder_pretrain.mat', {'AE_pretrain':AE_pretrain})



    sess, AE, kmeans, _ = main_supervised_unsupervised_1view(AE, sess, FLAGS)


    # infer and save out

    acc_whole, target_predicted, ass_total, manifold, index_total, AE_final, true_targets = \
        do_inference_main(AE, sess, FLAGS)


    # save mat
    sio.savemat(FLAGS.results_dir+'encoder.mat', {'AE_final':AE_final,'acc_whole': acc_whole,'true_targets': true_targets,
        'target_predicted': target_predicted, 'ass_total': ass_total, 'manifold': manifold,'index_total': index_total})

    # save pickle
    pickle_out = open(FLAGS.results_dir + 'encoder.pickle', 'wb')
    pickle.dump([AE_final, acc_whole, target_predicted, ass_total, index_total], pickle_out)
    print('Processing done!')











