
# -*- coding: utf-8 -*-
# Author: Cristian Cardellino

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cnn_input_data
import json
import numpy as np
import os
import sys
import tensorflow as tf

from tqdm import tqdm, trange
from utils import flat, max_pool, run_layer, run_transpose_layer
from gensim.models import KeyedVectors


def main(data_path, results_file, config):
    ####################################################################################
    # Previous operations
    ####################################################################################
    layers = config['layers']
    L = len(layers)

    tf.reset_default_graph()  # Clear the tensorflow graph (free reserved memory)

    ####################################################################################
    # Inputs setup
    ####################################################################################
    max_sentence_len = config['max_sentence_len']

    # feedforward_inputs (FFI): inputs for the feedforward network (i.e. the encoder).
    # Should contain the labeled training data (padded to max_sentence_len).
    feedforward_inputs = tf.placeholder(tf.int32,
                                        shape=(None, max_sentence_len),
                                        name="FFI")

    print('SHAPE feedforward_inputs', feedforward_inputs)

    # autoencoder_inputs (AEI): inputs for the autoencoder (encoder + decoder).
    # Should contain the unlabeled training data (also padded to max_sentence_len).
    autoencoder_inputs = tf.placeholder(tf.int32,
                                        shape=(None, max_sentence_len),
                                        name="AEI")
    print('SHAPE autoencoder_inputs', autoencoder_inputs)

    outputs = tf.placeholder(tf.float32)  # target
    training = tf.placeholder(tf.bool)  # training or evaluation

    # Not quite sure what is this for
    FFI = tf.reshape(feedforward_inputs, [-1] + [max_sentence_len])
    AEI = tf.reshape(autoencoder_inputs, [-1] + [max_sentence_len])

    print('SHAPE FFI', FFI)
    print('SHAPE AEI', AEI)

    ####################################################################################
    # Embeddings weights
    ####################################################################################

    embeddings_size = config['embeddings_size']
    vocab_size = config['vocab_size']
    embeddings_weights = tf.get_variable("embeddings",
                                         (vocab_size, embeddings_size),
                                         trainable=False)
                                         # initializer=tf.random_normal_initializer())

    place = tf.placeholder(tf.float32, shape=(vocab_size, embeddings_size))
    set_embeddings_weights = embeddings_weights.assign(place)

    FFI_embeddings = tf.expand_dims(
        tf.nn.embedding_lookup(embeddings_weights, FFI),
        axis=1,
        name="FFI_embeddings")

    print('SHAPE FFI_embeddings', FFI_embeddings)

    AEI_embeddings = tf.expand_dims(
        tf.nn.embedding_lookup(embeddings_weights, AEI),
        axis=1,
        name="AEI_embeddings")

    print('SHAPE AEI_embeddings', AEI_embeddings)

    ####################################################################################
    # Batch normalization setup & functions
    ####################################################################################
    # to calculate the moving averages of mean and variance
    ewma = tf.train.ExponentialMovingAverage(decay=0.99)
    # this list stores the updates to be made to average mean and variance
    bn_assigns = []

    def update_batch_normalization(batch, output_name="bn", scope_name="BN"):
        dim = len(batch.get_shape().as_list())
        mean, var = tf.nn.moments(batch, axes=list(range(0, dim - 1)))
        # Function to be used during the learning phase.
        # Normalize the batch and update running mean and variance.
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            running_mean = tf.get_variable("running_mean",
                                           mean.shape,
                                           initializer=tf.constant_initializer(0))
            running_var = tf.get_variable("running_var",
                                          mean.shape,
                                          initializer=tf.constant_initializer(1))

        assign_mean = running_mean.assign(mean)
        assign_var = running_var.assign(var)
        bn_assigns.append(ewma.apply([running_mean, running_var]))

        with tf.control_dependencies([assign_mean, assign_var]):
            z = (batch - mean) / tf.sqrt(var + 1e-10)
            return tf.identity(z, name=output_name)

    def batch_normalization(batch, mean=None, var=None, output_name="bn"):
        if mean is None or var is None:
            dim = len(batch.get_shape().as_list())
            mean, var = tf.nn.moments(batch, axes=list(range(0, dim - 1)))
        z = (batch - mean) / tf.sqrt(var + tf.constant(1e-10))
        return tf.identity(z, name=output_name)

    ####################################################################################
    # Encoder
    ####################################################################################
    def encoder_bloc(h, layer_spec, noise_std, update_BN, activation):
        # Run the layer
        z_pre = run_layer(h, layer_spec, output_name="z_pre")

        # Compute mean and variance of z_pre (to be used in the decoder)
        dim = len(z_pre.get_shape().as_list())
        mean, var = tf.nn.moments(z_pre, axes=list(range(0, dim - 1)))
        # Create a variable to store the values for latter retrieving them
        _ = tf.identity(mean, name="mean"), tf.identity(var, name="var")

        # Batch normalization
        def training_batch_norm():
            if update_BN:
                z = update_batch_normalization(z_pre)
            else:
                z = batch_normalization(z_pre)
            return z

        def eval_batch_norm():
            with tf.variable_scope("BN", reuse=tf.AUTO_REUSE):
                mean = ewma.average(tf.get_variable("running_mean",
                                                    shape=z_pre.shape[-1]))
                var = ewma.average(tf.get_variable("running_var",
                                                   shape=z_pre.shape[-1]))
            z = batch_normalization(z_pre, mean, var)
            return z

        # Perform batch norm depending to the phase (training or testing)
        z = tf.cond(training, training_batch_norm, eval_batch_norm)
        z += tf.random_normal(tf.shape(z)) * noise_std
        z = tf.identity(z, name="z")

        # Center and scale plus activation
        size = z.get_shape().as_list()[-1]
        beta = tf.get_variable("beta", [size],
                               initializer=tf.constant_initializer(0))
        gamma = tf.get_variable("gamma", [size],
                                initializer=tf.constant_initializer(1))

        h = activation(z * gamma + beta)
        return tf.identity(h, name="h")

    def encoder(h, noise_std, update_BN):
        # Perform encoding for each layer
        h += tf.random_normal(tf.shape(h)) * noise_std
        h = tf.identity(h, "h0")

        for i, layer_spec in enumerate(layers):
            print("Building encoder layer %s for %s encoder" %
                  (layer_spec["name"], "corrupted" if noise_std > 0 else "clean"),
                  file=sys.stderr)
            with tf.variable_scope("encoder_bloc_" + str(i + 1), reuse=tf.AUTO_REUSE):
                # Create an encoder bloc if the layer type is dense or conv2d
                if layer_spec["type"] == "flat":
                    h = flat(h, output_name="h")
                elif layer_spec["type"] == "max_pool":
                    h = max_pool(h, layer_spec, output_name="h")
                else:
                    if i == L - 1:
                        activation = tf.nn.softmax  # Only for the last layer
                    else:
                        activation = tf.nn.relu
                    h = encoder_bloc(h, layer_spec, noise_std,
                                     update_BN=update_BN,
                                     activation=activation)

        y = tf.identity(h, name="y")
        return y

    noise_std = config['noise_std']

    with tf.name_scope("FF_clean"):
        # output of the clean encoder. Used for prediction
        FF_y = encoder(FFI_embeddings, 0, update_BN=False)
    with tf.name_scope("FF_corrupted"):
        # output of the corrupted encoder. Used for training.
        FF_y_corr = encoder(FFI_embeddings, noise_std, update_BN=False)

    with tf.name_scope("AE_clean"):
        # corrupted encoding of unlabeled instances
        AE_y = encoder(AEI_embeddings, 0, update_BN=True)
    with tf.name_scope("AE_corrupted"):
        # corrupted encoding of unlabeled instances
        AE_y_corr = encoder(AEI_embeddings, noise_std, update_BN=False)

    ####################################################################################
    # Decoder
    ####################################################################################

    def g_gauss(z_c, u, output_name="z_est", scope_name="denoising_func"):
        # gaussian denoising function proposed in the original paper
        size = u.get_shape().as_list()[-1]

        def wi(inits, name):
            return tf.Variable(inits * tf.ones([size]), name=name)

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            a1 = wi(0., 'a1')
            a2 = wi(1., 'a2')
            a3 = wi(0., 'a3')
            a4 = wi(0., 'a4')
            a5 = wi(0., 'a5')

            a6 = wi(0., 'a6')
            a7 = wi(1., 'a7')
            a8 = wi(0., 'a8')
            a9 = wi(0., 'a9')
            a10 = wi(0., 'a10')

            mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
            v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

            z_est = (z_c - mu) * v + mu
        return tf.identity(z_est, name=output_name)

    def decoder_bloc(u, z_corr, mean, var, layer_spec=None):
        # Performs the decoding operations of a corresponding encoder bloc
        # Denoising
        z_est = g_gauss(z_corr, u)

        z_est_BN = (z_est - mean) / tf.sqrt(var + tf.constant(1e-10))
        z_est_BN = tf.identity(z_est_BN, name="z_est_BN")

        # run transposed layer
        if layer_spec is not None:
            u = run_transpose_layer(z_est, layer_spec)
            u = batch_normalization(u, output_name="u")

        return u, z_est_BN

    def get_tensor(input_name, num_encoder_bloc, name_tensor):
        return tf.get_default_graph().\
            get_tensor_by_name(input_name + "/encoder_bloc_" +
                               str(num_encoder_bloc) + "/" + name_tensor + ":0")

    denoising_cost = config['denoising_cost']
    d_cost = []
    u = batch_normalization(AE_y_corr, output_name="u_L")
    for i in range(L, 0, -1):
        layer_spec = layers[i - 1]
        print("Building decoder layer %s" % layer_spec["name"], file=sys.stderr)

        with tf.variable_scope("decoder_bloc_" + str(i), reuse=tf.AUTO_REUSE):
            if layer_spec["type"] in ["max_pool", "flat"]:
                # if the layer is max pooling or "flat", the transposed layer is run
                # without creating a decoder bloc.
                h = get_tensor("AE_corrupted", i - 1, "h")
                output_shape = tf.shape(h)
                u = run_transpose_layer(u, layer_spec, output_shape=output_shape)
            else:
                z_corr = get_tensor("AE_corrupted", i, "z")
                z = get_tensor("AE_clean", i, "z")
                mean = get_tensor("AE_clean", i, "mean")
                var = get_tensor("AE_clean", i, "var")

                u, z_est_BN = decoder_bloc(u, z_corr, mean, var,
                                           layer_spec=layer_spec)
                d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[i])

    # last decoding step
    with tf.variable_scope("decoder_bloc_0", reuse=tf.AUTO_REUSE):
        z_corr = tf.get_default_graph().get_tensor_by_name("AE_corrupted/h0:0")
        z = tf.get_default_graph().get_tensor_by_name("AE_clean/h0:0")
        mean, var = tf.constant(0.0), tf.constant(1.0)

        u, z_est_BN = decoder_bloc(u, z_corr, mean, var)
        d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[0])

    ####################################################################################
    # Loss, accuracy and optimization
    ####################################################################################

    u_cost = tf.add_n(d_cost)  # reconstruction cost
    corr_pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs * tf.log(FF_y_corr), 1))  # supervised cost
    clean_pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs * tf.log(FF_y), 1))

    loss = corr_pred_cost + u_cost  # total cost

    predictions = tf.argmax(FF_y, 1)
    correct_prediction = tf.equal(predictions, tf.argmax(outputs, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Optimization setting
    starter_learning_rate = config['starter_learning_rate']
    learning_rate = tf.Variable(starter_learning_rate, trainable=False)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # add the updates of batch normalization statistics to train_step
    bn_updates = tf.group(*bn_assigns)
    with tf.control_dependencies([train_step]):
        train_step = tf.group(bn_updates)

    n = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("There is a total of %d trainable parameters" % n, file=sys.stderr)

    ####################################################################################
    # Training
    ####################################################################################
    print("===  Loading Data ===", file=sys.stderr)
    data, w2v_model = cnn_input_data.read_data_sets(data_path,
                                                    n_classes=config['num_classes'],
                                                    n_labeled=config['num_labeled'],
                                                    maxlen=max_sentence_len)
    num_examples = data.train.unlabeled_ds.instances.shape[0]

    batch_size = config['batch_size']
    num_epochs = config['num_epochs']

    num_iter = (num_examples // batch_size) * num_epochs  # number of loop iterations

    print("===  Starting Session ===", file=sys.stderr)
    sess = tf.Session()
    results_log = open(results_file, "w")
    print("experiment,split,epoch,accuracy,lloss,true,pred", file=results_log)

    init = tf.global_variables_initializer()
    sess.run(init)

    print('=== Initializing embeddings with pre-trained weights ===')  # DUDA: es correcto?
    sess.run(set_embeddings_weights, feed_dict={place: w2v_model.vectors})  # syn0})

    print("=== Initial stats ===", file=sys.stderr)
    initial_stats = sess.run(
        [accuracy, clean_pred_cost, predictions],
        feed_dict={feedforward_inputs: data.train.labeled_ds.instances,
                   outputs: data.train.labeled_ds.labels,
                   training: False})
    print("Initial Accuracy for Training Data: %.3g" % initial_stats[0], file=sys.stderr)
    print("Initial Supervised Cost for Training Data: %.3g" % initial_stats[1], file=sys.stderr)

    true_labels = np.argmax(data.train.labeled_ds.labels, 1)
    for i in np.arange(true_labels.shape[0]):
        print("%s,training,0,%.3g,%.3g,%d,%d" %
              (config["experiment_id"],
               initial_stats[0],
               initial_stats[1],
               true_labels[i],
               initial_stats[2][i]), file=results_log)

    # For validation data we traverse in batches and save all the information
    validation_instances = data.validation.instances
    validation_labels = data.validation.labels
    mean_accuracy = []
    mean_loss = []

    for start in trange(0, len(validation_labels), batch_size):
        end = min(start + batch_size, len(validation_labels))
        initial_stats = sess.run(
            [accuracy, clean_pred_cost, predictions],
            feed_dict={feedforward_inputs: validation_instances[start:end],
                       outputs: validation_labels[start:end],
                       training: False})
        mean_accuracy.append(initial_stats[0])
        mean_loss.append(initial_stats[1])

        true_labels = np.argmax(validation_labels[start:end], 1)
        for i in np.arange(true_labels.shape[0]):
            print("%s,validation,0,%.3g,%.3g,%d,%d" %
                  (config["experiment_id"],
                   initial_stats[0],
                   initial_stats[1],
                   true_labels[i],
                   initial_stats[2][i]), file=results_log)

    print("Initial Accuracy for Validation Data: %.3g" % np.mean(mean_accuracy), file=sys.stderr)
    print("Initial Supervised Cost for Validation Data: %.3g" % np.mean(mean_loss), file=sys.stderr)

    results_log.flush()

    print("=== Training Start ===", file=sys.stderr)
    for i in trange(0, num_iter):
        labeled_instances, labels, unlabeled_instances = data.train.next_batch(batch_size)

        sess.run(train_step, feed_dict={feedforward_inputs: labeled_instances,
                                        outputs: labels,
                                        autoencoder_inputs: unlabeled_instances,
                                        training: True})

        if (i > 1) and ((i + 1) % (num_iter / num_epochs) == 0):
            # Compute train and validation stats for each epoch
            epoch_n = i // (num_examples // batch_size) + 1

            tqdm.write("=== Epoch %d stats ===" % epoch_n, file=sys.stderr)
            epoch_stats = sess.run(
                [accuracy, clean_pred_cost, predictions],
                feed_dict={feedforward_inputs: labeled_instances,
                           outputs: labels,
                           training: False})

            tqdm.write("Epoch %d: Accuracy for Training Data: %.3g" %
                       (epoch_n, epoch_stats[0]), file=sys.stderr)
            tqdm.write("Epoch %d: Supervised Cost for Training Data: %.3g" %
                       (epoch_n, epoch_stats[1]), file=sys.stderr)

            true_labels = np.argmax(labels, 1)
            for i in np.arange(true_labels.shape[0]):
                print("%s,training,%d,%.3g,%.3g,%d,%d" %
                      (config["experiment_id"],
                       epoch_n,
                       epoch_stats[0],
                       epoch_stats[1],
                       true_labels[i],
                       epoch_stats[2][i]), file=results_log)

            # For validation data we traverse in batches and save all the information
            validation_instances = data.validation.instances
            validation_labels = data.validation.labels
            mean_accuracy = []
            mean_loss = []

            for start in trange(0, len(validation_labels), batch_size):
                end = min(start + batch_size, len(validation_labels))
                epoch_stats = sess.run(
                    [accuracy, clean_pred_cost, predictions],
                    feed_dict={feedforward_inputs: validation_instances[start:end],
                               outputs: validation_labels[start:end],
                               training: False})

                mean_accuracy.append(epoch_stats[0])
                mean_loss.append(epoch_stats[1])

                true_labels = np.argmax(validation_labels[start:end], 1)
                for i in np.arange(true_labels.shape[0]):
                    print("%s,validation,%d,%.3g,%.3g,%d,%d" %
                          (config["experiment_id"],
                           epoch_n,
                           epoch_stats[0],
                           epoch_stats[1],
                           true_labels[i],
                           epoch_stats[2][i]), file=results_log)

            tqdm.write("Epoch %d: Accuracy for Validation Data: %.3g" %
                       (epoch_n, np.mean(mean_accuracy)), file=sys.stderr)
            tqdm.write("Epoch %d: Supervised Cost for Validation Data: %.3g" %
                       (epoch_n, np.mean(mean_loss)), file=sys.stderr)

            results_log.flush()

            decay_after = config['decay_after']
            if (epoch_n + 1) >= decay_after:
                # decay learning rate
                # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
                ratio = 1.0 * (num_epochs - (epoch_n + 1))  # epoch_n + 1 because learning rate is set for next epoch
                ratio = max(0, ratio / (num_epochs - decay_after))
                sess.run(learning_rate.assign(starter_learning_rate * ratio))

    print("=== Final stats ===", file=sys.stderr)
    epoch_n = num_iter // (num_examples // batch_size) + 1

    final_stats = sess.run(
        [accuracy, clean_pred_cost, predictions],
        feed_dict={feedforward_inputs: data.train.labeled_ds.instances,
                   outputs: data.train.labeled_ds.labels,
                   training: False})
    print("Final Accuracy for Training Data: %.3g" % final_stats[0], file=sys.stderr)
    print("Final Supervised Cost for Training Data: %.3g" % final_stats[1], file=sys.stderr)

    true_labels = np.argmax(data.train.labeled_ds.labels, 1)
    for i in np.arange(true_labels.shape[0]):
        print("%s,training,%d,%.3g,%.3g,%d,%d" %
              (config["experiment_id"],
               epoch_n,
               final_stats[0],
               final_stats[1],
               true_labels[i],
               final_stats[2][i]), file=results_log)

    # For validation data we traverse in batches and save all the information
    validation_instances = data.validation.instances
    validation_labels = data.validation.labels
    mean_accuracy = []
    mean_loss = []

    for start in trange(0, len(validation_labels), batch_size):
        end = min(start + batch_size, len(validation_labels))
        final_stats = sess.run(
            [accuracy, clean_pred_cost, predictions],
            feed_dict={feedforward_inputs: validation_instances[start:end],
                       outputs: validation_labels[start:end],
                       training: False})
        mean_accuracy.append(final_stats[0])
        mean_loss.append(final_stats[1])

        true_labels = np.argmax(validation_labels[start:end], 1)
        for i in np.arange(true_labels.shape[0]):
            print("%s,validation,%d,%.3g,%.3g,%d,%d" %
                  (config["experiment_id"],
                   epoch_n,
                   final_stats[0],
                   final_stats[1],
                   true_labels[i],
                   final_stats[2][i]), file=results_log)

    print("Final Accuracy for Validation Data: %.3g" % np.mean(mean_accuracy), file=sys.stderr)
    print("Final Supervised Cost for Validation Data: %.3g" % np.mean(mean_loss), file=sys.stderr)

    # TEST DATA

    test_instances = data.test.instances
    test_labels = data.test.labels

    for start in trange(0, len(test_labels), batch_size):
        end = min(start + batch_size, len(test_labels))
        final_stats = sess.run(
            [accuracy, clean_pred_cost, predictions],
            feed_dict={feedforward_inputs: test_instances[start:end],
                       outputs: test_labels[start:end],
                       training: False})

        true_labels = np.argmax(test_labels[start:end], 1)
        for i in np.arange(true_labels.shape[0]):
            print("%s,test,%d,%.3g,%.3g,%d,%d" %
                  (config["experiment_id"],
                   epoch_n,
                   final_stats[0],
                   final_stats[1],
                   true_labels[i],
                   final_stats[2][i]), file=results_log)

    print("=== Experiment finished ===", file=sys.stderr)
    sess.close()
    results_log.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("results_file")
    parser.add_argument("config")

    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = json.load(fh)

    main(args.data_path, args.results_file, config)
