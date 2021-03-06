
# -*- coding: utf-8 -*-
# Authors: Cristian Cardellino & Emiliano Kokic

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import input_data_seq_tag_cnn_depth_ladder
import json
import numpy as np
import os
import sys
import tensorflow as tf

from tqdm import tqdm, trange
from gensim.models import KeyedVectors


def main(data_path, results_file, config):
    ####################################################################################
    # Previous operations
    ####################################################################################
    conv_layers = config['conv_layers']
    conv_kernels = config['conv_kernels']
    conv_filters = config['conv_filters']
    num_classes = config['num_classes']

    tf.reset_default_graph()  # Clear the tensorflow graph (free reserved memory)

    ####################################################################################
    # Inputs setup
    ####################################################################################
    max_seq_len = config['max_sentence_len']

    # feedforward_inputs (FFI): inputs for the feedforward network (i.e. the encoder).
    # Should contain the labeled training data (padded to max_seq_len).
    feedforward_inputs = tf.placeholder(tf.int32,
                                        shape=(None, max_seq_len),
                                        name="FFI")

    print('SHAPE feedforward_inputs', feedforward_inputs.shape)

    # autoencoder_inputs (AEI): inputs for the autoencoder (encoder + decoder).
    # Should contain the unlabeled training data (also padded to max_seq_len).
    autoencoder_inputs = tf.placeholder(tf.int32,
                                        shape=(None, max_seq_len),
                                        name="AEI")
    print('SHAPE autoencoder_inputs', autoencoder_inputs.shape)

    outputs = tf.placeholder(tf.int64)  # (tf.float32)  # target
    training = tf.placeholder(tf.bool)  # training or evaluation

    # Not quite sure what is this for
    FFI = tf.reshape(feedforward_inputs, [-1] + [max_seq_len])
    AEI = tf.reshape(autoencoder_inputs, [-1] + [max_seq_len])

    print('SHAPE FFI', FFI.shape)
    print('SHAPE AEI', AEI.shape)

    ####################################################################################
    # Embeddings weights
    ####################################################################################

    embeddings_size = config['embeddings_size']
    vocab_size = config['vocab_size']
    embeddings_weights = tf.get_variable("embeddings",
                                         (vocab_size, embeddings_size),
                                         trainable=False)

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

    def batch_normalization(batch, output_name="bn"):
        dim = len(batch.get_shape().as_list())
        mean, var = tf.nn.moments(batch, axes=list(range(0, dim - 1)))
        z = (batch - mean) / tf.sqrt(var + tf.constant(1e-10))
        return tf.identity(z, name=output_name)

    ####################################################################################
    # Encoder
    ####################################################################################
    def encoder_layer(z_pre, noise_std, activation=None):
        # Compute mean and variance of z_pre (to be used in the decoder)
        dim = len(z_pre.get_shape().as_list())
        mean, var = tf.nn.moments(z_pre, axes=list(range(0, dim - 1)))
        # Create a variable to store the values for latter retrieving them
        _ = tf.identity(mean, name="mean"), tf.identity(var, name="var")

        z = batch_normalization(z_pre)
        z += tf.random_normal(tf.shape(z)) * noise_std
        z = tf.identity(z, name="z")

        # Center and scale plus activation
        size = z.get_shape().as_list()[-1]
        beta = tf.get_variable("beta", [size],
                               initializer=tf.constant_initializer(0))
        gamma = tf.get_variable("gamma", [size],
                                initializer=tf.constant_initializer(1))
        if activation == None:  # output layer (the softmax activation is computed with the loss together.)
            h = gamma * (z + beta)
        else:
            h = activation(gamma * (z + beta))  # try removing gamma variable in this case.

        return tf.identity(h, name="h")

    def encoder(h, noise_std):
        # Perform encoding for each layer
        h += tf.random_normal(tf.shape(h)) * noise_std
        h = tf.identity(h, "h0")

        weight_variables = []

        print('Building conv layers...')

        for i in range(conv_layers):
            ksize = 2  # TODO: kernel size configurable
            with tf.variable_scope("encoder_bloc_" + str(i), reuse=tf.AUTO_REUSE):

                print('conv layer:', i)
                if i == 0:  # first conv layer
                    W = tf.get_variable("W",
                                        (1, ksize, embeddings_size, conv_filters),
                                        initializer=tf.truncated_normal_initializer())
                    weight_variables.append(W)

                else:  # other conv layer
                    W = tf.get_variable("W",
                                        (1, ksize, conv_filters, conv_filters),
                                        initializer=tf.truncated_normal_initializer())
                    weight_variables.append(W)

                print('-------------- h shape:', h.shape)
                print('-------------- W shape:', W.shape)

                z_pre = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1],
                                     padding="SAME", name="z_pre")

                h = encoder_layer(z_pre, noise_std, activation=tf.nn.relu)

        # Build the features to classes layer ("last" layer)
        with tf.variable_scope("encoder_bloc_" + str(conv_layers), reuse=tf.AUTO_REUSE):
            print('Building output layer...')

            W = tf.get_variable("W", (conv_filters, num_classes),
                                initializer=tf.random_normal_initializer())
            weight_variables.append(W)

            print('-------------- h shape', h.shape)

            flatten_conv = tf.reshape(h, [-1, conv_filters])

            print('-------------- flatten_conv shape', h.shape)
            print('-------------- W shape', W.shape)

            last_layer = tf.matmul(flatten_conv, W)

            print('-------------- last_layer shape', last_layer.shape)

            logits = encoder_layer(last_layer, noise_std,
                                   activation=None)  # softmax activation is computed together with the loss.
            print('-------------- logits shape:', h.get_shape().as_list())

            h = tf.reshape(logits, [-1, max_seq_len, num_classes])

            print('-------------- y shape', h.shape)
        y = tf.identity(h, name="y")
        return y, weight_variables, logits

    noise_std = config['noise_std']

    with tf.name_scope("FF_clean"):
        # output of the clean encoder. Used for prediction
        FF_y, weight_variables, FF_y_logits = encoder(FFI_embeddings, 0)
    with tf.name_scope("FF_corrupted"):
        # output of the corrupted encoder. Used for training.
        FF_y_corr, _, _ = encoder(FFI_embeddings, noise_std)

    with tf.name_scope("AE_clean"):
        # corrupted encoding of unlabeled instances
        AE_y, _, _ = encoder(AEI_embeddings, 0)
    with tf.name_scope("AE_corrupted"):
        # corrupted encoding of unlabeled instances
        AE_y_corr, _, AE_logits_corr = encoder(AEI_embeddings, noise_std)

    l2_reg = tf.constant(0.0)
    for we_var in weight_variables:
        l2_reg += tf.nn.l2_loss(we_var)
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

    def get_tensor(input_name, num_encoder_bloc, name_tensor):
        return tf.get_default_graph().\
            get_tensor_by_name(input_name + "/encoder_bloc_" +
                               str(num_encoder_bloc) + "/" + name_tensor + ":0")

    denoising_cost = config['denoising_cost']
    d_cost = []
    # u = batch_normalization(AE_y_corr, output_name="u_L")
    u = batch_normalization(AE_logits_corr, output_name="u_L")

    # Build first decoder layer (corresponding to the output layer)
    with tf.variable_scope("decoder_bloc_" + str(conv_layers), reuse=tf.AUTO_REUSE):
        z_corr = get_tensor("AE_corrupted", conv_layers, "z")
        z = get_tensor("AE_clean", conv_layers, "z")
        mean = get_tensor("AE_clean", conv_layers, "mean")
        var = get_tensor("AE_clean", conv_layers, "var")
        # Performs the decoding operations of a corresponding encoder bloc
        # Denoising
        print('first decoder layer')
        print('-------------- z_corr shape', z_corr.shape)
        print('-------------- u shape', u.shape)

        z_est = g_gauss(z_corr, u)

        z_est_BN = (z_est - mean) / tf.sqrt(var + tf.constant(1e-10))
        z_est_BN = tf.identity(z_est_BN, name="z_est_BN")

        V = tf.get_variable("V", (num_classes, conv_filters),
                            initializer=tf.random_normal_initializer())

        l2_reg += tf.nn.l2_loss(V)
        u = tf.matmul(z_est, V)

        # Decoding flatten_conv
        u = tf.reshape(u, [-1, 1, max_seq_len, conv_filters])
        print('decoding flatten_conv')
        print('-------------- u shape', u.shape)

        u = batch_normalization(u, output_name="u")

        d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[-1])

    deconv_layers = []

    for i in range(conv_layers - 1, -1, -1):
        ksize = 2
        with tf.variable_scope("decoder_bloc_" + str(i), reuse=tf.AUTO_REUSE):

            z_corr = get_tensor("AE_corrupted", i, "z")
            z = get_tensor("AE_clean", i, "z")
            mean = get_tensor("AE_clean", i, "mean")
            var = get_tensor("AE_clean", i, "var")

            print('deconv_layer:', i)
            print('-------------- z_corr shape', z_corr.shape)
            print('-------------- u shape', u.shape)

            z_est = g_gauss(z_corr, u)

            z_est_BN = (z_est - mean) / tf.sqrt(var + tf.constant(1e-10))
            z_est_BN = tf.identity(z_est_BN, name="z_est_BN")

            # run deconvolutional (transposed convolution) layer
            if (i == 0):
                V = tf.get_variable("V",
                                    (1, ksize, conv_filters, embeddings_size),
                                    initializer=tf.truncated_normal_initializer())

                print('-------------- z_est shape', z_est.shape)
                print('-------------- V shape', V.shape)
                print('AEI_embeddings shape', AEI_embeddings.shape)

                u = tf.nn.conv2d(z_est, V, strides=[1, 1, 1, 1], padding='SAME')
            else:

                V = tf.get_variable("V",
                                    (1, ksize, conv_filters, conv_filters),
                                    initializer=tf.truncated_normal_initializer())
                print('-------------- z_est shape', z_est.shape)
                print('-------------- V shape', V.shape)
                u = tf.nn.conv2d(z_est, V, strides=[1, 1, 1, 1], padding='SAME')

            l2_reg += tf.nn.l2_loss(V)

            u = batch_normalization(u, output_name="u")
            deconv_layers.append(u)
            d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[i])

    # last decoding step
    with tf.variable_scope("decoder_bloc_0", reuse=tf.AUTO_REUSE):
        z_corr = tf.get_default_graph().get_tensor_by_name("AE_corrupted/h0:0")
        z = tf.get_default_graph().get_tensor_by_name("AE_clean/h0:0")
        mean, var = tf.constant(0.0), tf.constant(1.0)

        print('-------------- z_corr shape', z_corr.shape)
        print('-------------- u shape', u.shape)
        z_est = g_gauss(z_corr, u)

        z_est_BN = (z_est - mean) / tf.sqrt(var + tf.constant(1e-10))
        z_est_BN = tf.identity(z_est_BN, name="z_est_BN")

        d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[0])

    ####################################################################################
    # Loss, accuracy and optimization
    ####################################################################################

    u_cost = tf.add_n(d_cost)  # reconstruction cost

    corr_pred_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=outputs, logits=FF_y_corr))
    clean_pred_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=outputs, logits=FF_y))

    loss = corr_pred_cost + u_cost * config['u_cost_weight'] + config.get("lambda", 0.0) * l2_reg  # total cost

    predictions = tf.argmax(FF_y, 2)

    print('predictions shape', predictions.shape)
    print('outputs shape', outputs.shape)
    correct_prediction = tf.equal(predictions, outputs)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Optimization setting
    starter_learning_rate = config['starter_learning_rate']
    learning_rate = tf.Variable(starter_learning_rate, trainable=False)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # add the updates of batch normalization statistics to train_step
    # bn_updates = tf.group(*bn_assigns)
    # with tf.control_dependencies([train_step]):
    #     train_step = tf.group(bn_updates)

    n = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("There is a total of %d trainable parameters" % n, file=sys.stderr)

    ####################################################################################
    # Training
    ####################################################################################
    print("===  Loading Data ===", file=sys.stderr)
    data, w2v_model = input_data_seq_tag_cnn_depth_ladder.read_data_sets(data_path,
                                                                         n_classes=config['num_classes'],
                                                                         n_labeled=config['num_labeled'],
                                                                         maxlen=max_seq_len)
    num_examples = data.train.unlabeled_ds.instances.shape[0]

    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    exp_name = '_'.join([config['experiment_id'],
                         'conv_layers', str(config['conv_layers']),
                         'conv_filters', str(config['conv_filters']),
                         'conv_kernels', str(config['conv_kernels']),
                         'u_cost_weight', str(config['u_cost_weight'])])

    num_iter = (num_examples // batch_size) * num_epochs  # number of loop iterations

    print("===  Starting Session ===", file=sys.stderr)
    dev_config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    dev_config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    dev_config.gpu_options.per_process_gpu_memory_fraction = 1  # 0.5
    sess = tf.Session(config=dev_config)

    if not os.path.exists(results_file):
        results_log = open(results_file, "w")
        print("experiment,split,epoch,accuracy,tloss,lloss,true,pred", file=results_log)

    else:
        results_log = open(results_file, "a")

    init = tf.global_variables_initializer()
    sess.run(init)

    print('=== Initializing embeddings with pre-trained weights ===')
    sess.run(set_embeddings_weights, feed_dict={place: w2v_model.vectors})

    print("=== Training Start ===", file=sys.stderr)
    tr = trange(0, num_iter, desc="iter: nan - loss: nan")
    for i in tr:
        labeled_instances, labels, unlabeled_instances = data.train.next_batch(batch_size)

        _, tloss, lloss = sess.run([train_step, loss, clean_pred_cost],
                                   feed_dict={feedforward_inputs: labeled_instances,
                                              outputs: labels,
                                              autoencoder_inputs: unlabeled_instances,
                                              training: True})
        tr.set_description("loss: %.5g - lloss: %.5g" % (tloss, lloss))

        if (i > 1) and ((i + 1) % (num_iter / num_epochs) == 0) and i < num_iter - 1:
            # Compute train and validation stats for each epoch
            epoch_n = i // (num_examples // batch_size) + 1

            tqdm.write("=== Epoch %d stats ===" % epoch_n, file=sys.stderr)
            # For training data we traverse in batches and save all the information
            training_instances = data.train.labeled_ds.instances
            training_labels = data.train.labeled_ds.labels
            mean_accuracy = []
            mean_loss = []

            for start in trange(0, len(training_labels), batch_size):
                end = min(start + batch_size, len(training_labels))
                epoch_stats = sess.run(
                    [accuracy, loss, clean_pred_cost, predictions],
                    feed_dict={feedforward_inputs: training_instances[start:end],
                               outputs: training_labels[start:end],
                               autoencoder_inputs: unlabeled_instances,
                               training: False})

                mean_accuracy.append(epoch_stats[0])
                mean_loss.append(epoch_stats[2])

                # true_labels = np.argmax(training_labels[start:end], 1)
                true_labels = training_labels[start:end]

                for i in np.arange(true_labels.shape[0]):
                    print("%s,training,%d,%.3g,%.3g,%.3g,%s,%s" %
                          (exp_name,
                           epoch_n,
                           epoch_stats[0],
                           epoch_stats[1],
                           epoch_stats[2],
                           np.array2string(true_labels[i]),
                           np.array2string(epoch_stats[3][i])),
                          file=results_log)

            tqdm.write("Epoch %d: Accuracy for Training Data: %.3g" %
                       (epoch_n, np.mean(mean_accuracy)), file=sys.stderr)
            tqdm.write("Epoch %d: Supervised Cost for Training Data: %.3g" %
                       (epoch_n, np.mean(mean_loss)), file=sys.stderr)

            # For validation data we traverse in batches and save all the information
            validation_instances = data.validation.instances
            validation_labels = data.validation.labels
            mean_accuracy = []
            mean_loss = []

            for start in trange(0, len(validation_labels), batch_size):
                end = min(start + batch_size, len(validation_labels))
                epoch_stats = sess.run(
                    [accuracy, loss, clean_pred_cost, predictions],
                    feed_dict={feedforward_inputs: validation_instances[start:end],
                               outputs: validation_labels[start:end],
                               autoencoder_inputs: unlabeled_instances,
                               training: False})

                mean_accuracy.append(epoch_stats[0])
                mean_loss.append(epoch_stats[2])

                true_labels = validation_labels[start:end]
                for i in np.arange(true_labels.shape[0]):
                    print("%s,validation,%d,%.3g,%.3g,%.3g,%s,%s" %
                          (exp_name,
                           epoch_n,
                           epoch_stats[0],
                           epoch_stats[1],
                           epoch_stats[2],
                           np.array2string(true_labels[i]),
                           np.array2string(epoch_stats[3][i])),
                          file=results_log)

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

    training_instances = data.train.labeled_ds.instances
    training_labels = data.train.labeled_ds.labels
    mean_accuracy = []
    mean_loss = []

    for start in trange(0, len(training_labels), batch_size):
        end = min(start + batch_size, len(training_labels))
        final_stats = sess.run(
            [accuracy, loss, clean_pred_cost, predictions],
            feed_dict={feedforward_inputs: training_instances[start:end],
                       outputs: training_labels[start:end],
                       autoencoder_inputs: unlabeled_instances,
                       training: False})

        mean_accuracy.append(final_stats[0])
        mean_loss.append(final_stats[2])

        # true_labels = np.argmax(training_labels[start:end], 1)
        true_labels = training_labels[start:end]
        for i in np.arange(true_labels.shape[0]):
            print("%s,training,%d,%.3g,%.3g,%.3g,%s,%s" %
                  (exp_name,
                   epoch_n,
                   final_stats[0],
                   final_stats[1],
                   final_stats[2],
                   np.array2string(true_labels[i]),
                   np.array2string(final_stats[3][i])),
                  file=results_log)

    print("Final Accuracy for Training Data: %.3g" % np.mean(mean_accuracy), file=sys.stderr)
    print("Final Supervised Cost for Training Data: %.3g" % np.mean(mean_loss), file=sys.stderr)

    # For validation data we traverse in batches and save all the information
    validation_instances = data.validation.instances
    validation_labels = data.validation.labels
    mean_accuracy = []
    mean_loss = []

    for start in trange(0, len(validation_labels), batch_size):
        end = min(start + batch_size, len(validation_labels))
        final_stats = sess.run(
            [accuracy, loss, clean_pred_cost, predictions],
            feed_dict={feedforward_inputs: validation_instances[start:end],
                       outputs: validation_labels[start:end],
                       autoencoder_inputs: unlabeled_instances,
                       training: False})
        mean_accuracy.append(final_stats[0])
        mean_loss.append(final_stats[2])

        # true_labels = np.argmax(validation_labels[start:end], 1)
        true_labels = validation_labels[start:end]
        for i in np.arange(true_labels.shape[0]):
            print("%s,validation,%d,%.3g,%.3g,%.3g,%s,%s" %
                  (exp_name,
                   epoch_n,
                   final_stats[0],
                   final_stats[1],
                   final_stats[2],
                   np.array2string(true_labels[i]),
                   np.array2string(final_stats[3][i])),
                  file=results_log)

    print("Final Accuracy for Validation Data: %.3g" % np.mean(mean_accuracy), file=sys.stderr)
    print("Final Supervised Cost for Validation Data: %.3g" % np.mean(mean_loss), file=sys.stderr)

    # TEST DATA

    test_instances = data.test.instances
    test_labels = data.test.labels

    for start in trange(0, len(test_labels), batch_size):
        end = min(start + batch_size, len(test_labels))
        final_stats = sess.run(
            [accuracy, loss, clean_pred_cost, predictions],
            feed_dict={feedforward_inputs: test_instances[start:end],
                       outputs: test_labels[start:end],
                       autoencoder_inputs: unlabeled_instances,
                       training: False})

        true_labels = test_labels[start:end]
        for i in np.arange(true_labels.shape[0]):
            print("%s,test,%d,%.3g,%.3g,%.3g,%s,%s" %
                  (exp_name,
                   epoch_n,
                   final_stats[0],
                   final_stats[1],
                   final_stats[2],
                   np.array2string(true_labels[i]),
                   np.array2string(final_stats[3][i])),
                  file=results_log)

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
