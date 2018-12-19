
# -*- coding: utf-8 -*-
# Author: Cristian Cardellino

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import input_data_cnn_wide_ladder
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
    ###    layers = config['layers']
    ###    L = len(layers)

    conv_kernels = config['conv_kernels']
    conv_filters = config['conv_filters']
    num_classes = config['num_classes']

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

    # autoencoder_inputs (AEI): inputs for the autoencoder (encoder + decoder).
    # Should contain the unlabeled training data (also padded to max_sentence_len).
    autoencoder_inputs = tf.placeholder(tf.int32,
                                        shape=(None, max_sentence_len),
                                        name="AEI")

    outputs = tf.placeholder(tf.float32)  # target
    training = tf.placeholder(tf.bool)  # training or evaluation

    # Not quite sure what is this for
    FFI = tf.reshape(feedforward_inputs, [-1] + [max_sentence_len])
    AEI = tf.reshape(autoencoder_inputs, [-1] + [max_sentence_len])

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
        axis=-1,
        name="FFI_embeddings")

    AEI_embeddings = tf.expand_dims(
        tf.nn.embedding_lookup(embeddings_weights, AEI),
        axis=-1,
        name="AEI_embeddings")

    ####################################################################################
    # Batch normalization setup & functions
    ####################################################################################
    # to calculate the moving averages of mean and variance
    # ewma = tf.train.ExponentialMovingAverage(decay=0.99)
    # # this list stores the updates to be made to average mean and variance
    # bn_assigns = []

    # def update_batch_normalization(batch, output_name="bn", scope_name="BN"):
    #     dim = len(batch.get_shape().as_list())
    #     mean, var = tf.nn.moments(batch, axes=list(range(0, dim - 1)))
    #     # Function to be used during the learning phase.
    #     # Normalize the batch and update running mean and variance.
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    #         running_mean = tf.get_variable("running_mean",
    #                                        mean.shape,
    #                                        initializer=tf.constant_initializer(0))
    #         running_var = tf.get_variable("running_var",
    #                                       mean.shape,
    #                                       initializer=tf.constant_initializer(1))

    #     assign_mean = running_mean.assign(mean)
    #     assign_var = running_var.assign(var)
    #     bn_assigns.append(ewma.apply([running_mean, running_var]))

    #     with tf.control_dependencies([assign_mean, assign_var]):
    #         z = (batch - mean) / tf.sqrt(var + 1e-10)
    #         return tf.identity(z, name=output_name)

    def batch_normalization(batch, output_name="bn"):

        dim = len(batch.get_shape().as_list())
        mean, var = tf.nn.moments(batch, axes=list(range(0, dim - 1)))
        # if mean is None or var is None:
        #     dim = len(batch.get_shape().as_list())
        #     mean, var = tf.nn.moments(batch, axes=list(range(0, dim - 1)))
        z = (batch - mean) / tf.sqrt(var + tf.constant(1e-10))
        return tf.identity(z, name=output_name)

    ####################################################################################
    # Encoder
    ####################################################################################
    def encoder_layer(z_pre, noise_std, activation):
        # Run the layer
        # z_pre = run_layer(h, layer_spec, output_name="z_pre")

        # Compute mean and variance of z_pre (to be used in the decoder)
        dim = len(z_pre.get_shape().as_list())
        mean, var = tf.nn.moments(z_pre, axes=list(range(0, dim - 1)))
        # Create a variable to store the values for latter retrieving them
        _ = tf.identity(mean, name="mean"), tf.identity(var, name="var")

        # # Batch normalization
        # def training_batch_norm():
        #     if update_BN:
        #         z = update_batch_normalization(z_pre)
        #     else:
        #         z = batch_normalization(z_pre)
        #     return z

        # def eval_batch_norm():
        #     with tf.variable_scope("BN", reuse=tf.AUTO_REUSE):
        #         mean = ewma.average(tf.get_variable("running_mean",
        #                                             shape=z_pre.shape[-1]))
        #         var = ewma.average(tf.get_variable("running_var",
        #                                            shape=z_pre.shape[-1]))
        #     z = batch_normalization(z_pre, mean, var)
        #     return z

        # Perform batch norm depending to the phase (training or testing)
        # z = tf.cond(training, training_batch_norm, eval_batch_norm)
        z = batch_normalization(z_pre)
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

    def encoder(x, noise_std):
        # Perform encoding for each layer
        x += tf.random_normal(tf.shape(x)) * noise_std
        x = tf.identity(x, "h0")

        # Build the "wide" convolutional layer for each conv_kernel
        # This is the "first" layer
        conv_features = []
        weight_variables = []
        for i, ksize in enumerate(conv_kernels, start=1):
            with tf.variable_scope("encoder_bloc_" + str(i), reuse=tf.AUTO_REUSE):
                W = tf.get_variable("W",
                                    (ksize, embeddings_size, 1, conv_filters),
                                    initializer=tf.truncated_normal_initializer())
                weight_variables.append(W)
                z_pre = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                                     padding="VALID", name="z_pre")
                h = encoder_layer(z_pre, noise_std,  # update_BN=update_BN,
                                  activation=tf.nn.relu)
                h = tf.nn.max_pool(h,
                                   ksize=[1, max_sentence_len - ksize + 1, 1, 1],
                                   strides=[1, 1, 1, 1],
                                   padding="VALID",
                                   name="global_max_pool")
                conv_features.append(h)

        # Build the features layer ("second" layer)
        total_kernels = len(conv_kernels)
        total_conv_features = total_kernels * conv_filters
        with tf.variable_scope("encoder_bloc_" + str(total_kernels + 1), reuse=tf.AUTO_REUSE):
            h = tf.concat(conv_features, 3)
            h = tf.reshape(h, (-1, total_conv_features), name="h")

        # Build the features to classes layer ("last" layer)
        with tf.variable_scope("encoder_bloc_" + str(total_kernels + 2), reuse=tf.AUTO_REUSE):
            W = tf.get_variable("W", (total_conv_features, num_classes),
                                initializer=tf.random_normal_initializer())
            weight_variables.append(W)


            print('h shape', h.shape)
            print('W shape', W.shape)


            z_pre = tf.matmul(h, W, name="z_pre")
            h = encoder_layer(z_pre, noise_std,  # update_BN=update_BN,
                              activation=tf.nn.softmax)

        y = tf.identity(h, name="y")
        return y, weight_variables

    noise_std = config['noise_std']

    with tf.name_scope("FF_clean"):
        # output of the clean encoder. Used for prediction
        FF_y, weight_variables = encoder(FFI_embeddings, 0)  # , update_BN=False)
    with tf.name_scope("FF_corrupted"):
        # output of the corrupted encoder. Used for training.
        FF_y_corr, _ = encoder(FFI_embeddings, noise_std)  # , update_BN=False)

    with tf.name_scope("AE_clean"):
        # corrupted encoding of unlabeled instances
        AE_y, _ = encoder(AEI_embeddings, 0)  # , update_BN=True)
    with tf.name_scope("AE_corrupted"):
        # corrupted encoding of unlabeled instances
        AE_y_corr, _ = encoder(AEI_embeddings, noise_std)  # , update_BN=False)

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
    u = batch_normalization(AE_y_corr, output_name="u_L")

    # Build first decoder layer (corresponding to the dense layer)
    total_kernels = len(conv_kernels)
    total_conv_features = total_kernels * conv_filters
    with tf.variable_scope("decoder_bloc_" + str(total_kernels + 2), reuse=tf.AUTO_REUSE):
        z_corr = get_tensor("AE_corrupted", total_kernels + 2, "z")
        z = get_tensor("AE_clean", total_kernels + 2, "z")
        mean = get_tensor("AE_clean", total_kernels + 2, "mean")
        var = get_tensor("AE_clean", total_kernels + 2, "var")
        # Performs the decoding operations of a corresponding encoder bloc
        # Denoising
        z_est = g_gauss(z_corr, u)

        z_est_BN = (z_est - mean) / tf.sqrt(var + tf.constant(1e-10))
        z_est_BN = tf.identity(z_est_BN, name="z_est_BN")

        # run decoder layer
        V = tf.get_variable("V", (num_classes, total_conv_features),
                            initializer=tf.random_normal_initializer())
        l2_reg += tf.nn.l2_loss(V)
        u = tf.matmul(z_est, V)
        u = batch_normalization(u, output_name="u")

        d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[2])

    # Build second decoder layer (corresponding to the concatenation+flat layer)
    with tf.variable_scope("decoder_bloc_" + str(total_kernels + 1), reuse=tf.AUTO_REUSE):
        u = tf.reshape(u, (-1, 1, 1, total_conv_features))
        deconv_features = tf.split(u, total_kernels, axis=3)

    # Build the final "wide convolutional" layer
    deconv_layers = []
    for i, gmp_layer in enumerate(deconv_features, start=1):
        ksize = conv_kernels[i - 1]
        with tf.variable_scope("decoder_bloc_" + str(i), reuse=tf.AUTO_REUSE):
            u = tf.keras.layers.UpSampling2D(
                size=(max_sentence_len - ksize + 1, 1))(gmp_layer)

            z_corr = get_tensor("AE_corrupted", i, "z")
            z = get_tensor("AE_clean", i, "z")
            mean = get_tensor("AE_clean", i, "mean")
            var = get_tensor("AE_clean", i, "var")
            z_est = g_gauss(z_corr, u)

            z_est_BN = (z_est - mean) / tf.sqrt(var + tf.constant(1e-10))
            z_est_BN = tf.identity(z_est_BN, name="z_est_BN")

            # run deconvolutional (transposed convolution) layer
            V = tf.get_variable("V",
                                (ksize, embeddings_size, 1, conv_filters),
                                initializer=tf.truncated_normal_initializer())
            l2_reg += tf.nn.l2_loss(V)

            u = tf.nn.conv2d_transpose(z_est, V,
                                       output_shape=tf.shape(AEI_embeddings),
                                       strides=[1, 1, 1, 1], padding='VALID')
            u = batch_normalization(u, output_name="u")
            deconv_layers.append(u)
            d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[1])

    # last decoding step
    u = tf.concat(deconv_layers, 2)
    with tf.variable_scope("decoder_bloc_0", reuse=tf.AUTO_REUSE):
        z_corr = tf.get_default_graph().get_tensor_by_name("AE_corrupted/h0:0")
        z_corr = tf.concat([z_corr] * total_kernels, 2)
        z = tf.get_default_graph().get_tensor_by_name("AE_clean/h0:0")
        z = tf.concat([z] * total_kernels, 2)
        z_est = g_gauss(z_corr, u)
        d_cost.append((tf.reduce_mean(tf.square(z_est - z))) * denoising_cost[0])

    ####################################################################################
    # Loss, accuracy and optimization
    ####################################################################################

    u_cost = tf.add_n(d_cost)  # reconstruction cost
    corr_pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs * tf.log(FF_y_corr), 1))  # supervised cost
    clean_pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs * tf.log(FF_y), 1))

    loss = corr_pred_cost + u_cost + config.get("lambda", 0.0) * l2_reg  # total cost

    predictions = tf.argmax(FF_y, 1)
    correct_prediction = tf.equal(predictions, tf.argmax(outputs, 1))
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
    data, w2v_model = input_data_cnn_wide_ladder.read_data_sets(data_path,
                                                    n_classes=config['num_classes'],
                                                    n_labeled=config['num_labeled'],
                                                    maxlen=max_sentence_len)
    num_examples = data.train.unlabeled_ds.instances.shape[0]

    batch_size = config['batch_size']
    num_epochs = config['num_epochs']

    num_iter = (num_examples // batch_size) * num_epochs  # number of loop iterations

    print("===  Starting Session ===", file=sys.stderr)
    dev_config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    dev_config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    dev_config.gpu_options.per_process_gpu_memory_fraction = 1#0.5
    sess = tf.Session(config=dev_config)

    if not os.path.exists(results_file):
        results_log = open(results_file, "w")
        print("experiment,split,epoch,accuracy,tloss,lloss,true,pred", file=results_log)

    else:
        results_log = open(results_file, "a")

    init = tf.global_variables_initializer()
    sess.run(init)

    print('=== Initializing embeddings with pre-trained weights ===')
    sess.run(set_embeddings_weights, feed_dict={place: w2v_model.syn0})  #.vectors})

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

                true_labels = np.argmax(training_labels[start:end], 1)
                for i in np.arange(true_labels.shape[0]):
                    print("%s,training,%d,%.3g,%.3g,%.3g,%d,%d" %
                          (config["experiment_id"],
                           epoch_n,
                           epoch_stats[0],
                           epoch_stats[1],
                           epoch_stats[2],
                           true_labels[i],
                           epoch_stats[3][i]), file=results_log)

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

                true_labels = np.argmax(validation_labels[start:end], 1)
                for i in np.arange(true_labels.shape[0]):
                    print("%s,validation,%d,%.3g,%.3g,%.3g,%d,%d" %
                          (config["experiment_id"],
                           epoch_n,
                           epoch_stats[0],
                           epoch_stats[1],
                           epoch_stats[2],
                           true_labels[i],
                           epoch_stats[3][i]), file=results_log)

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

        true_labels = np.argmax(training_labels[start:end], 1)
        for i in np.arange(true_labels.shape[0]):
            print("%s,training,%d,%.3g,%.3g,%.3g,%d,%d" %
                  (config["experiment_id"],
                   epoch_n,
                   final_stats[0],
                   final_stats[1],
                   final_stats[2],
                   true_labels[i],
                   final_stats[3][i]), file=results_log)

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

        true_labels = np.argmax(validation_labels[start:end], 1)
        for i in np.arange(true_labels.shape[0]):
            print("%s,validation,%d,%.3g,%.3g,%.3g,%d,%d" %
                  (config["experiment_id"],
                   epoch_n,
                   final_stats[0],
                   final_stats[1],
                   final_stats[2],
                   true_labels[i],
                   final_stats[3][i]), file=results_log)

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

        true_labels = np.argmax(test_labels[start:end], 1)
        for i in np.arange(true_labels.shape[0]):
            print("%s,test,%d,%.3g,%.3g,%.3g,%d,%d" %
                  (config["experiment_id"],
                   epoch_n,
                   final_stats[0],
                   final_stats[1],
                   final_stats[2],
                   true_labels[i],
                   final_stats[3][i]), file=results_log)

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
