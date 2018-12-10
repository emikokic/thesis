
import csv
import argparse
import ast
import numpy as np
import pandas as pd
import tensorflow as tf

import os
from os import path
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Embedding
from keras import regularizers, optimizers
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import History, ModelCheckpoint, CSVLogger
from gensim.models import KeyedVectors


NUM_CLASSES = 5  # PER - LOC - ORG - MISC - O


def load_dataset():
    train_data = pd.read_csv('~/thesis/corpus_WiNER/cnn_instances/words_entity_W_2_cnn_train.csv')
    dev_data = pd.read_csv('~/thesis/corpus_WiNER/cnn_instances/words_entity_W_2_cnn_dev.csv')
    test_data = pd.read_csv('~/thesis/corpus_WiNER/cnn_instances/words_entity_W_2_cnn_test.csv')

    return train_data, dev_data, test_data


def transform_input(instances, mapping):
    """Replaces the words in instances with their index in mapping.
    Args:
        instances: a list of text instances.
        mapping: an dictionary from words to indices.
    Returns:
        A matrix with shape (n_instances, m_words)."""
    word_indices = []
    for instance in instances:
        l = []
        for word in ast.literal_eval(instance):
            try:
                l.append(mapping[word].index)
            except KeyError:
                l.append(0)  # index to '</s>' word vector
        word_indices.append(l)

    return word_indices


def tagToInt(tag):
    return {'PER': 0, 'LOC': 1, 'ORG': 2, 'MISC': 3, 'O': 4}[tag]


def preprocess_data(train_data, dev_data, test_data, w2v_model, train_examples_rate):

    X_train = train_data['words'].values[:int(100000 * train_examples_rate)]
    y_train = train_data['entityType'].values[:int(100000 * train_examples_rate)]
    X_dev = dev_data['words'].values[:20000]
    y_dev = dev_data['entityType'].values[:20000]
    X_test = test_data['words'].values[:20000]
    y_test = test_data['entityType'].values[:20000]

    X_train = np.asarray(transform_input(X_train, w2v_model.vocab))
    X_dev = np.asarray(transform_input(X_dev, w2v_model.vocab))
    X_test = np.asarray(transform_input(X_test, w2v_model.vocab))

    y_train = [tagToInt(y) for y in y_train]  # this transformation is needed to apply
    y_dev = [tagToInt(y) for y in y_dev]     # to_categorical() keras method
    y_test = [tagToInt(y) for y in y_test]

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_dev = to_categorical(y_dev, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def build_model(h_params, input_shape, w2v_model):
    ''' shape: A shape tuple (integer), not including the batch size.
        For instance, `shape=(32,)` indicates that the expected input
        will be batches of 32-dimensional vectors.'''
    inp = Input(shape=input_shape)

    emb = Embedding(len(w2v_model.vocab),  # Vocabulary size
                    w2v_model.vector_size,  # Embedding size
                    weights=[w2v_model.vectors],  # Word vectors
                    trainable=False  # This indicates the word vectors must not be changed
                                     # during training.
                    )(inp)  # The output here has shape (batch_size (?), words_in_reviews (?), embedding_size)

    # Specify each convolution layer and their kernel size i.e. n-grams
    conv1_1 = Conv1D(filters=h_params.num_filters, kernel_size=2, activation='relu')(emb)
    btch1_1 = BatchNormalization()(conv1_1)
    maxp1_1 = MaxPooling1D(pool_size=h_params.pool_size)(btch1_1)
    flat1_1 = Flatten()(maxp1_1)

    conv1_2 = Conv1D(filters=h_params.num_filters, kernel_size=3, activation='relu')(emb)
    btch1_2 = BatchNormalization()(conv1_2)
    maxp1_2 = MaxPooling1D(pool_size=h_params.pool_size)(btch1_2)
    flat1_2 = Flatten()(maxp1_2)

    conv1_3 = Conv1D(filters=h_params.num_filters, kernel_size=4, activation='relu')(emb)
    btch1_3 = BatchNormalization()(conv1_3)
    maxp1_3 = MaxPooling1D(pool_size=h_params.pool_size)(btch1_3)
    flat1_3 = Flatten()(maxp1_3)

    # Gather all convolution layers
    cnct = concatenate([flat1_1, flat1_2, flat1_3], axis=1)
    drp1 = Dropout(h_params.drop)(cnct)

    dns1 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(h_params.l2))(drp1)
    out = Dense(NUM_CLASSES, activation='softmax')(dns1)

    model = Model(inputs=inp, outputs=out)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CNN baseline experiment')
    parser.add_argument('--train_examples_rate',
                        default=1,
                        type=float,
                        help='1 -> 100%,  0.25 -> 25%, etc.')
    parser.add_argument('--num_filters',
                        default=10,
                        type=int,
                        help='Number of convolution filters per kernel_size.')
    parser.add_argument('--pool_size',
                        default=2,
                        type=int,
                        help='Pool size')
    parser.add_argument('--drop',
                        default=0,
                        type=float,
                        help='Randomly fraction rate of input units to 0'
                             'at each update during training time')
    parser.add_argument('--l2',
                        default=0.1,
                        type=float,
                        help='L2 kernel regularizer')
    parser.add_argument('--batch_size',
                        default=512,
                        type=int,
                        help='Number of instances in each batch.')
    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='Number of training epochs.')

    args = parser.parse_args()

    train_examples_rate = args.train_examples_rate
    num_filters = args.num_filters
    pool_size = args.pool_size
    drop = args.drop
    l2 = args.l2
    batch_size = args.batch_size
    epochs = args.epochs

    print('Loading dataset...')
    train_data, dev_data, test_data = load_dataset()

    print('Loading word2vec model...')
    w2v_model = KeyedVectors.load('/home/ekokic/thesis/models/google/word2vecGoogle.model')

    print('Preprocessing input data...')
    X_train, X_dev, X_test, y_train, y_dev, y_test = preprocess_data(train_data, dev_data,
                                                                     test_data, w2v_model,
                                                                     train_examples_rate)
    print('Building model...')
    input_shape = (X_train.shape[1], )  # == 2 * W + 1
    model = build_model(args, input_shape, w2v_model)

    print('Compiling model...')
    model.compile(loss=categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=['accuracy'])

    print('Training model...')
    args = list(vars(args).items())
    experiment_name = 'cnn_wide_supervised'
    for key, value in args:
        experiment_name += ('_' + str(key) + '_' + str(value))
    if not os.path.exists('experiments' + '/' + experiment_name):
        os.makedirs('experiments' + '/' + experiment_name)

    callbacks = [
        ModelCheckpoint(filepath=os.path.join('experiments', experiment_name,
                                              'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                        monitor='val_acc',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=False,
                        mode='auto'),
        CSVLogger(filename=os.path.join('experiments', experiment_name, 'train_logs.csv'),
                  separator=',',
                  append=False)
    ]
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_dev, y_dev),
                        callbacks=callbacks)

    # print('Saving model metrics...')
    # train_loss = np.array(history.history['loss'])
    # dev_loss = np.array(history.history['val_loss'])
    # train_acc = np.array(history.history['acc'])
    # dev_acc = np.array(history.history['val_acc'])

    # if not path.exists('/home/ekokic/thesis/models/experiments/cnn_wide_metrics.csv'):
    #     with open('/home/ekokic/thesis/models/experiments/cnn_wide_metrics.csv', 'w', newline='') as csvfile:
    #         metric_writer = csv.writer(csvfile, delimiter=',')
    #         metric_writer.writerow(['model_name', 'train_loss', 'dev_loss', 'train_acc', 'dev_acc'])

    # with open('/home/ekokic/thesis/models/experiments/cnn_wide_metrics.csv', 'a', newline='') as csvfile:
    #     metric_writer = csv.writer(csvfile, delimiter=',')
    #     metric_writer.writerow([experiment_name, train_loss, dev_loss, train_acc, dev_acc])

    # history_df = pd.DataFrame({'train_loss': train_loss, 'dev_loss': dev_loss,
    #                            'train_acc': train_acc, 'val_acc': val_acc})
    # history_df.to_csv('./thesis/models/history/model_{}.h5'.format(experiment_name))

    # np.savetxt('./thesis/models/losses/model_{}.h5'.format(experiment_name), loss_history, delimiter=",")
    # np.savetxt('./thesis/models/accs/model_{}.h5'.format(experiment_name), acc_history, delimiter=",")

# TODO 4: Evaluate the model, calculating the metrics.
# Option 1: Use the model.evaluate() method. For this, the model must be
# already compiled with the metrics.
# performance = model.evaluate(transform_input(X_test), y_test)

# Option 2: Use the model.predict() method and calculate the metrics using
# sklearn. We recommend this, because you can store the predictions if
# you need more analysis later. Also, if you calculate the metrics on a
# notebook, then you can compare multiple classifiers.
# predictions = ...
# performance = ...

# TODO 5: Save the results.
# ...

# One way to store the predictions:
# results = pandas.DataFrame(y_test_orginal, columns=['true_label'])
# results.loc[:, 'predicted'] = predictions
# results.to_csv('predictions_{}.csv'.format(args.experiment_name),
#                index=False)

    # print('Saving model...')
    # model.save_weights('./thesis/models/weights/{}.h5'.format(experiment_name))
    # model.save('~/thesis/models/saved/{}.h5'.format(experiment_name))
