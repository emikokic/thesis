
from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd
import ast
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors


def transform_input(instances, mapping, maxlen):
    """Replaces the words in instances with their index in mapping.
    Args:
        instances: a list of text instances.
        mapping: an dictionary from words to indices.
    Returns:
        A matrix with shape (n_instances, m_words)."""
    word_indices = []
    for instance in instances:
        l = []
        for word in instance:
            try:
                l.append(mapping[word].index)
            except KeyError:
                l.append(0)  # index to '</s>' word vector
        word_indices.append(l)

    return pad_sequences(word_indices, maxlen=maxlen)


def tagToInt(tag):
    return {'PER': 0, 'LOC': 1, 'ORG': 2, 'MISC': 3, 'O': 4}[tag]


def preprocess_data(train_data, val_data, test_data, w2v_model, maxlen, num_classes):

    X_train = train_data['sentence'].map(lambda sen: np.array(ast.literal_eval(sen))).values
    y_train = train_data['entities'].map(lambda sen: np.array(ast.literal_eval(sen))).values
    X_val = val_data['sentence'].map(lambda sen: np.array(ast.literal_eval(sen))).values
    y_val = val_data['entities'].map(lambda sen: np.array(ast.literal_eval(sen))).values
    X_test = test_data['sentence'].map(lambda sen: np.array(ast.literal_eval(sen))).values
    y_test = test_data['entities'].map(lambda sen: np.array(ast.literal_eval(sen))).values

    def fun(y): return tagToInt(y)
    y_train = np.array([np.array(list(map(fun, labels))) for labels in y_train])
    y_val = np.array([np.array(list(map(fun, labels))) for labels in y_val])
    y_test = np.array([np.array(list(map(fun, labels))) for labels in y_test])

    X_train = transform_input(X_train, w2v_model.vocab, maxlen)
    X_val = transform_input(X_val, w2v_model.vocab, maxlen)
    X_test = transform_input(X_test, w2v_model.vocab, maxlen)

    # # convert class vectors to binary class matrices
    # y_train = np.array([to_categorical(labels, num_classes) for labels in y_train])
    # y_val = np.array([to_categorical(labels, num_classes) for labels in y_val])
    # y_test = np.array([to_categorical(labels, num_classes) for labels in y_test])

    # padding target labels
    y_train = pad_sequences(y_train, maxlen=maxlen,  # [0,0,0,0,1] one-hot encoding for 'O' label
                            # value=np.array([0, 0, 0, 0, 1], dtype='float32'))
                            value=4)  # 'O' tag representation
    y_val = pad_sequences(y_val, maxlen=maxlen,  # [0,0,0,0,1] one-hot encoding for 'O' label
                          # value=np.array([0, 0, 0, 0, 1], dtype='float32'))
                          value=4)  # 'O' tag representation
    y_test = pad_sequences(y_test, maxlen=maxlen,  # [0,0,0,0,1] one-hot encoding for 'O' label
                           # value=np.array([0, 0, 0, 0, 1], dtype='float32'))
                           value=4)  # 'O' tag representation

    return X_train, X_val, X_test, y_train, y_val, y_test


class DataSet(object):

    def __init__(self, instances, labels):
        assert len(labels) == 0 or instances.shape[0] == labels.shape[0], (
            "instances.shape: %s labels.shape: %s" % (instances.shape,
                                                      labels.shape))
        self._num_examples = instances.shape[0]
        self._instances = instances
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def instances(self):
        return self._instances

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._instances = self._instances[perm]
            if len(self._labels) > 0:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        if len(self._labels) > 0:
            return self._instances[start:end], self._labels[start:end]
        else:
            return self._instances[start:end], None


class SemiDataSet(object):
    def __init__(self, instances, labels, n_labeled, n_classes):
        self.n_labeled = n_labeled

        # Unlabled DataSet
        self.unlabeled_ds = DataSet(instances, [])
        self.num_examples = self.unlabeled_ds.num_examples

        # Labeled DataSet
        # indices = np.arange(self.num_examples)
        # shuffled_indices = np.random.permutation(indices)
        # instances = instances[shuffled_indices]
        # labels = labels[shuffled_indices]
        # y = np.array([np.arange(n_classes)[lbl == 1][0] for lbl in labels])
        # n_from_each_class = n_labeled // n_classes
        # i_labeled = []
        # for c in range(n_classes):
        #     i = indices[y == c][:n_from_each_class]
        #     i_labeled += list(i)
        # l_instances = instances[i_labeled]
        # l_labels = labels[i_labeled]
        # self.labeled_ds = DataSet(l_instances, l_labels)

        # Experimento 3 ##### (descomentar lo siguiente solo si se quiere excluir que
        # los datos anotados se utilicen como anotados)
        # # Unlabled DataSet
        # self.unlabeled_ds = DataSet(instances[n_labeled:], [])
        # self.num_examples = instances.shape[0]  # self.unlabeled_ds.num_examples

        # Labeled DataSet
        l_instances = instances[0:n_labeled]
        l_labels = labels[0:n_labeled]
        self.labeled_ds = DataSet(l_instances, l_labels)

    def next_batch(self, batch_size):
        unlabeled_instances, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_instances, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_instances, labels = self.labeled_ds.next_batch(batch_size)
        return labeled_instances, labels, unlabeled_instances


def read_data_sets(data_path, n_classes, n_labeled=100, maxlen=None):
    class DataSets(object):
        pass

    data_sets = DataSets()

    print('Loading dataset...')
    train_data = pd.read_csv('%ssen_entities_max_len_30_train.csv' % data_path)
    val_data = pd.read_csv('%ssen_entities_max_len_30_val.csv' % data_path)
    test_data = pd.read_csv('%ssen_entities_max_len_30_test.csv' % data_path)

    print('Loading word2vec model...')
    w2v_model = KeyedVectors.load('/home/ekokic/thesis/models/google/word2vecGoogle.model')

    print('Preprocessing input data...')
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(train_data, val_data, test_data,
                                                                     w2v_model, maxlen, n_classes)

    print('X_train shape', X_train.shape)
    print('y_train shape', y_train.shape)
    print('n_labeled', n_labeled)
    print('n_classes', n_classes)

    data_sets.train = SemiDataSet(X_train, y_train, n_labeled, n_classes)
    data_sets.validation = DataSet(X_val, y_val)
    data_sets.test = DataSet(X_test, y_test)

    return data_sets, w2v_model
