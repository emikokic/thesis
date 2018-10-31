
from __future__ import print_function
# import gzip
# import os
# import urllib
import keras
import numpy as np

# SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


# def maybe_download(filename, work_directory):
#   """Download the data from Yann's website, unless it's already here."""
#   if not os.path.exists(work_directory):
#     os.mkdir(work_directory)
#   filepath = os.path.join(work_directory, filename)
#   if not os.path.exists(filepath):
#     filepath, _ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
#     statinfo = os.stat(filepath)
#     print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
#   return filepath


# def extract_images(filename):
#   """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
#   print('Extracting', filename)
#   with gzip.open(filename) as bytestream:
#     magic = _read32(bytestream)
#     if magic != 2051:
#       raise ValueError(
#           'Invalid magic number %d in MNIST image file: %s' %
#           (magic, filename))
#     num_images = _read32(bytestream)
#     rows = _read32(bytestream)
#     cols = _read32(bytestream)
#     buf = bytestream.read(rows * cols * num_images)
#     data = np.frombuffer(buf, dtype=np.uint8)
#     data = data.reshape(num_images, rows, cols, 1)
#     return data


def extract_words(filename, num_instances):

  print('Extracting', filename)
  word_vectors = np.load(filename)
  word_vectors = word_vectors.items()[0][1][:num_instances]

  return word_vectors


# def dense_to_one_hot(labels_dense, num_classes=5): # TODO: esto tal vez se podria reemplazar utilizando
#                                                    # keras.utils.to_categorical(y_train, num_classes)
#   """Convert class labels from scalars to one-hot vectors."""
#   num_labels = labels_dense.shape[0]
#   index_offset = np.arange(num_labels) * num_classes
#   labels_one_hot = np.zeros((num_labels, num_classes))
#   labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#   return labels_one_hot


# def extract_labels(filename, one_hot=False):
#   """Extract the labels into a 1D uint8 numpy array [index]."""
#   print('Extracting', filename)
#   with gzip.open(filename) as bytestream:
#     magic = _read32(bytestream)
#     if magic != 2049:
#       raise ValueError(
#           'Invalid magic number %d in MNIST label file: %s' %
#           (magic, filename))
#     num_items = _read32(bytestream)
#     buf = bytestream.read(num_items)
#     labels = np.frombuffer(buf, dtype=np.uint8)
#     if one_hot:
#       return dense_to_one_hot(labels)
#     return labels


def extract_labels(filename, one_hot=False, num_instances=0):

  print('Extracting', filename)
  entity_vector = np.load(filename)
  entities = entity_vector.items()[0][1][:num_instances]

  if one_hot:
    def labelToInt(label): return {'O': 0, 'PER': 1, 'ORG': 2, 'LOC': 3, 'MISC': 4}[label]
    entities = [labelToInt(entity) for entity in entities]
    entities = keras.utils.to_categorical(entities, 5)  # 5 == NUM_CLASSES

  return entities


class DataSet(object):

  def __init__(self, words, labels, fake_data=False):
    # if fake_data:
    #   self._num_examples = 10000
    # else:
    # assert images.shape[0] == labels.shape[0], (
    #     "images.shape: %s labels.shape: %s" % (images.shape,
    #                                            labels.shape))
    self._num_examples = words.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    # assert images.shape[3] == 1
    # images = images.reshape(images.shape[0],
    #                         images.shape[1] * images.shape[2])
    # Convert from [0, 255] -> [0.0, 1.0].
    # images = images.astype(np.float32)
    # images = np.multiply(images, 1.0 / 255.0)
    self._words = words
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def words(self):
    return self._words

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    # if fake_data:
    #   fake_image = [1.0 for _ in range(784)]
    #   fake_label = 0
    #   return [fake_image for _ in range(batch_size)], [
    #       fake_label for _ in range(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._words = self._words[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._words[start:end], self._labels[start:end]


class SemiDataSet(object):
  def __init__(self, words, labels, n_labeled):
    self.n_labeled = n_labeled

    # Unlabled DataSet
    self.unlabeled_ds = DataSet(words, labels)

    # Labeled DataSet
    self.num_examples = self.unlabeled_ds.num_examples
    indices = np.arange(self.num_examples)
    shuffled_indices = np.random.permutation(indices)
    words = words[shuffled_indices]
    labels = labels[shuffled_indices]
    n_classes = 5
    y = np.array([np.arange(n_classes)[l == 1][0] for l in labels])
    # idx = indices[y == 0][:5]
    # n_classes = y.max() + 1
    n_from_each_class = n_labeled / n_classes
    print('n_from_each_class', n_from_each_class)
    i_labeled = []
    for c in range(n_classes):
      i = indices[y == c][:int(n_from_each_class)]
      i_labeled += list(i)

    l_words = words[i_labeled]
    l_labels = labels[i_labeled]

    # print('l_words[0]', l_words[0])
    # print('l_labels[0]', l_labels[0])
    # print('l_labels[:10]', l_labels)

    # unique_elements, counts_elements = np.unique(l_labels, return_counts=True)
    # print("Frequency of unique values of the said array:")
    # print(np.asarray((unique_elements, counts_elements)))

    # DUDA: estoy utilizando mismos datos para sup como para no sup?
    # con el codigo anterior tambien pasa esto solo que con el agregado de que la eleccion
    # de los datos etiquetados es random?
    self.labeled_ds = DataSet(l_words, l_labels)  # DataSet(words[:n_labeled], labels[:n_labeled])

  def next_batch(self, batch_size):
    unlabeled_words, _ = self.unlabeled_ds.next_batch(batch_size)
    if batch_size > self.n_labeled:
      labeled_words, labels = self.labeled_ds.next_batch(self.n_labeled)
    else:
      labeled_words, labels = self.labeled_ds.next_batch(batch_size)
    words = np.vstack([labeled_words, unlabeled_words])
    return words, labels


def read_data_sets(train_dir, n_labeled=100, fake_data=False, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()

  # if fake_data:
  #   data_sets.train = DataSet([], [], fake_data=True)
  #   data_sets.validation = DataSet([], [], fake_data=True)
  #   data_sets.test = DataSet([], [], fake_data=True)
  #   return data_sets

  # TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  # TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  # TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  # TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  # VALIDATION_SIZE = 0

  # local_file = maybe_download(TRAIN_IMAGES, train_dir)
  # train_images = extract_images(local_file)

  X_train = extract_words('/users/ekokic/thesis/corpus_WiNER/word_vectors/wv_train_exp_decay_W_5.npz',
                          num_instances=100000)
  X_dev = extract_words('/users/ekokic/thesis/corpus_WiNER/word_vectors/wv_dev_exp_decay_W_5.npz',
                        num_instances=20000)
  X_test = extract_words('/users/ekokic/thesis/corpus_WiNER/word_vectors/wv_test_exp_decay_W_5.npz',
                         num_instances=20000)

  # local_file = maybe_download(TRAIN_LABELS, train_dir)
  y_train = extract_labels('/users/ekokic/thesis/corpus_WiNER/entity_vectors/ev_train_exp_decay_W_5.npz',
                           one_hot=one_hot, num_instances=100000)
  y_dev = extract_labels('/users/ekokic/thesis/corpus_WiNER/entity_vectors/ev_dev_exp_decay_W_5.npz',
                         one_hot=one_hot, num_instances=20000)
  y_test = extract_labels('/users/ekokic/thesis/corpus_WiNER/entity_vectors/ev_test_exp_decay_W_5.npz',
                          one_hot=one_hot, num_instances=20000)

  print('X_train len', len(X_train))
  print('X_dev len', len(X_dev))
  print('X_test len', len(X_test))
  # local_file = maybe_download(TEST_IMAGES, train_dir)
  # test_images = extract_images(local_file)

  # local_file = maybe_download(TEST_LABELS, train_dir)
  # test_labels = extract_labels(local_file, one_hot=one_hot)

  # validation_images = train_images[:VALIDATION_SIZE]
  # validation_labels = train_labels[:VALIDATION_SIZE]
  # train_images = train_images[VALIDATION_SIZE:]
  # train_labels = train_labels[VALIDATION_SIZE:]

  # data_sets.train = SemiDataSet(train_images, train_labels, n_labeled)
  # data_sets.validation = DataSet(validation_images, validation_labels)
  # data_sets.test = DataSet(test_images, test_labels)
  data_sets.train = SemiDataSet(X_train, y_train, n_labeled)
  data_sets.validation = DataSet(X_dev, y_dev)
  data_sets.test = DataSet(X_test, y_test)

  return data_sets
