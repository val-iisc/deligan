"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
from os import listdir
from os.path import isdir, join
import numpy as np
import numpy 
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from random import randint
import random
from random import shuffle
import cv2
random.seed(0)
total=0


class DataSet(object):

    def __init__(self, images, labels, one_hot=False,
                 dtype=np.float32):
        """Construct a DataSet.

        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        assert images.shape[0] == labels.shape[0], (
              'images.shape: %s labels.shape: %s' % (images.shape,
                                                     labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] , images.shape[2])
        if dtype == np.float32:
          # Convert from [0, 255] -> [0.0, 1.0].
          images = images.astype(numpy.float32)
          images = numpy.multiply(images, 1.0 / 255.0)
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        images = images[perm]
        labels = labels[perm]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

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
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        #print(self._images.shape)
        return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=np.float32):
    class DataSets(object):
        pass
    data_sets = DataSets()

    train_fraction=1
    validate_fraction=1

    data_dir=train_dir
    rootDir=data_dir
    imagelist = []
    dir_id=1
    count=0
    aug=2
    num_images=0

    for dirName, subdirList, fileList in os.walk(rootDir):
        if dir_id>0:count+=1
        dir_id+=1
        for fname in fileList:
            imagelist.append(dirName+"/"+fname)
            num_images+=1

    data=[]
    label=[]
    image_count=0
    img = np.array(imagelist)
    print(img.shape)
    for image_name in imagelist:

        image=cv2.imread(image_name,0)
        ret,image=cv2.threshold(image,250,255,cv2.THRESH_BINARY_INV)
        kernel=np.ones((3,3),np.uint8)
        image=cv2.dilate(image,kernel)
        ret,image=cv2.threshold(image,20,255,cv2.THRESH_BINARY)
        image=cv2.resize(image, (32,32), interpolation=cv2.INTER_AREA)
        folder_name,file_name = image_name.split('/')[-2:]
        data.append(image)
        label.append(1)
        image_flip=cv2.flip(image,1)
        data.append(image_flip)
        label.append(1)
        image_count+=1
    label=np.array(label)
    data=np.array(data)
    data=data.reshape(data.shape[0],data.shape[1],data.shape[2],1)

    TRAIN_SIZE=int(train_fraction*data.shape[0])
    VALIDATION_SIZE =int(validate_fraction*data.shape[0])

    train_images=data[:TRAIN_SIZE] 
    train_labels=label[:TRAIN_SIZE]
    test_images=data[TRAIN_SIZE:]
    test_labels=label[TRAIN_SIZE:]

    idx_train=random.sample(range(train_images.shape[0]),train_images.shape[0])
    train_images=train_images[idx_train]
    train_labels=train_labels[idx_train]

    idx_test=random.sample(range(test_images.shape[0]),test_images.shape[0])
    test_images=test_images[idx_test]
    test_labels=test_labels[idx_test]

    validation_images = test_images[:VALIDATION_SIZE]
    validation_labels = test_labels[:VALIDATION_SIZE]
    test_images=test_images[VALIDATION_SIZE:]
    test_labels=test_labels[VALIDATION_SIZE:]

    print((np.array(imagelist).shape))
    print((train_labels.shape))
    print((train_images.shape))
    print((validation_images.shape))
    print((test_images.shape))
    data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
   # data_sets.validation = DataSet(validation_images,  validation_labels, dtype=dtype)
   # data_sets.test = DataSet(test_images, test_labels, dtype=dtype)

    return data_sets
