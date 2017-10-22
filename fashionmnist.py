import gzip
import os
import struct

import numpy
import six

from chainer.dataset import download
from chainer.datasets import tuple_dataset


def get_fmnist(withlabel=True, ndim=1, scale=1., dtype=numpy.float32,
              label_dtype=numpy.int32, rgb_format=False):
    """Gets the Fashion-MNIST dataset.
    `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_
    is a dataset of Zalando's article images.
    In the same way as MNIST, each images represented by grey-scale 28x28 images.
    In the original images, each pixel is represented by one-byte unsigned integer.
    This function scales the pixels to floating point values in the interval ``[0, scale]``.
    This function returns the training set and the test set of the official
    Fashion-MNIST dataset. If ``withlabel`` is ``True``, each dataset consists of
    tuples of images and labels, otherwise it only consists of images.
    Args:
        withlabel (bool): If ``True``, it returns datasets with labels. In this
            case, each example is a tuple of an image and a label. Otherwise,
            the datasets only contain images.
        ndim (int): Number of dimensions of each image. The shape of each image
            is determined depending on ``ndim`` as follows:
            - ``ndim == 1``: the shape is ``(784,)``
            - ``ndim == 2``: the shape is ``(28, 28)``
            - ``ndim == 3``: the shape is ``(1, 28, 28)``
        scale (float): Pixel value scale. If it is 1 (default), pixels are
            scaled to the interval ``[0, 1]``.
        dtype: Data type of resulting image arrays.
        label_dtype: Data type of the labels.
        rgb_format (bool): if ``ndim == 3`` and ``rgb_format`` is ``True``, the
            image will be converted to rgb format by duplicating the channels
            so the image shape is (3, 28, 28). Default is ``False``.
    Returns:
        A tuple of two datasets. If ``withlabel`` is ``True``, both datasets
        are :class:`~chainer.datasets.TupleDataset` instances. Otherwise, both
        datasets are arrays of images.
    """
    train_raw = _retrieve_fmnist_training()
    train = _preprocess_fmnist(train_raw, withlabel, ndim, scale, dtype,
                               label_dtype, rgb_format)
    test_raw = _retrieve_fmnist_test()
    test = _preprocess_fmnist(test_raw, withlabel, ndim, scale, dtype,
                              label_dtype, rgb_format)
    return train, test


def _preprocess_fmnist(raw, withlabel, ndim, scale, image_dtype, label_dtype,
                       rgb_format):
    images = raw['x']
    if ndim == 2:
        images = images.reshape(-1, 28, 28)
    elif ndim == 3:
        images = images.reshape(-1, 1, 28, 28)
        if rgb_format:
            images = numpy.broadcast_to(images,
                                        (len(images), 3) + images.shape[2:])
    elif ndim != 1:
        raise ValueError('invalid ndim for FMNIST dataset')
    images = images.astype(image_dtype)
    images *= scale / 255.

    if withlabel:
        labels = raw['y'].astype(label_dtype)
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images


def _retrieve_fmnist_training():
    urls = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz']
    return _retrieve_fmnist('train.npz', urls)


def _retrieve_fmnist_test():
    urls = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz']
    return _retrieve_fmnist('test.npz', urls)


def _retrieve_fmnist(name, urls):
    root = download.get_dataset_directory('fmnist/fmnist')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, urls), numpy.load)


def _make_npz(path, urls):
    x_url, y_url = urls
    x_path = download.cached_download(x_url)
    y_path = download.cached_download(y_url)

    with gzip.open(x_path, 'rb') as imgpath, gzip.open(y_path, 'rb') as lblpath:
        labels = numpy.frombuffer(lblpath.read(), dtype=numpy.uint8,
                                  offset=8)
        images = numpy.frombuffer(imgpath.read(), dtype=numpy.uint8,
                                  offset=16).reshape(len(labels), 784)
    numpy.savez_compressed(path, x=images, y=labels)
    return {'x': images, 'y': labels}
