import sys
import os
import glob

import h5py
import imread
import json
from scipy import ndimage

from keras.layers.core import Layer, Activation, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.models import Model, Sequential
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.optimizers import SGD
from keras import callbacks


import numpy as np

def residual_block(input, num_feature_maps, filter_size=3):
    conv_1 = BatchNormalization(axis=1, mode=2)(input)
    conv_1 = Activation('relu')(conv_1)
    conv_1 = Convolution2D(num_feature_maps, filter_size, filter_size,
                           border_mode='same', bias=False)(conv_1)

    conv_2 = Dropout(0.1)(conv_1)
    conv_2 = BatchNormalization(axis=1, mode=2)(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_2 = Convolution2D(num_feature_maps, filter_size, filter_size,
                           border_mode='same', bias=False)(conv_2)

    return merge([input, conv_2], mode='sum')


def residual_chain(input, num_blocks, num_features_maps, filter_size=3):
    output = input
    for idx in range(num_blocks):
        output = residual_block(output, num_features_maps, filter_size)
    return output

def load_image_volume(image_directory, h5, extension='tif'):
    filenames = sorted(glob.glob(os.path.join(image_directory, '*.' + extension)))
    size = imread.imread(filenames[0]).shape
    images = h5.require_dataset('images', shape=((len(filenames),) + size), dtype=np.uint8)
    for idx, f in enumerate(filenames):
        images[idx, ...] = imread.imread(f)
    return images

def load_labels(label_directory, num_labels, size, h5):
    filenames = [os.path.join(label_directory, 'labels_{:06d}.json'.format(idx + 1)) for idx in range(num_labels)]
    dists = h5.require_dataset('dists', shape=((len(filenames),) + size), dtype=np.float32)
    for idx, f in enumerate(filenames):
        if os.path.exists(f):
            labels = json.load(open(f, "r"))
            mask = np.ones(size)
            for l in labels:
                ii, jj = zip(*l)
                mask[ii, jj] = 0
            dists[idx, ...] = ndimage.distance_transform_edt(mask)
        else:
            dists[idx] = np.inf
                
    return dists

def image_L1_abs_loss(y_true, y_pred, max_true_dist=3):
    mask = y_true <= max_true_dist
    mask_counts = K.sum(mask, axis=[1, 2, 3]) + 0.1
    masked_abs_diff_abs = mask * K.abs(K.abs(y_true) - K.abs(y_pred))
    L1 = K.sum(masked_abs_diff_abs, axis=[1, 2, 3]) / mask_counts
    return K.mean(L1, axis=-1)

def generate_single_sample(image, dists, input_shape, min_zeros=20):
    while True:
        idx = np.random.choice(image.shape[0])
        i = np.random.choice(image.shape[1] - input_shape[1])
        j = np.random.choice(image.shape[2] - input_shape[2])

        i_slice = slice(i, i + input_shape[1])
        j_slice = slice(j, j + input_shape[2])
        # verify that we have enough zero pixels to train on
        true_dist = dists[idx, i_slice, j_slice]
        if np.sum(true_dist < 1) < min_zeros:
            continue

        half = input_shape[0] // 2
        src_indices = [min(max(0, si + idx), image.shape[0] - 1) for si in range(- half, half + 1)]
        in_images = [images[si, i_slice, j_slice] for si in src_indices]
        
        yield np.stack(in_images), true_dist[np.newaxis, ...]

def generate_batched_samples(image, dists, input_shape, batchsize):
    gen = generate_single_sample(image, dists, input_shape)
    while True:
        images = np.zeros((batchsize,) + input_shape, dtype=np.float32)
        dists = np.zeros((batchsize, 1) + input_shape[1:], dtype=np.float32)
        for bidx in range(batchsize):
            im, d = next(gen)
            images[bidx, ...] = im
            dists[bidx, ...] = d
        yield images, dists


class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        print("Loss", logs.get('loss'))
        self.losses.append(logs.get('loss'))

if __name__ == '__main__':
    h5 = h5py.File('input_labels_output.hdf5')
    images = load_image_volume(sys.argv[1], h5)
    dists = load_labels(sys.argv[2], images.shape[0], images.shape[1:], h5)

    num_images = images.shape[0]
    num_feature_maps = 64
    num_output_per_voxel = 3

    INPUT_SHAPE = (5, 128, 128)

    x = Input(shape=INPUT_SHAPE)
    pre = Convolution2D(num_feature_maps, 5, 5, bias=True, border_mode='same')(x)
    post = residual_chain(pre, 3, num_feature_maps)
    # 3 outputs per voxel, 3x3 final filter
    output = Convolution2D(1, 3, 3, activation=None, border_mode='same')(post)

    model = Model(input=x, output=output)
    model.compile(loss=image_L1_abs_loss, optimizer=SGD(lr=0.0001, momentum=0.9))

    model.fit_generator(generate_batched_samples(images, dists, INPUT_SHAPE, 16),
                        nb_epoch=100,
                        samples_per_epoch=100,
                        verbose=2,
                        callbacks=[LossHistory()])
