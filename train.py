import sys
import os
import glob

import h5py
import imread
import skimage.measure

from keras.layers.core import Layer, Activation, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.models import Model, Sequential
from keras import backend as K
from keras.utils.visualize_util import plot

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

def load_label_volume(label_directory, num_labels, h5):
    filenames = [os.path.join(label_directory, 'labels_{:06d}.png'.format(idx + 1)) for idx in range(num_labels)]
    size = imread.imread(filenames[0]).shape
    labels = h5.require_dataset('labels', shape=((len(filenames),) + size), dtype=np.int32)
    for idx, f in enumerate(filenames):
        if os.path.exists(f):
            labels[idx, ...] = imread.imread(f)

    # use skimage to label individual objects
    relabel = skimage.measure.label(labels[...], background=0)
    labels[...] = relabel
    return labels

if __name__ == '__main__':
    h5 = h5py.File('input_labels_output.hdf5')
    images = load_image_volume(sys.argv[1], h5)
    labels = load_label_volume(sys.argv[2], images.shape[0], h5)

    num_images = images.shape[0]
    num_feature_maps = 64
    num_output_per_voxel = 3

    VOLUME_SHAPE = (5, 1024, 1024)
    INPUT_SHAPE = VOLUME_SHAPE

    x = Input(shape=INPUT_SHAPE)
    pre = Convolution2D(num_feature_maps, 5, 5, bias=True, border_mode='same')(x)
    post = residual_chain(pre, 3, num_feature_maps)
    # 3 outputs per voxel, 3x3 final filter
    output = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(post)

    model = Model(input=x, output=output)

    affinities = h5.require_dataset('affinities', (num_images, 3) + images.shape[1:], dtype=np.float32)

    for idx in range(num_images):
        print("pred", idx)
        input_volume = np.stack([images[min(max(idx + offset, 0), num_images - 1), ...]
                                 for offset in range(-2, 3)],
                                axis=0)
        pred = model.predict_on_batch(input_volume[np.newaxis, ...])
        print(pred.shape)
        affinities[idx, ...] = pred
