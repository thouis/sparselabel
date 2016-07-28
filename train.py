import sys
import os
import glob
import time

import h5py
import imread
import skimage.measure

import numpy as np

import malis as m

from keras.layers.core import Layer, Activation, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.models import Model, Sequential
from keras import backend as K

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


def update_predictions(images, affinities, model):
    num_images = images.shape[0]
    st = time.time()
    for idx in range(num_images):
        input_volume = np.stack([images[min(max(idx + offset, 0), num_images - 1), ...]
                                 for offset in range(-2, 3)],
                                axis=0)
        pred = model.predict_on_batch(input_volume[np.newaxis, ...])
        affinities[:, idx, ...] = pred
    print("Prediction took {} seconds".format(int(time.time() - st)))


def compute_malis_counts(affinities, labels):
    nhood = m.mknhood3d(radius=1)
    assert nhood.shape[0] == affinities.shape[0]
    subvolume_shape = labels.shape
    node_idx_1, node_idx_2 = m.nodelist_like(subvolume_shape, nhood)
    node_idx_1, node_idx_2 = node_idx_1.ravel(), node_idx_2.ravel()
    print (node_idx_1[:10], node_idx_2[:10], labels.shape)
    flat_labels = labels[...].ravel()
    flat_affinties = affinities[...].ravel()
    pos_counts = m.malis_loss_weights(flat_labels,
                                      node_idx_1, node_idx_2,
                                      flat_affinties,
                                      1)
    neg_counts = m.malis_loss_weights(flat_labels,
                                      node_idx_1, node_idx_2,
                                      flat_affinties,
                                      0)
    pos_counts = pos_counts.reshape(affinities.shape)
    neg_counts = neg_counts.reshape(affinities.shape)
    return pos_counts, neg_counts

def err_and_deriv(affinities, pos_counts, neg_counts, idx=74118595):
    V_Rand_split = (affinities ** 2) * pos_counts / pos_counts.sum()
    V_Rand_merge = ((1.0 - affinities) ** 2) * neg_counts / neg_counts.sum()
    sum_VRS = V_Rand_split.sum()
    sum_VRM = V_Rand_merge.sum()
    print (sum_VRS, sum_VRM, V_Rand_split.ravel()[idx], V_Rand_merge.ravel()[idx])
    err = 2 * sum_VRS * sum_VRM / (sum_VRS + sum_VRM)
    d_VRS_d_aff = 2 * affinities * pos_counts / pos_counts.sum()
    d_VRM_d_aff = (2 * affinities - 2) * neg_counts / neg_counts.sum()
    d_err_d_aff = 2 * ((d_VRS_d_aff * sum_VRM  ** 2 + d_VRM_d_aff * sum_VRS ** 2) / 
                       (sum_VRS + sum_VRM) ** 2)
    print (d_VRS_d_aff.ravel()[idx], d_VRM_d_aff.ravel()[idx], d_err_d_aff.ravel()[idx])
    return err, d_err_d_aff


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

    affinities = h5.require_dataset('affinities', (3,) + images.shape, dtype=np.float32)

    update_predictions(images, affinities, model)
    pos_counts, neg_counts = compute_malis_counts(affinities, labels)

    err, d_err_d_aff = err_and_deriv(affinities[...], pos_counts, neg_counts)
