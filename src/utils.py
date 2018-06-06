from scipy.ndimage import convolve
import numpy as np
import matplotlib.pyplot as plt


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                     weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


def plot_sample(x, img_size=(8, 8)):
    plt.figure(figsize=(4.2, 4))
    plt.imshow(x.reshape(img_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

def inversion_symmetric_features(X, image_shape=(8,8)):
    n_samples, n_features = X.shape
    assert n_features==image_shape[0]*image_shape[1]
    _X = X.reshape(n_samples, *image_shape)
    result = _X*np.roll(_X, shift=-1, axis=2)
    return result.reshape(n_samples, n_features)
