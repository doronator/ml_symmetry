import numpy as np

from src.utils import inversion_symmetric_features


def test_inversion_symmetric_features():
    eight = np.array([[0, 0, 1, 1, 0, 0],
                     [0, 1, 0, 0, 1, 0],
                     [0, 1, 0, 0, 1, 0],
                     [0, 1, 1, 1, 1, 0],
                     [0, 1, 0, 0, 1, 0],
                     [0, 0, 1, 1, 0, 0]
                     ])

    six = np.array([[0, 0, 1, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 1, 1, 1, 0, 0],
                     [0, 1, 0, 0, 1, 0],
                     [0, 0, 1, 1, 0, 0]
                     ])

    X = np.array([2*eight-1, 2*six-1]).reshape(2,36)

    expected_eight = np.array([
        [1, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1]
        ])*2 - 1

    expected_six = np.array([
        [1, 0, 1, 0, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1]
        ])*2 - 1

    XX = inversion_symmetric_features(X, image_shape=(6, 6))
    actual_eight = XX[0, :].reshape(6,6)
    actual_six = XX[1, :].reshape(6,6)

    np.testing.assert_allclose(actual=actual_eight, desired=expected_eight)
    np.testing.assert_allclose(actual=actual_six, desired=expected_six)


if __name__ == '__main__':
    test_inversion_symmetric_features()

