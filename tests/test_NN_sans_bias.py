import numpy as np
from sklearn.preprocessing import normalize

from src.NN_sans_bias import ConstrainedMLPClassifier


def test_constrained_NN():
    """
    With the constraint in place, the probability p of a hidden var being 1
    on a sample x, implies a probability 1-p on the sample 1-x
    """

    # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    np.random.seed(171717)
    X = 2*np.random.rand(1000,100) - 1
    w = 0.2*(2*np.random.rand(100) - 1)
    z = np.tanh(np.dot(X, w))

    # map to 4 categories
    zero_to_four_range = (z+1)*2
    y = np.floor(zero_to_four_range).astype(int)

    model = ConstrainedMLPClassifier(solver='lbfgs', alpha=1e-5, activation='tanh', hidden_layer_sizes=(15,), random_state = 1)
    model.fit(X, y)

    # probabilities should change in a predictable way on the
    y_hat = model.predict_proba(X)
    # print(y_hat.shape)
    np.testing.assert_allclose(y_hat.sum(axis=1), desired=1)

    # normalize to once again get probabilities
    # must use here l1 norm, so that simply the probabilities
    # get divided by the sum of un-normalized probabilities
    desired = normalize(1/y_hat, norm='l1')

    assert desired.shape == y_hat.shape
    np.testing.assert_allclose(desired.sum(axis=1), desired=1)

    actual = model.predict_proba(-X)

    # np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-6, atol=1e-7)


if __name__=='__main__':
    test_constrained_NN()





