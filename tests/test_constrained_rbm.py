import numpy as np
from src.simple_constrained_rbm import ConstrainedBernoulliRBM


def test_constrained_RBM():
    """
    With the constraint in place, the probability p of a hidden var being 1
    on a sample x, implies a probability 1-p on the sample 1-x
    """

    # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    np.random.seed(171717)
    X = np.random.rand(100,1000)
    model = ConstrainedBernoulliRBM(n_components=2)
    model.fit(X)

    # probabilities for hidden==1 on the original dataset
    y1 = model.transform(X)
    y2 = model.transform(1-X)

    np.testing.assert_allclose(actual=1-y1, desired=y2, rtol=1e-6, atol=1e-7)


if __name__=='__main__':
    test_constrained_RBM()
