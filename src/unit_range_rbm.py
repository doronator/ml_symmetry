"""Restricted Boltzmann Machine
reformulated to optionally work with visible and hidden variables in the [-1,+1] range (unit range)
and top optionally work without intercept terms
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.special import expit  # logistic function

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils import issparse
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import log_logistic
from sklearn.utils.validation import check_is_fitted


class BernoulliRBM(BaseEstimator, TransformerMixin):
    """Bernoulli Restricted Boltzmann Machine (RBM).

    A Restricted Boltzmann Machine with binary visible units and
    binary hidden units. Parameters are estimated using Stochastic Maximum
    Likelihood (SML), also known as Persistent Contrastive Divergence (PCD)
    [2].

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components.

    Read more in the :ref:`User Guide <rbm>`.

    Parameters
    ----------
    n_components : int, optional
        Number of binary hidden units.

    learning_rate : float, optional
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, optional
        Number of examples per minibatch.

    n_iter : int, optional
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.

    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    intercept_hidden_ : array-like, shape (n_components,)
        Biases of the hidden units.

    intercept_visible_ : array-like, shape (n_features,)
        Biases of the visible units.

    components_ : array-like, shape (n_components, n_features)
        Weight matrix, where n_features in the number of
        visible units and n_components is the number of hidden units.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.neural_network import BernoulliRBM
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = BernoulliRBM(n_components=2)
    >>> model.fit(X)
    BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=2, n_iter=10,
           random_state=None, verbose=0)

    References
    ----------

    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    [2] Tieleman, T. Training Restricted Boltzmann Machines using
        Approximations to the Likelihood Gradient. International Conference
        on Machine Learning (ICML) 2008
    """
    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None,
                 unit_range=False, fit_intercept=True):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        self.unit_range = unit_range
        self.fit_intercept = fit_intercept

    def transform(self, X):
        """Compute the hidden layer activation probabilities, P(h=1|v=X).

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        h : array, shape (n_samples, n_components)
            Latent representations of the data.
        """
        check_is_fitted(self, "components_")

        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        return self._mean_hiddens(X)

    def _hidden_probabilites(self, v):
        """Computes the probabilities P(h=1|v).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        p = safe_sparse_dot(v, self.components_.T)
        p += self.intercept_hidden_
        if self.unit_range:
            p *= 2
        return expit(p, out=p)

    def _mean_hiddens(self, v):
        """Compute mean values of hidden variables, conditional on the visible ones"""
        if self.unit_range:
            return 2 * self._hidden_probabilites(v) - 1
        else:
            return self._hidden_probabilites(v)

    def _sample_hiddens(self, v, rng):
        """Sample from the distribution P(h|v).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to sample from.

        rng : RandomState
            Random number generator to use.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer.
        """
        p = self._hidden_probabilites(v)
        result = (rng.random_sample(size=p.shape) < p)
        if self.unit_range:
            # map 0,1 values to -1,+1:
            return 2*result - 1
        return result

    def _sample_visibles(self, h, rng):
        """Sample from the distribution P(v|h).

        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        rng : RandomState
            Random number generator to use.

        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        if self.unit_range:
            p *= 2
        expit(p, out=p)
        result = (rng.random_sample(size=p.shape) < p)

        if self.unit_range:
            # map 0,1 values to -1,+1:
            result = 2*result - 1
        return result

    def _free_energy(self, v):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.
        """
        temp = safe_sparse_dot(v, self.components_.T) + self.intercept_hidden_
        if self.unit_range:
            hidden_term = np.logaddexp(-temp, temp).sum(axis=1)
        else:
            hidden_term = np.logaddexp(0, temp).sum(axis=1)
        return (- safe_sparse_dot(v, self.intercept_visible_) - hidden_term)

    def gibbs(self, v):
        """Perform one Gibbs sampling step.

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to start from.

        Returns
        -------
        v_new : array-like, shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step.
        """
        check_is_fitted(self, "components_")
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        h_ = self._sample_hiddens(v, self.random_state_)
        v_ = self._sample_visibles(h_, self.random_state_)

        return v_

    def partial_fit(self, X, y=None):
        """Fit the model to the data X which should contain a partial
        segment of the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if not hasattr(self, 'random_state_'):
            self.random_state_ = check_random_state(self.random_state)
        if not hasattr(self, 'components_'):
            self.components_ = np.asarray(
                self.random_state_.normal(
                    0,
                    0.01,
                    (self.n_components, X.shape[1])
                ),
                order='F')
        if not hasattr(self, 'intercept_hidden_'):
            self.intercept_hidden_ = np.zeros(self.n_components, )
        if not hasattr(self, 'intercept_visible_'):
            self.intercept_visible_ = np.zeros(X.shape[1], )
        if not hasattr(self, 'h_samples_'):
            self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        self._fit(X, self.random_state_)

    def _fit(self, v_pos, rng):
        """Inner fit for one mini-batch.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        Parameters
        ----------
        v_pos : array-like, shape (n_samples, n_features)
            The data to use for training.

        rng : RandomState
            Random number generator to use for sampling.
        """
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_, rng)
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg)
        self.components_ += lr * update
        if self.fit_intercept:
            self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
            self.intercept_visible_ += lr * (np.asarray(
                                             v_pos.sum(axis=0)).squeeze() -
                                             v_neg.sum(axis=0))

        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def score_samples(self, X):
        """Compute the pseudo-likelihood of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).

        Returns
        -------
        pseudo_likelihood : array-like, shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).

        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        check_is_fitted(self, "components_")

        v = check_array(X, accept_sparse='csr')
        rng = check_random_state(self.random_state)

        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]),
               rng.randint(0, v.shape[1], v.shape[0]))
        if issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]

        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        return v.shape[1] * log_logistic(fe_ - fe)

    def fit(self, X, y=None):
        """Fit the model to the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        self.components_ = np.asarray(
            rng.normal(0, 0.01, (self.n_components, X.shape[1])),
            order='F')
        self.intercept_hidden_ = np.zeros(self.n_components, )
        self.intercept_visible_ = np.zeros(X.shape[1], )
        self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))
        verbose = self.verbose
        begin = time.time()
        for iteration in xrange(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                self._fit(X[batch_slice], rng)

            if verbose:
                end = time.time()
                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, iteration,
                         self.score_samples(X).mean(), end - begin))
                begin = end

        return self
