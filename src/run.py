import numpy as np
from sklearn.neural_network import BernoulliRBM

from src.simple_constrained_rbm import ConstrainedBernoulliRBM

X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
model = BernoulliRBM(n_components=2)
model.fit(X)

print(model.transform(X))

print("and now the new RBM...")

model2 = ConstrainedBernoulliRBM(n_components=2, fit_intercepts=False)
model2.fit(X)

print(model2.transform(X))