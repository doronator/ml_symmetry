from nueral_network_sans_bias import BernoulliRBMSansBias
import numpy as np
from sklearn.neural_network import BernoulliRBM

X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
model = BernoulliRBM(n_components=2)
model.fit(X)

print(model.transform(X))

print("and now the new RBM...")

model2 = BernoulliRBMSansBias(n_components=2, use_intercepts=False)
model2.fit(X)

print(model2.transform(X))