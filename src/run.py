import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.NN_sans_bias import ConstrainedMLPClassifier
from src.utils import nudge_dataset, plot_sample

digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
X, Y = nudge_dataset(X, digits.target)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
X = 2*X - 1 # [-1,1] scaling

# plot_sample(X[0,:])
# plot_sample(-X[0,:])
# exit()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)

model1 = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        activation='tanh',
                        hidden_layer_sizes=(20, 10),
                        random_state=1)

model2 = ConstrainedMLPClassifier(solver='lbfgs',
                                  alpha=1e-5, activation='tanh',
                                  hidden_layer_sizes=(20, 10),
                                  random_state=1,
                                  fit_intercepts=False)

model1.fit(X_train, Y_train)
model2.fit(X_train, Y_train)


print("un-constrained NN model test results:\n{}\n".format(
    metrics.classification_report(
        Y_test,
        model1.predict(X_test))))

print("constrained NN model test results:\n{}\n".format(
    metrics.classification_report(
        Y_test,
        model2.predict(X_test))))

print("And now on the inverted set:")


print("un-constrained NN model test results:\n{}\n".format(
    metrics.classification_report(
        Y_test,
        model1.predict(-X_test))))

print("constrained NN model test results:\n{}\n".format(
    metrics.classification_report(
        Y_test,
        model2.predict(-X_test))))

