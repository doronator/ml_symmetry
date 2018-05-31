import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.NN_sans_bias import ConstrainedMLPClassifier
from src.utils import nudge_dataset, plot_sample, inversion_symmetric_features

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

# model1.fit(X_train, Y_train)
# model2.fit(X_train, Y_train)
#
#
# print("un-constrained NN model test results:\n{}\n".format(
#     metrics.classification_report(
#         Y_test,
#         model1.predict(X_test))))
#
# print("constrained NN model test results:\n{}\n".format(
#     metrics.classification_report(
#         Y_test,
#         model2.predict(X_test))))
#
# print("And now on the inverted set:")
#
#
# print("un-constrained NN model test results:\n{}\n".format(
#     metrics.classification_report(
#         Y_test,
#         model1.predict(-X_test))))
#
# print("constrained NN model test results:\n{}\n".format(
#     metrics.classification_report(
#         Y_test,
#         model2.predict(-X_test))))

# print("training on both the original data and the inverted data")
# X_train_2 = np.vstack((X_train, -X_train))
# Y_train_2 = np.hstack((Y_train,  Y_train))
#
# model1.fit(X_train_2, Y_train_2)
# model2.fit(X_train_2, Y_train_2)
#
# print("un-constrained NN model test results:\n{}\n".format(
#     metrics.classification_report(
#         Y_test,
#         model1.predict(X_test))))
#
# print("constrained NN model test results:\n{}\n".format(
#     metrics.classification_report(
#         Y_test,
#         model2.predict(X_test))))
#
# print("And now on the inverted set:")
#
#
# print("un-constrained NN model test results:\n{}\n".format(
#     metrics.classification_report(
#         Y_test,
#         model1.predict(-X_test))))
#
# print("constrained NN model test results:\n{}\n".format(
#     metrics.classification_report(
#         Y_test,
#         model2.predict(-X_test))))

# use symmetric features

model3 = ConstrainedMLPClassifier(solver='lbfgs',
                                  alpha=1e-5, activation='tanh',
                                  hidden_layer_sizes=(20, 10),
                                  random_state=1,
                                  fit_intercepts=False)

XX_train = inversion_symmetric_features(X_train)
XX_test = inversion_symmetric_features(X_test)

# print(XX_train.shape, X_train.shape)
# exit()

np.testing.assert_allclose(XX_test, inversion_symmetric_features(-X_test))

model3.fit(XX_train, Y_train)

print("constrained NN model test results:\n{}\n".format(
    metrics.classification_report(
        Y_test,
        model3.predict(XX_test))))

# idx = 0
# print(f"now plot the features for y={digits.target[idx]}")
# x = digits.data[idx, :]
# plot_sample(x)
# print(x.shape)
#
# _x = x.reshape(1, 64)
# _xx = -1 + 2*(_x - np.min(_x)) / (np.max(_x) - np.min(_x) + 0.0001)  # [-1,1] scaling
#
# xx = inversion_symmetric_features(_xx)
# plot_sample(xx)


idx = 7
print(f"now plot the features for y={Y_train[idx]}")
plot_sample(X_train[idx,:])
plot_sample(XX_train[idx,:])