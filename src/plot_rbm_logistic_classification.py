"""
==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================

For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

In order to learn good latent representations from a small dataset, we
artificially generate more labeled data by perturbing the training data with
linear shifts of 1 pixel in each direction.

This example shows how to build a classification pipeline with a BernoulliRBM
feature extractor and a :class:`LogisticRegression
<sklearn.linear_model.LogisticRegression>` classifier. The hyperparameters
of the entire model (learning rate, hidden layer size, regularization)
were optimized by grid search, but the search is not reproduced here because
of runtime constraints.

Logistic regression on raw pixel values is presented for comparison. The
example shows that the features extracted by the BernoulliRBM help improve the
classification accuracy.
"""
# from src.nueral_network_sans_bias import BernoulliRBMSansBias
from src.simple_constrained_rbm import ConstrainedBernoulliRBM
from src.utils import nudge_dataset

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline


# #############################################################################
# Setting up

# Load Data
digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
X, Y = nudge_dataset(X, digits.target)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
# X = 2*X - 1 # [-1,1] scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression(fit_intercept=False)
# rbm = BernoulliRBM(random_state=0, verbose=True)
rbm = ConstrainedBernoulliRBM(random_state=0, verbose=True, fit_intercepts=False)

# classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
pipe = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

# #############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000.0

param_grid = {
    "rbm__fit_intercepts": [False],
    "rbm__learning_rate": [0.01, 0.03, 0.06, 0.09, 0.2],
    "rbm__n_iter": [20],
    "rbm__n_components" : [100],
    "logistic__fit_intercept": [False],
    # "logistic__C": [6000],
    "logistic__C": [1000, 3000, 6000, 9000],
}

classifier = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid)

# Training RBM-Logistic Pipeline
classifier.fit(X_train, Y_train)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)

# #############################################################################
# Evaluation

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))

print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))

# #############################################################################
# Plotting

print("best params:")
print(classifier.best_params_)
exit()

temp = classifier.best_estimator_.named_steps['rbm'].components_

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(temp):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
