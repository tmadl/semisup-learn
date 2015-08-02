import sklearn.svm
import numpy as np
import random

from frameworks.CPLELearning import CPLELearningModel
from methods import scikitTSVM
from examples.plotutils import evaluate_and_plot

kernel = "linear"

# number of data points
N = 60
supevised_data_points = 2
noise_probability = 0.1

# generate data-
cov = [[0.5, 0], [0, 0.5]]
Xs = np.random.multivariate_normal([0.5,0.5], cov, (N,))
ytrue = []
for i in range(N):
    if np.random.random() < noise_probability:
        ytrue.append(np.random.randint(2))
    else:
        ytrue.append(1 if np.sum(Xs[i])>1 else 0)
Xs = np.array(Xs)
ytrue = np.array(ytrue).astype(int)

ys = np.array([-1]*N)
sidx = random.sample(np.where(ytrue == 0)[0], supevised_data_points/2)+random.sample(np.where(ytrue == 1)[0], supevised_data_points/2)
ys[sidx] = ytrue[sidx]

Xsupervised = Xs[ys!=-1, :]
ysupervised = ys[ys!=-1]
    
# compare models     
lbl = "Purely supervised SVM:"
print lbl
model = sklearn.svm.SVC(kernel=kernel, probability=True)
model.fit(Xsupervised, ysupervised)
evaluate_and_plot(model, Xs, ys, ytrue, lbl, 1)

lbl =  "S3VM (Gieseke et al. 2012):"
print lbl
model = scikitTSVM.SKTSVM(kernel=kernel)
model.fit(Xs, ys.astype(int))
evaluate_and_plot(model, Xs, ys, ytrue, lbl, 2)

lbl = "CPLE(pessimistic) SVM:"
print lbl
model = CPLELearningModel(sklearn.svm.SVC(kernel=kernel, probability=True), predict_from_probabilities=True)
model.fit(Xs, ys.astype(int))
evaluate_and_plot(model, Xs, ys, ytrue, lbl, 3)

lbl = "CPLE(optimistic) SVM:"
print lbl
CPLELearningModel.pessimistic = False
model = CPLELearningModel(sklearn.svm.SVC(kernel=kernel, probability=True), predict_from_probabilities=True)
model.fit(Xs, ys.astype(int))
evaluate_and_plot(model, Xs, ys, ytrue, lbl, 4, block=True)