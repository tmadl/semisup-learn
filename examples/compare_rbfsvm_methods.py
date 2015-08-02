import numpy as np
import random
import sklearn.svm

from frameworks.CPLELearning import CPLELearningModel
from methods import scikitTSVM
from examples.plotutils import evaluate_and_plot

kernel = "rbf"

# number of data points
N = 60
supevised_data_points = 4

# generate data
meandistance = 2

s = np.random.random()
cov = [[s, 0], [0, s]]
# some random Gaussians
gaussians = 6 #np.random.randint(4, 7)
Xs = np.random.multivariate_normal([np.random.random()*meandistance, np.random.random()*meandistance], cov, (N/gaussians,))
for i in range(gaussians-1):
    Xs = np.vstack(( Xs, np.random.multivariate_normal([np.random.random()*meandistance, np.random.random()*meandistance], cov, (N/gaussians,)) ))

# cut data into XOR
ytrue = ((Xs[:, 0] < np.mean(Xs[:, 0]))*(Xs[:, 1] < np.mean(Xs[:, 1])) + (Xs[:, 0] > np.mean(Xs[:, 0]))*(Xs[:, 1] > np.mean(Xs[:, 1])))*1

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
model.fit(Xs, ys)
evaluate_and_plot(model, Xs, ys, ytrue, lbl, 2)


lbl = "CPLE(pessimistic) SVM:"
print lbl
model = CPLELearningModel(sklearn.svm.SVC(kernel=kernel, probability=True), predict_from_probabilities=True)
model.fit(Xs, ys)
evaluate_and_plot(model, Xs, ys, ytrue, lbl, 3)


lbl = "CPLE(optimistic) SVM:"
print lbl
CPLELearningModel.pessimistic = False
model = CPLELearningModel(sklearn.svm.SVC(kernel=kernel, probability=True), predict_from_probabilities=True)
model.fit(Xs, ys)
evaluate_and_plot(model, Xs, ys, ytrue, lbl, 4, block=True)