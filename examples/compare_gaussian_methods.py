import numpy as np

from frameworks.CPLELearning import CPLELearningModel
from frameworks.SelfLearning import SelfLearningModel
from methods.scikitWQDA import WQDA
from examples.plotutils import evaluate_and_plot

# number of data points
N = 60
supevised_data_points = 4

# generate data
meandistance = 1

s = np.random.random()
cov = [[s, 0], [0, s]]
Xs = np.random.multivariate_normal([-s*meandistance, -s*meandistance], cov, (N,))
Xs = np.vstack(( Xs, np.random.multivariate_normal([s*meandistance, s*meandistance], cov, (N,)) ))
ytrue = np.array([0]*N + [1]*N)

ys = np.array([-1]*(2*N))
for i in range(supevised_data_points/2):
    ys[np.random.randint(0, N)] = 0
for i in range(supevised_data_points/2):
    ys[np.random.randint(N, 2*N)] = 1
    
Xsupervised = Xs[ys!=-1, :]
ysupervised = ys[ys!=-1]

# compare models

lbl = "Purely supervised QDA:"
print lbl
model = WQDA()
model.fit(Xsupervised, ysupervised)
evaluate_and_plot(model, Xs, ys, ytrue, lbl, 1)

lbl = "SelfLearning QDA:"
print lbl
model = SelfLearningModel(WQDA())
model.fit(Xs, ys)
evaluate_and_plot(model, Xs, ys, ytrue, lbl, 2)

lbl = "CPLE(pessimistic) QDA:"
print lbl
model = CPLELearningModel(WQDA(), predict_from_probabilities=True)
model.fit(Xs, ys)
evaluate_and_plot(model, Xs, ys, ytrue, lbl, 3)

lbl = "CPLE(optimistic) QDA:"
print lbl
CPLELearningModel.pessimistic = False
model = CPLELearningModel(WQDA(), predict_from_probabilities=True)
model.fit(Xs, ys)
evaluate_and_plot(model, Xs, ys, ytrue, lbl, 4, block=True)
