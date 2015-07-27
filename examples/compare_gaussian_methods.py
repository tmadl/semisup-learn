import time
import numpy as np
import matplotlib.pyplot as plt


from frameworks.CPLELearning import CPLELearningModel
from frameworks.SelfLearning import SelfLearningModel
from methods.scikitWQDA import WQDA
from sklearn.qda import QDA

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

plt.figure()
cols = [np.array([1,0,0]),np.array([0,1,0])] # colors

# loop through and compare methods     
for i in range(4):
    plt.subplot(2,2,i+1)
    
    t1=time.time()
    # train model
    if i == 0:
        lbl= "Purely supervised QDA:"
        model = WQDA()
        model.fit(Xsupervised, ysupervised)
    else:
        if i == 1:
            lbl= "SelfLearning QDA:"
            model = SelfLearningModel(WQDA())
        if i == 2:
            lbl= "CPLE(pessimistic) QDA:"
            model = CPLELearningModel(WQDA())
        elif i == 3:
            lbl= "CPLE(optimistic) QDA:"
            CPLELearningModel.pessimistic = False
            model = CPLELearningModel(WQDA())
        model.fit(Xs, ys)
    print ""
    print lbl
    print "Model training time: ", round(time.time()-t1, 3)

    # predict, and evaluate
    pred = model.predict(Xs)
    
    acc = np.mean(pred==ytrue)
    print "accuracy:", round(acc, 3)
    
    # plot probabilities
    [minx, maxx] = [np.min(Xs[:, 0]), np.max(Xs[:, 0])]
    [miny, maxy] = [np.min(Xs[:, 1]), np.max(Xs[:, 1])]
    gridsize = 100
    xx = np.linspace(minx, maxx, gridsize)
    yy = np.linspace(miny, maxy, gridsize).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    probas = model.predict_proba(Xfull)
    plt.imshow(probas[:, 1].reshape((gridsize, gridsize)), extent=(minx, maxx, miny, maxy), origin='lower')
    
    # plot decision boundary
    try:
        if i > 1:
            plt.contour((probas[:, 0]<np.average(probas[:, 0])).reshape((gridsize, gridsize)), extent=(minx, maxx, miny, maxy), origin='lower')
        else:
            plt.contour(model.predict(Xfull).reshape((gridsize, gridsize)), extent=(minx, maxx, miny, maxy), origin='lower')
    except:
        print "contour failed"
    
    # plot data points
    P = np.max(model.predict_proba(Xs), axis=1)
    plt.scatter(Xs[:, 0], Xs[:,1], c=ytrue, s=(ys>-1)*300+100, linewidth=1, edgecolor=[cols[p]*P[p] for p in model.predict(Xs).astype(int)], cmap='hot')
    plt.scatter(Xs[ys>-1, 0], Xs[ys>-1,1], c=ytrue[ys>-1], s=300, linewidth=1, edgecolor=[cols[p]*P[p] for p in model.predict(Xs).astype(int)], cmap='hot')
    plt.title(lbl + str(round(acc, 2)))
    
plt.show(block=True)
