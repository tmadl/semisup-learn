import matplotlib.pyplot as plt
import numpy as np

cols = [np.array([1,0,0]),np.array([0,1,0])] # colors

def evaluate_and_plot(model, Xs, ys, ytrue, lbl, subplot = None, block=False):
    if subplot != None:
        plt.subplot(2,2,subplot)
    
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
        if hasattr(model, 'predict_from_probabilities') and model.predict_from_probabilities:
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
    
    plt.show(block=block)