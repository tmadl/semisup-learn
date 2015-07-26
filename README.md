Semi-supervised learning frameworks for Python
===============

This project contains a number of basic Python implementations for semi-supervised
learning, made compatible with scikit-learn, including

- **Self learning** (self training), a naive semi-supervised learning framework applicable for any classifier

- **Contrastive Pessimistic Likelihood Estimation (CPLE)** (based on - but not equivalent to - [Loog, 2015](http://arxiv.org/abs/1503.00269)), a `safe' framework applicable for all classifiers which can yield prediction probabilities
(safe here means that the model trained on both labelled and unlabelled data should not be worse than models trained only on the labelled data)

- **Semi-Supervised Support Vector Machine (S3VM)** - a simple scikit-learn compatible wrapper for the QN-S3VM code developed by 
Fabian Gieseke, Antti Airola, Tapio Pahikkala, Oliver Kramer (see http://www.fabiangieseke.de/index.php/code/qns3vm ) 
This method was included for comparison

Examples
===============

Two-class classification examples with 56 unlabelled (small circles in the plot) and 4 labelled (large circles in the plot) data points. 
Plot titles show classification accuracies (percentage of data points correctly classified by the model)

In the second example, the state-of-the-art S3VM performs worse than the purely supervised SVM, while the CPLE SVM (by means of the 
pessimistic assumption) provides a slight increase in accuracy.

Quadratic Discriminant Analysis (from left to right: supervised QDA, pessimistic CPLE QDA, optimistic CPLE QDA) 
![Comparison of supervised QDA with CPLE QDA](qdaexample.png)

Support Vector Machine (from left to right: supervised SVM, S3VM [(Gieseke et al., 2012)](http://www.sciencedirect.com/science/article/pii/S0925231213003706), optimistic CPLE QDA)
![Comparison of supervised SVM, S3VM, and CPLE SVM](svmexample1.png)
 
Support Vector Machine (from left to right: supervised SVM, S3VM [(Gieseke et al., 2012)](http://www.sciencedirect.com/science/article/pii/S0925231213003706), optimistic CPLE QDA)
![Comparison of supervised SVM, S3VM, and CPLE SVM](svmexample2.png)

Motivation
===============

Current semi-supervised learning approaches require strong assumptions, and perform badly if those 
assumptions are violated (e.g. low density assumption, clustering assumption). Furthermore, the vast majority require O(N^2) memory.  

[(Loog, 2015)](http://arxiv.org/abs/1503.00269) has suggested an elegant framework (called Contrastive Pessimistic Likelihood Estimation / CPLE) which 
**only uses assumptions intrinsic to the chosen classifier**, and thus allows choosing likelihood-based classifiers which fit the domain / data 
distribution at hand, and can work even if some of the assumptions mentioned above are violated. The idea is to pessimistically assign soft labels 
to the unlabelled data, such that the improvement over the supervised version is minimal (i.e. assume the worst case for the unknown labels); 
and at the same time maximize log likelihood over labelled data. 

The parameters in CPLE can be estimated according to:
![CPLE Equation](eq1.png)

The original CPLE framework is only applicable to likelihood-based classifiers, and (Loog, 2015) only provides solutions for Linear Discriminant Analysis and the Nearest Mean Classifier.

The CPLE implementation in this project
===============

Building on this idea, this project contains a general semi-supervised learning framework allowing plugging in **any classifier** which allows 1) instance weighting and 2) can generate probability 
estimates (such probability estimates can also be provided by [Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling) for classifiers which don't support them. Also, an experimental feature 
is included to make the approach work with classifiers not supporting instance weighting).

In order to make the approach work with any classifier, the discriminative likelihood (DL) is used instead of the generative likelihood, which is the first major difference to (Loog, 2015). The second 
difference is that only the unlabelled data is included in the first term of the minimization objective below, which leads to pessimistic minimization of the DL over the unlabelled data, but maximization
of the DL over the labelled data. 

![CPLE Equation](alg1.png)

The resulting semi-supervised learning framework is highly computationally expensive, but has the following advantages:

- it is a generally applicable framework (works with most scikit-learn classifiers)

- it needs low memory (as opposed to e.g. Label Spreading which needs O(n^2)), and 

- it makes no additional assumptions except for the ones made by the choice of classifier 
