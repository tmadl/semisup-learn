############################################################################################
# QN-S3VM BFGS optimizer for semi-supervised support vector machines. 
#
# This implementation provides both a L-BFGS optimization scheme
# for semi-supvised support vector machines. Details can be found in:
#
#   F. Gieseke, A. Airola, T. Pahikkala, O. Kramer, Sparse quasi-
#   Newton optimization for semi-supervised support vector ma-
#   chines, in: Proc. of the 1st Int. Conf. on Pattern Recognition
#   Applications and Methods, 2012, pp. 45-54.
#
# Version: 0.1 (September, 2012)
#
# Bugs: Please send any bugs to "f DOT gieseke AT uni-oldenburg.de"
#
#
# Copyright (C) 2012  Fabian Gieseke, Antti Airola, Tapio Pahikkala, Oliver Kramer
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# 
# INSTALLATION and DEPENDENCIES
#
# The module should work out of the box, given Python and Numpy (http://numpy.scipy.org/)
# and Scipy (http://scipy.org/) installed correctly. 
# 
# We have tested the code on Ubuntu 12.04 (32 Bit) with Python 2.7.3, Numpy 1.6.1, 
# and Scipy 0.9.0. Installing these packages on a Ubuntu- or Debian-based systems 
# can be done via "sudo apt-get install python python-numpy python-scipy".
#
#
# RUNNING THE EXAMPLES
# 
# For a description of the data sets, see the paper mentioned above and the references 
# therein. Running the command "python qns3vm.py" should yield an output similar to:
# 
# Sparse text data set instance
# Number of labeled patterns:  48
# Number of unlabeled patterns:  924
# Number of test patterns:  974
# Time needed to compute the model:  0.775886058807  seconds
# Classification error of QN-S3VM:  0.0667351129363
#
# Dense gaussian data set instance
# Number of labeled patterns:  25
# Number of unlabeled patterns:  225
# Number of test patterns:  250
# Time needed to compute the model:  0.464584112167  seconds
# Classification error of QN-S3VM:  0.012
#
# Dense moons data set instance
# Number of labeled patterns:  5
# Number of unlabeled patterns:  495
# Number of test patterns:  500
# Time needed to compute the model:  0.69714307785  seconds
# Classification error of QN-S3VM:  0.0

############################################################################################

import array as arr
import math
import copy as cp
import logging
import numpy as np
from numpy import *
import operator
from time import time
import sys
from scipy import optimize
import scipy.sparse.csc as csc
from scipy import sparse
import scipy
import warnings
warnings.simplefilter('error')

__author__ =  'Fabian Gieseke, Antti Airola, Tapio Pahikkala, Oliver Kramer'
__version__=  '0.1'

class QN_S3VM:
    """
    L-BFGS optimizer for semi-supervised support vector machines (S3VM).
    """
    def __init__(self, X_l, L_l, X_u, random_generator = None, ** kw):
        """
        Initializes the model. Detects automatically if dense or sparse data is provided.

        Keyword arguments:
        X_l -- patterns of labeled part of the data
        L_l -- labels of labeled part of the data
        X_u -- patterns of unlabeled part of the data
        random_generator -- particular instance of a random_generator (default None)
        kw -- additional parameters for the optimizer
        lam -- regularization parameter lambda (default 1, must be a float > 0)
        lamU -- cost parameter that determines influence of unlabeled patterns (default 1, must be float > 0)
        sigma -- kernel width for RBF kernel (default 1.0, must be a float > 0)
        kernel_type -- "Linear" or "RBF" (default "Linear")
        numR -- implementation of subset of regressors. If None is provided, all patterns are used
                (no approximation). Must fulfill 0 <= numR <= len(X_l) + len(X_u) (default None)
        estimate_r -- desired ratio for positive and negative assigments for 
                      unlabeled patterns (-1.0 <= estimate_r <= 1.0). If estimate_r=None, 
                      then L_l is used to estimate this ratio (in case len(L_l) >= 
                      minimum_labeled_patterns_for_estimate_r. Otherwise use estimate_r = 0.0
                      (default None)
        minimum_labeled_patterns_for_estimate_r -- see above (default 0)
        BFGS_m -- BFGS parameter (default 50)
        BFGS_maxfun -- BFGS parameter, maximum number of function calls (default 500)
        BFGS_factr -- BFGS parameter (default 1E12)
        BFGS_pgtol -- BFGS parameter (default 1.0000000000000001e-05)
        """
        self.__model = None
        # Initiate model for sparse data
        if isinstance(X_l, csc.csc_matrix):
            self.__data_type = "sparse"
            self.__model = QN_S3VM_Sparse(X_l, L_l, X_u, random_generator, ** kw)
        # Initiate model for dense data
        elif (isinstance(X_l[0], list)) or (isinstance(X_l[0], np.ndarray)):
            self.__data_type = "dense"
            self.__model = QN_S3VM_Dense(X_l, L_l, X_u, random_generator, ** kw)
        # Data format unknown
        if self.__model == None:
            logging.info("Data format for patterns is unknown.")
            sys.exit(0)

    def train(self):
        """
        Training phase.

        Returns:
        The computed partition for the unlabeled patterns.
        """
        return self.__model.train()

    def getPredictions(self, X, real_valued=False):
        """
        Computes the predicted labels for a given set of patterns

        Keyword arguments:
        X -- The set of patterns 
        real_valued -- If True, then the real prediction values are returned

        Returns:
        The predictions for the list X of patterns.
        """
        return self.__model.getPredictions(X, real_valued=False)

    def predict(self, x):
        """
        Predicts a label (-1 or +1) for the pattern

        Keyword arguments:
        x -- The pattern 

        Returns:
        The prediction for x.
        """
        return self.__model.predict(x)

    def predictValue(self, x):
        """
        Computes f(x) for a given pattern (see Representer Theorem)
    
        Keyword arguments:
        x -- The pattern 

        Returns:
        The (real) prediction value for x.
        """
        return self.__model.predictValue(x)

    def getNeededFunctionCalls(self):
        """
        Returns the number of function calls needed during 
        the optimization process.
        """
        return self.__model.getNeededFunctionCalls()

    def mygetPreds(self, X, real_valued=False):
        return self.__model.mygetPreds(X, real_valued)

############################################################################################
############################################################################################
class QN_S3VM_Dense:

    """
    BFGS optimizer for semi-supervised support vector machines (S3VM).

    Dense Data
    """
    parameters = {
    'lam': 1,
    'lamU':1,
    'sigma': 1,
    'kernel_type': "Linear",
    'numR':None,
    'estimate_r':None,
    'minimum_labeled_patterns_for_estimate_r':0,
    'BFGS_m':50,
    'BFGS_maxfun':500,
    'BFGS_factr':1E12,
    'BFGS_pgtol':1.0000000000000001e-05,
    'BFGS_verbose':-1,
    'surrogate_s':3.0,
    'surrogate_gamma':20.0,
    'breakpoint_for_exp':500
    }

    def __init__(self, X_l, L_l, X_u, random_generator, ** kw):
        """
        Intializes the S3VM optimizer.
        """
        self.__random_generator = random_generator
        self.__X_l, self.__X_u, self.__L_l = X_l, X_u, L_l
        assert len(X_l) == len(L_l)
        self.__X = cp.deepcopy(self.__X_l)
        self.__X.extend(cp.deepcopy(self.__X_u))
        self.__size_l, self.__size_u, self.__size_n = len(X_l), len(X_u), len(X_l) + len(X_u)
        self.__matrices_initialized = False
        self.__setParameters( ** kw)
        self.__kw = kw

    def train(self):
        """
        Training phase.

        Returns:
        The computed partition for the unlabeled patterns.
        """
        indi_opt = self.__optimize()
        self.__recomputeModel(indi_opt)
        predictions = self.__getTrainingPredictions(self.__X)
        return predictions

    def mygetPreds(self, X, real_valued=False):
        KNR = self.__kernel.computeKernelMatrix(X, self.__Xreg)
        KNU_bar = self.__kernel.computeKernelMatrix(X, self.__X_u_subset, symmetric=False)
        KNU_bar_horizontal_sum = (1.0 / len(self.__X_u_subset)) * KNU_bar.sum(axis=1)
        KNR = KNR - KNU_bar_horizontal_sum - self.__KU_barR_vertical_sum + self.__KU_barU_bar_sum
        preds = KNR * self.__c[0:self.__dim-1,:] + self.__c[self.__dim-1,:]
        return preds

    def getPredictions(self, X, real_valued=False):
        """
        Computes the predicted labels for a given set of patterns

        Keyword arguments:
        X -- The set of patterns 
        real_valued -- If True, then the real prediction values are returned

        Returns:
        The predictions for the list X of patterns.
        """
        KNR = self.__kernel.computeKernelMatrix(X, self.__Xreg)
        KNU_bar = self.__kernel.computeKernelMatrix(X, self.__X_u_subset, symmetric=False)
        KNU_bar_horizontal_sum = (1.0 / len(self.__X_u_subset)) * KNU_bar.sum(axis=1)
        KNR = KNR - KNU_bar_horizontal_sum - self.__KU_barR_vertical_sum + self.__KU_barU_bar_sum
        preds = KNR * self.__c[0:self.__dim-1,:] + self.__c[self.__dim-1,:]
        if real_valued == True:
            return preds.flatten(1).tolist()[0]
        else:
            return np.sign(np.sign(preds)+0.1).flatten(1).tolist()[0]
    
    def predict(self, x):
        """
        Predicts a label for the pattern

        Keyword arguments:
        x -- The pattern 

        Returns:
        The prediction for x.
        """
        return self.getPredictions([x], real_valued=False)[0]
        
    def predictValue(self, x):
        """
        Computes f(x) for a given pattern (see Representer Theorem)
    
        Keyword arguments:
        x -- The pattern 

        Returns:
        The (real) prediction value for x.
        """
        return self.getPredictions([x], real_valued=True)[0]

    def getNeededFunctionCalls(self):
        """
        Returns the number of function calls needed during 
        the optimization process.
        """
        return self.__needed_function_calls

    def __setParameters(self,  ** kw):
        for attr, val in kw.items():
            self.parameters[attr] = val
        self.__lam = float(self.parameters['lam'])
        assert self.__lam > 0
        self.__lamU = float(self.parameters['lamU'])
        assert self.__lamU > 0
        self.__lam_Uvec = [float(self.__lamU)*i for i in [0,0.000001,0.0001,0.01,0.1,0.5,1]]
        self.__sigma = float(self.parameters['sigma'])
        assert self.__sigma > 0
        self.__kernel_type = str(self.parameters['kernel_type'])
        if self.parameters['numR'] != None:
            self.__numR = int(self.parameters['numR'])
            assert (self.__numR <= len(self.__X)) and (self.__numR > 0)
        else:
            self.__numR = len(self.__X)
        self.__regressors_indices = sorted(self.__random_generator.sample( range(0,len(self.__X)), self.__numR ))
        self.__dim = self.__numR + 1 # add bias term b
        self.__minimum_labeled_patterns_for_estimate_r = float(self.parameters['minimum_labeled_patterns_for_estimate_r'])
        # If reliable estimate is available or can be estimated, use it, otherwise
        # assume classes to be balanced (i.e., estimate_r=0.0)
        if self.parameters['estimate_r'] != None:
            self.__estimate_r = float(self.parameters['estimate_r'])
        elif len(self.__L_l) >= self.__minimum_labeled_patterns_for_estimate_r:
            self.__estimate_r = (1.0 / len(self.__L_l)) * np.sum(self.__L_l)
        else:
            self.__estimate_r = 0.0
        self.__BFGS_m = int(self.parameters['BFGS_m'])
        self.__BFGS_maxfun = int(self.parameters['BFGS_maxfun'])
        self.__BFGS_factr = float(self.parameters['BFGS_factr'])
        # This is a hack for 64 bit systems (Linux). The machine precision 
        # is different for the BFGS optimizer (Fortran code) and we fix this by:
        is_64bits = sys.maxsize > 2**32
        if is_64bits:
            logging.debug("64-bit system detected, modifying BFGS_factr!")
            self.__BFGS_factr = 0.000488288*self.__BFGS_factr
        self.__BFGS_pgtol = float(self.parameters['BFGS_pgtol'])
        self.__BFGS_verbose = int(self.parameters['BFGS_verbose'])
        self.__surrogate_gamma = float(self.parameters['surrogate_gamma'])
        self.__s = float(self.parameters['surrogate_s'])
        self.__breakpoint_for_exp = float(self.parameters['breakpoint_for_exp'])
        self.__b = self.__estimate_r
        # size of unlabeled patterns to estimate mean (used for balancing constraint)
        self.__max_unlabeled_subset_size = 1000


    def __optimize(self):
        logging.debug("Starting optimization with BFGS ...")
        self.__needed_function_calls = 0
        self.__initializeMatrices()
        # starting point
        c_current = zeros(self.__dim, float64)
        c_current[self.__dim-1] = self.__b
        # Annealing sequence.
        for i in xrange(len(self.__lam_Uvec)):
            self.__lamU = self.__lam_Uvec[i]
            # crop one dimension (in case the offset b is fixed)
            c_current = c_current[:self.__dim-1]
            c_current = self.__localSearch(c_current)
            # reappend it if needed
            c_current = np.append(c_current, self.__b)
        f_opt = self.__getFitness(c_current)
        return c_current, f_opt

    def __localSearch(self, start):
        c_opt, f_opt, d = optimize.fmin_l_bfgs_b(self.__getFitness, start, m=self.__BFGS_m, \
                            fprime=self.__getFitness_Prime, maxfun=self.__BFGS_maxfun, factr=self.__BFGS_factr,\
                            pgtol=self.__BFGS_pgtol, iprint=self.__BFGS_verbose)
        self.__needed_function_calls += int(d['funcalls'])
        return c_opt

    def __initializeMatrices(self):
        if self.__matrices_initialized == False:
            logging.debug("Initializing matrices...")
            # Initialize labels
            x = arr.array('i')
            for l in self.__L_l:
                x.append(l)
            self.__YL = mat(x, dtype=np.float64)
            self.__YL = self.__YL.transpose()
            # Initialize kernel matrices
            if (self.__kernel_type == "Linear"):
                self.__kernel = LinearKernel()
            elif (self.__kernel_type == "RBF"):
                self.__kernel = RBFKernel(self.__sigma)
            self.__Xreg = (mat(self.__X)[self.__regressors_indices,:].tolist())
            self.__KLR = self.__kernel.computeKernelMatrix(self.__X_l,self.__Xreg, symmetric=False)
            self.__KUR = self.__kernel.computeKernelMatrix(self.__X_u,self.__Xreg, symmetric=False)
            self.__KNR = cp.deepcopy(bmat([[self.__KLR], [self.__KUR]]))
            self.__KRR = self.__KNR[self.__regressors_indices,:]
            # Center patterns in feature space (with respect to approximated mean of unlabeled patterns in the feature space)
            subset_unlabled_indices = sorted(self.__random_generator.sample( range(0,len(self.__X_u)), min(self.__max_unlabeled_subset_size, len(self.__X_u)) ))
            self.__X_u_subset = (mat(self.__X_u)[subset_unlabled_indices,:].tolist())
            self.__KNU_bar = self.__kernel.computeKernelMatrix(self.__X, self.__X_u_subset, symmetric=False)
            self.__KNU_bar_horizontal_sum = (1.0 / len(self.__X_u_subset)) * self.__KNU_bar.sum(axis=1)
            self.__KU_barR = self.__kernel.computeKernelMatrix(self.__X_u_subset, self.__Xreg, symmetric=False)
            self.__KU_barR_vertical_sum = (1.0 / len(self.__X_u_subset)) * self.__KU_barR.sum(axis=0)
            self.__KU_barU_bar = self.__kernel.computeKernelMatrix(self.__X_u_subset, self.__X_u_subset, symmetric=False)
            self.__KU_barU_bar_sum = (1.0 / (len(self.__X_u_subset)))**2 * self.__KU_barU_bar.sum()
            self.__KNR = self.__KNR - self.__KNU_bar_horizontal_sum - self.__KU_barR_vertical_sum + self.__KU_barU_bar_sum
            self.__KRR = self.__KNR[self.__regressors_indices,:]
            self.__KLR = self.__KNR[range(0,len(self.__X_l)),:]
            self.__KUR = self.__KNR[range(len(self.__X_l),len(self.__X)),:]
            self.__matrices_initialized = True

    def __getFitness(self,c):
        # Check whether the function is called from the bfgs solver 
        # (that does not optimize the offset b) or not
        if len(c) == self.__dim - 1:
            c = np.append(c, self.__b)
        c = mat(c)
        b = c[:,self.__dim-1].T
        c_new = c[:,0:self.__dim-1].T
        preds_labeled = self.__surrogate_gamma*(1.0 - multiply(self.__YL, self.__KLR * c_new + b))
        preds_unlabeled = self.__KUR * c_new + b
        # This vector has a "one" for each "numerically instable" entry; "zeros" for "good ones". 
        preds_labeled_conflict_indicator = np.sign(np.sign(preds_labeled/self.__breakpoint_for_exp - 1.0) + 1.0)
        # This vector has a one for each good entry and zero otherwise
        preds_labeled_good_indicator = (-1)*(preds_labeled_conflict_indicator - 1.0)
        preds_labeled_for_conflicts = multiply(preds_labeled_conflict_indicator,preds_labeled) 
        preds_labeled = multiply(preds_labeled,preds_labeled_good_indicator)
        # Compute values for good entries
        preds_labeled_log_exp = np.log(1.0 + np.exp(preds_labeled))
        # Compute values for instable entries
        preds_labeled_log_exp = multiply(preds_labeled_good_indicator, preds_labeled_log_exp)
        # Replace critical values with values 
        preds_labeled_final = preds_labeled_log_exp + preds_labeled_for_conflicts
        term1 = (1.0/(self.__surrogate_gamma*self.__size_l)) * np.sum(preds_labeled_final)
        preds_unlabeled_squared = multiply(preds_unlabeled,preds_unlabeled)
        term2 = (float(self.__lamU)/float(self.__size_u))*np.sum(np.exp(-self.__s * preds_unlabeled_squared))
        term3 = self.__lam * (c_new.T * self.__KRR * c_new)
        return (term1 + term2 + term3)[0,0]

    def __getFitness_Prime(self,c):
        # Check whether the function is called from the bfgs solver 
        # (that does not optimize the offset b) or not
        if len(c) == self.__dim - 1:
            c = np.append(c, self.__b)
        c = mat(c)
        b = c[:,self.__dim-1].T
        c_new = c[:,0:self.__dim-1].T
        preds_labeled = self.__surrogate_gamma * (1.0 - multiply(self.__YL, self.__KLR * c_new + b))
        preds_unlabeled = (self.__KUR * c_new + b)
        # This vector has a "one" for each "numerically instable" entry; "zeros" for "good ones". 
        preds_labeled_conflict_indicator = np.sign(np.sign(preds_labeled/self.__breakpoint_for_exp - 1.0) + 1.0)
        # This vector has a one for each good entry and zero otherwise
        preds_labeled_good_indicator = (-1)*(preds_labeled_conflict_indicator - 1.0)
        preds_labeled = multiply(preds_labeled,preds_labeled_good_indicator)
        preds_labeled_exp = np.exp(preds_labeled)
        term1 = multiply(preds_labeled_exp, 1.0/(1.0 + preds_labeled_exp))
        term1 = multiply(preds_labeled_good_indicator, term1)
        # Replace critical values with "1.0"
        term1 = term1 + preds_labeled_conflict_indicator
        term1 = multiply(self.__YL, term1)
        preds_unlabeled_squared_exp_f = multiply(preds_unlabeled,preds_unlabeled)
        preds_unlabeled_squared_exp_f = np.exp(-self.__s * preds_unlabeled_squared_exp_f)
        preds_unlabeled_squared_exp_f = multiply(preds_unlabeled_squared_exp_f, preds_unlabeled)
        term1 = (-1.0/self.__size_l) * (term1.T * self.__KLR).T
        term2 = ((-2.0 * self.__s * self.__lamU)/float(self.__size_u)) * (preds_unlabeled_squared_exp_f.T * self.__KUR).T
        term3 = 2*self.__lam*(self.__KRR * c_new)
        return array((term1 + term2 + term3).T)[0]

    def __recomputeModel(self, indi):
        self.__c = mat(indi[0]).T

    def __getTrainingPredictions(self, X, real_valued=False):
        preds = self.__KNR * self.__c[0:self.__dim-1,:] + self.__c[self.__dim-1,:]
        if real_valued == True:
            return preds.flatten(1).tolist()[0]
        else:
            return np.sign(np.sign(preds)+0.1).flatten(1).tolist()[0]

    def __check_matrix(self, M):
        smallesteval = scipy.linalg.eigvalsh(M, eigvals=(0,0))[0]
        if smallesteval < 0.0:
            shift = abs(smallesteval) + 0.0000001 
            M = M + shift
        return M

############################################################################################
############################################################################################
class QN_S3VM_Sparse:
    """
    BFGS optimizer for semi-supervised support vector machines (S3VM).

    Sparse Data
    """
    parameters = {
    'lam': 1,
    'lamU':1,
    'estimate_r':None,
    'minimum_labeled_patterns_for_estimate_r':0,
    'BFGS_m':50,
    'BFGS_maxfun':500,
    'BFGS_factr':1E12,
    'BFGS_pgtol':1.0000000000000001e-05,
    'BFGS_verbose':-1,
    'surrogate_s':3.0,
    'surrogate_gamma':20.0,
    'breakpoint_for_exp':500
    }


    def __init__(self, X_l, L_l, X_u, random_generator, ** kw):
        """
        Intializes the S3VM optimizer.
        """
        self.__random_generator = random_generator
        # This is a nuisance, but we may need to pad extra dimensions to either X_l or X_u
        # in case the highest feature indices appear only in one of the two data matrices
        if X_l.shape[1] > X_u.shape[1]:
            X_u = sparse.hstack([X_u, sparse.coo_matrix(X_u.shape[0], X_l.shape[1] - X_u.shape[1])])
        elif X_l.shape[1] < X_u.shape[1]:
            X_l = sparse.hstack([X_l, sparse.coo_matrix(X_l.shape[0], X_u.shape[1] - X_u.shape[1])])
        # We vertically stack the data matrices into one big matrix
        X = sparse.vstack([X_l, X_u])
        self.__size_l, self.__size_u, self.__size_n = X_l.shape[0], X_u.shape[0], X_l.shape[0]+ X_u.shape[0]
        x = arr.array('i')
        for l in L_l:
            x.append(int(l))
        self.__YL = mat(x, dtype=np.float64)
        self.__YL = self.__YL.transpose()
        self.__setParameters( ** kw)
        self.__kw = kw
        self.X_l = X_l.tocsr()
        self.X_u = X_u.tocsr()
        self.X = X.tocsr()
        # compute mean of unlabeled patterns
        self.__mean_u = self.X_u.mean(axis=0)
        self.X_u_T = X_u.tocsc().T
        self.X_l_T = X_l.tocsc().T
        self.X_T = X.tocsc().T

    def train(self):
        """
        Training phase.

        Returns:
        The computed partition for the unlabeled patterns.
        """
        indi_opt = self.__optimize()
        self.__recomputeModel(indi_opt)
        predictions = self.getPredictions(self.X)
        return predictions

    def getPredictions(self, X, real_valued=False):
        """
        Computes the predicted labels for a given set of patterns

        Keyword arguments:
        X -- The set of patterns 
        real_valued -- If True, then the real prediction values are returned

        Returns:
        The predictions for the list X of patterns.
        """
        c_new = self.__c[:self.__dim-1]
        W = self.X.T*c_new - self.__mean_u.T*np.sum(c_new)
        # Again, possibility of dimension mismatch due to use of sparse matrices
        if X.shape[1] > W.shape[0]:
            X = X[:,range(W.shape[0])]
        if X.shape[1] < W.shape[0]:
            W = W[range(X.shape[1])]
        X = X.tocsc()
        preds = X * W + self.__b
        if real_valued == True:
            return preds.flatten(1).tolist()[0]
        else:
            return np.sign(np.sign(preds)+0.1).flatten(1).tolist()[0]

    def predict(self, x):
        """
        Predicts a label for the pattern

        Keyword arguments:
        x -- The pattern 

        Returns:
        The prediction for x.
        """
        return self.getPredictions([x], real_valued=False)[0]
        
    def predictValue(self, x):
        """
        Computes f(x) for a given pattern (see Representer Theorem)
    
        Keyword arguments:
        x -- The pattern 

        Returns:
        The (real) prediction value for x.
        """
        return self.getPredictions([x], real_valued=True)[0]

    def getNeededFunctionCalls(self):
        """
        Returns the number of function calls needed during 
        the optimization process.
        """
        return self.__needed_function_calls

    def __setParameters(self,  ** kw):
        for attr, val in kw.items():
            self.parameters[attr] = val
        self.__lam = float(self.parameters['lam'])
        assert self.__lam > 0
        self.__lamU = float(self.parameters['lamU'])
        assert self.__lamU > 0
        self.__lam_Uvec = [float(self.__lamU)*i for i in [0,0.000001,0.0001,0.01,0.1,0.5,1]]
        self.__minimum_labeled_patterns_for_estimate_r = float(self.parameters['minimum_labeled_patterns_for_estimate_r'])
        # If reliable estimate is available or can be estimated, use it, otherwise
        # assume classes to be balanced (i.e., estimate_r=0.0)
        if self.parameters['estimate_r'] != None:
            self.__estimate_r = float(self.parameters['estimate_r'])
        elif self.__YL.shape[0] > self.__minimum_labeled_patterns_for_estimate_r:
            self.__estimate_r = (1.0 / self.__YL.shape[0]) * np.sum(self.__YL[0:])
        else:
            self.__estimate_r = 0.0
        self.__dim = self.__size_n + 1 # for offset term b
        self.__BFGS_m = int(self.parameters['BFGS_m'])
        self.__BFGS_maxfun = int(self.parameters['BFGS_maxfun'])
        self.__BFGS_factr = float(self.parameters['BFGS_factr'])
        # This is a hack for 64 bit systems (Linux). The machine precision 
        # is different for the BFGS optimizer (Fortran code) and we fix this by:
        is_64bits = sys.maxsize > 2**32
        if is_64bits:
            logging.debug("64-bit system detected, modifying BFGS_factr!")
            self.__BFGS_factr = 0.000488288*self.__BFGS_factr
        self.__BFGS_pgtol = float(self.parameters['BFGS_pgtol'])
        self.__BFGS_verbose = int(self.parameters['BFGS_verbose'])
        self.__surrogate_gamma = float(self.parameters['surrogate_gamma'])
        self.__s = float(self.parameters['surrogate_s'])
        self.__breakpoint_for_exp = float(self.parameters['breakpoint_for_exp'])
        self.__b = self.__estimate_r

    def __optimize(self):
        logging.debug("Starting optimization with BFGS ...")
        self.__needed_function_calls = 0
        # starting_point
        c_current = zeros(self.__dim, float64)
        c_current[self.__dim-1] = self.__b
        # Annealing sequence.
        for i in xrange(len(self.__lam_Uvec)):
            self.__lamU = self.__lam_Uvec[i]
            # crop one dimension (in case the offset b is fixed)
            c_current = c_current[:self.__dim-1]
            c_current = self.__localSearch(c_current)
            # reappend it if needed
            c_current = np.append(c_current, self.__b)
        f_opt = self.__getFitness(c_current)
        return c_current, f_opt

    def __localSearch(self, start):
        c_opt, f_opt, d = optimize.fmin_l_bfgs_b(self.__getFitness, start, m=self.__BFGS_m, \
                                     fprime=self.__getFitness_Prime, maxfun=self.__BFGS_maxfun,\
                                     factr=self.__BFGS_factr, pgtol=self.__BFGS_pgtol, iprint=self.__BFGS_verbose)
        self.__needed_function_calls += int(d['funcalls'])
        return c_opt

    def __getFitness(self,c):
        # check whether the function is called from the bfgs solver 
        # (that does not optimize the offset b) or not
        if len(c) == self.__dim - 1:
            c = np.append(c, self.__b)
        c = mat(c)
        b = c[:,self.__dim-1].T
        c_new = c[:,0:self.__dim-1].T
        c_new_sum = np.sum(c_new)
        XTc = self.X_T*c_new - self.__mean_u.T*c_new_sum
        preds_labeled = self.__surrogate_gamma*(1.0 - multiply(self.__YL, (self.X_l*XTc - self.__mean_u*XTc) + b[0,0]))
        preds_unlabeled = (self.X_u*XTc - self.__mean_u*XTc)  + b[0,0]
        # This vector has a "one" for each "numerically instable" entry; "zeros" for "good ones". 
        preds_labeled_conflict_indicator = np.sign(np.sign(preds_labeled/self.__breakpoint_for_exp - 1.0) + 1.0)
        # This vector has a one for each good entry and zero otherwise
        preds_labeled_good_indicator = (-1)*(preds_labeled_conflict_indicator - 1.0)
        preds_labeled_for_conflicts = multiply(preds_labeled_conflict_indicator,preds_labeled) 
        preds_labeled = multiply(preds_labeled,preds_labeled_good_indicator)
        # Compute values for good entries
        preds_labeled_log_exp = np.log(1.0 + np.exp(preds_labeled))
        # Compute values for instable entries
        preds_labeled_log_exp = multiply(preds_labeled_good_indicator, preds_labeled_log_exp)
        # Replace critical values with values 
        preds_labeled_final = preds_labeled_log_exp + preds_labeled_for_conflicts
        term1 = (1.0/(self.__surrogate_gamma*self.__size_l)) * np.sum(preds_labeled_final)
        preds_unlabeled_squared = multiply(preds_unlabeled,preds_unlabeled)
        term2 = (float(self.__lamU)/float(self.__size_u))*np.sum(np.exp(-self.__s * preds_unlabeled_squared))
        term3 = self.__lam * c_new.T * (self.X * XTc - self.__mean_u*XTc)
        return (term1 + term2 + term3)[0,0]

    def __getFitness_Prime(self,c):
        # check whether the function is called from the bfgs solver 
        # (that does not optimize the offset b) or not
        if len(c) == self.__dim - 1:
            c = np.append(c, self.__b)
        c = mat(c)
        b = c[:,self.__dim-1].T
        c_new = c[:,0:self.__dim-1].T
        c_new_sum = np.sum(c_new)
        XTc = self.X_T*c_new - self.__mean_u.T*c_new_sum
        preds_labeled = self.__surrogate_gamma*(1.0 - multiply(self.__YL, (self.X_l*XTc -self.__mean_u*XTc) + b[0,0]))
        preds_unlabeled = (self.X_u*XTc - self.__mean_u*XTc )+ b[0,0]
        preds_labeled_conflict_indicator = np.sign(np.sign(preds_labeled/self.__breakpoint_for_exp - 1.0) + 1.0)
        # This vector has a one for each good entry and zero otherwise
        preds_labeled_good_indicator = (-1)*(preds_labeled_conflict_indicator - 1.0)
        preds_labeled = multiply(preds_labeled,preds_labeled_good_indicator)
        preds_labeled_exp = np.exp(preds_labeled)
        term1 = multiply(preds_labeled_exp, 1.0/(1.0 + preds_labeled_exp))
        term1 = multiply(preds_labeled_good_indicator, term1)
        # Replace critical values with "1.0"
        term1 = term1 + preds_labeled_conflict_indicator
        term1 = multiply(self.__YL, term1)
        preds_unlabeled_squared_exp_f = multiply(preds_unlabeled,preds_unlabeled)
        preds_unlabeled_squared_exp_f = np.exp(-self.__s * preds_unlabeled_squared_exp_f)
        preds_unlabeled_squared_exp_f = multiply(preds_unlabeled_squared_exp_f, preds_unlabeled)
        term1_sum = np.sum(term1)
        tmp = self.X_l_T * term1 - self.__mean_u.T*term1_sum
        term1 = (-1.0/self.__size_l) * (self.X * tmp - self.__mean_u*tmp)
        preds_unlabeled_squared_exp_f_sum = np.sum(preds_unlabeled_squared_exp_f)
        tmp_unlabeled = self.X_u_T * preds_unlabeled_squared_exp_f - self.__mean_u.T * preds_unlabeled_squared_exp_f_sum
        term2 = ((-2.0 * self.__s * self.__lamU)/float(self.__size_u)) * (self.X * tmp_unlabeled - self.__mean_u*tmp_unlabeled)
        XTc_sum = np.sum(XTc)
        term3 = 2*self.__lam*(self.X * XTc - self.__mean_u*XTc)
        return array((term1 + term2 + term3).T)[0]

    def __recomputeModel(self, indi):
        self.__c = mat(indi[0]).T

############################################################################################
############################################################################################
class LinearKernel():
    """
    Linear Kernel
    """
    def __init__(self):
        pass

    def computeKernelMatrix(self, data1, data2, symmetric=False):
        """
        Computes the kernel matrix
        """
        logging.debug("Starting Linear Kernel Matrix Computation...")
        self._data1 = mat(data1)
        self._data2 = mat(data2)
        assert self._data1.shape[1] == (self._data2.T).shape[0]
        try:
            return self._data1 * self._data2.T
        except Exception, e:
            logging.error("Error while computing kernel matrix: " + str(e))
            import traceback
            traceback.print_exc()
            sys.exit()
        logging.debug("Kernel Matrix computed...")

    def getKernelValue(self, xi, xj):
        """
        Returns a single kernel value.
        """
        xi = array(xi)
        xj = array(xj)
        val = dot(xi, xj)
        return val


class DictLinearKernel():
    """
    Linear Kernel (for dictionaries)
    """
    def __init__(self):
        pass

    def computeKernelMatrix(self, data1, data2, symmetric=False):
        """
        Computes the kernel matrix
        """
        logging.debug("Starting Linear Kernel Matrix Computation...")
        self._data1 = data1
        self._data2 = data2
        self._dim1 = len(data1)
        self._dim2 = len(data2)
        self._symmetric = symmetric
        self.__km = None
        try:
            km = mat(zeros((self._dim1, self._dim2), dtype=float64))
            if self._symmetric:
                for i in xrange(self._dim1):
                    message = 'Kernel Matrix Progress: %dx%d/%dx%d' % (i, self._dim2,self._dim1,self._dim2)
                    logging.debug(message)
                    for j in xrange(i, self._dim2):
                        val = self.getKernelValue(self._data1[i], self._data2[j])
                        km[i, j] = val
                        km[j, i] = val
                return km
            else:
                for i in xrange(self._dim1):
                    message = 'Kernel Matrix Progress: %dx%d/%dx%d' % (i, self._dim2,self._dim1,self._dim2)
                    logging.debug(message)
                    for j in xrange(0, self._dim2):
                        val = self.getKernelValue(self._data1[i], self._data2[j])
                        km[i, j] = val
                return km
            
        except Exception, e:
            logging.error("Error while computing kernel matrix: " + str(e))
            sys.exit()
        logging.debug("Kernel Matrix computed...")

    def getKernelValue(self, xi, xj):
        """
        Returns a single kernel value.
        """
        val = 0.
        for key in xi:
            if key in xj:
                val += xi[key]*xj[key]
        return val

class RBFKernel():
    """
    RBF Kernel
    """
    def __init__(self, sigma):
        self.__sigma = sigma
        self.__sigma_squared_inv = 1.0 / (2* (self.__sigma ** 2) )

    def computeKernelMatrix(self, data1, data2, symmetric=False):
        """
        Computes the kernel matrix
        """
        logging.debug("Starting RBF Kernel Matrix Computation...")
        self._data1 = mat(data1)
        self._data2 = mat(data2)
        assert self._data1.shape[1] == (self._data2.T).shape[0]
        self._dim1 = len(data1)
        self._dim2 = len(data2)
        self._symmetric = symmetric
        self.__km = None
        try:
            if self._symmetric:
                linearkm = self._data1 * self._data2.T
                trnorms = mat(np.diag(linearkm)).T
                trace_matrix = trnorms * mat(np.ones((1, self._dim1), dtype = float64))
                self.__km = trace_matrix + trace_matrix.T
                self.__km = self.__km - 2*linearkm
                self.__km = - self.__sigma_squared_inv * self.__km
                self.__km = np.exp(self.__km)
                return self.__km   
            else:
                m = self._data1.shape[0]
                n = self._data2.shape[0]
                assert self._data1.shape[1] == self._data2.shape[1]
                linkm = mat(self._data1 * self._data2.T)
                trnorms1 = []
                for i in xrange(m):
                    trnorms1.append((self._data1[i] * self._data1[i].T)[0,0])
                trnorms1 = mat(trnorms1).T
                trnorms2 = []
                for i in xrange(n):
                    trnorms2.append((self._data2[i] * self._data2[i].T)[0,0])
                trnorms2 = mat(trnorms2).T
                self.__km = trnorms1 * mat(np.ones((n, 1), dtype = float64)).T
                self.__km = self.__km + mat(np.ones((m, 1), dtype = float64)) * trnorms2.T
                self.__km = self.__km - 2 * linkm
                self.__km = - self.__sigma_squared_inv * self.__km
                self.__km = np.exp(self.__km)
                return self.__km
        except Exception, e:
            logging.error("Error while computing kernel matrix: " + str(e))
            sys.exit()

    def getKernelValue(self, xi, xj):
        """
        Returns a single kernel value.
        """
        xi = array(xi)
        xj = array(xj)
        diff = xi-xj
        val = exp(-self.__sigma_squared_inv * (dot(diff, diff)))
        return val

class DictRBFKernel():
    """
    RBF Kernel (for dictionaries)
    """
    def __init__(self, sigma):
        self.__sigma = sigma
        self.__sigma_squared_inv = 1.0 / ((self.__sigma ** 2))

    def computeKernelMatrix(self, data1, data2, symmetric=False):
        """
        Computes the kernel matrix
        """
        logging.debug("Starting RBF Kernel Matrix Computation...")
        self._data1 = data1
        self._data2 = data2
        self._dim1 = len(data1)
        self._dim2 = len(data2)
        self._symmetric = symmetric
        self.__km = None
        try:
            km = mat(zeros((self._dim1, self._dim2), dtype=float64))
            if self._symmetric:
                for i in xrange(self._dim1):
                    message = 'Kernel Matrix Progress: %dx%d/%dx%d' % (i, self._dim2,self._dim1,self._dim2)
                    logging.debug(message)
                    for j in xrange(i, self._dim2):
                        val = self.getKernelValue(self._data1[i], self._data2[j])
                        km[i, j] = val
                        km[j, i] = val
                return km
            else:
                for i in xrange(0, self._dim1):
                    message = 'Kernel Matrix Progress: %dx%d/%dx%d' % (i, self._dim2,self._dim1,self._dim2)
                    logging.debug(message)
                    for j in xrange(0, self._dim2):
                        val = self.getKernelValue(self._data1[i], self._data2[j])
                        km[i, j] = val
                return km
        except Exception, e:
            logging.error("Error while computing kernel matrix: " + str(e))
            sys.exit()
        logging.info("Kernel Matrix computed...")

    def getKernelValue(self, xi, xj):
        """
        Returns a single kernel value.
        """
        diff = xi.copy()
        for key in xj:
            if key in diff:
                diff[key]-=xj[key]
            else:
                diff[key]=-xj[key]
        diff = diff.values()
        val = exp(-self.__sigma_squared_inv * (dot(diff, diff)))
        return val