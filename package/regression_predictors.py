"""
GRN Co-Expression Inference with regression methods
===================================================
This module allows to infer co-expression  Gene Regulatory Networks using
gene expression data (RNAseq or Microarray). This module implements severall
inference algorithms based on regression, using `scikit-learn`_.

.. _scikit-learn:
    https://scikit-learn.org
"""
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import RandomizedLasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lars
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np

__author__ = "Sergio Peignier, Pauline Schmitt"
__copyright__ = "Copyright 2019, The GReNaDIne Project"
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Sergio Peignier"
__email__ = "sergio.peignier@insa-lyon.fr"
__status__ = "pre-alpha"

def GENIE3(X,y,**rf_parameters):
    """
    GENIE3, score predictor function based on `scikit-learn`_
    RandomForestRegressor.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **rf_parameters: Named parameters for the sklearn RandomForestRegressor

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        RandomForestRegressor to the regulatory relationship between the target
        gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = GENIE3(tfs,tg)
        >>> scores
        array([0.11983888, 0.28071399, 0.59944713])
    """
    regressor = RandomForestRegressor(**rf_parameters)
    regressor.fit(X, y)
    scores = regressor.feature_importances_
    return(scores)

def XGENIE3(X,y,**rf_parameters):
    """
    XGENIE3, score predictor function based on `scikit-learn`_
    ExtraTreesRegressor.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **rf_parameters: Named parameters for the sklearn RandomForestRegressor

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        ExtraTreesRegressor to the regulatory relationship between the target
        gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = XGENIE3(tfs,tg)
        >>> scores
        array([0.24905241, 0.43503283, 0.31591477])
    """
    regressor = ExtraTreesRegressor(**rf_parameters)
    regressor.fit(X, y)
    scores = regressor.feature_importances_
    return(scores)

def GRNBoost2(X,y,**boost_parameters):
    """
    GRNBoost2 score predictor based on `scikit-learn`_
    GradientBoostingRegressor.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **boost_parameters: Named parameters for GradientBoostingRegressor

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        GradientBoostingRegressor to the regulatory relationship between the
        target gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = GRNBoost2(tfs,tg)
        >>> scores
        array([0.83904506, 0.01783977, 0.14311517])
    """
    regressor = GradientBoostingRegressor(**boost_parameters)
    regressor.fit(X, y)
    scores = regressor.feature_importances_
    return(scores)

def BayesianRidgeScore(X,y,**brr_parameters):
    """
    Score predictor based on `scikit-learn`_ BayesianRidge regression.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **brr_parameters: Named parameters for sklearn BayesianRidge regression

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        sklearn BayesianRidge regressor to the regulatory relationship between
        the target gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = BayesianRidgeScore(tfs,tg)
        >>> scores
        array([1.32082000e-03, 6.24177371e-05, 3.32319918e-04])
    """
    regressor = BayesianRidge(**brr_parameters)
    regressor.fit(X, y)
    scores = np.abs(regressor.coef_)
    return(scores)

def SVR_score(X,y,**svr_parameters):
    """
    Score predictor based on `scikit-learn`_ SVR (Support Vector Regression).

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **svr_parameters: Named parameters for sklearn SVR regression

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        sklearn SVR regressor to the regulatory relationship between
        the target gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = SVR_score(tfs,tg)
        >>> scores
        array([[-0.38156814,  0.28128811, -1.0230867 ]])
    """
    svr_parameters["kernel"] = 'linear'
    regressor = SVR(**svr_parameters)
    regressor.fit(X, y)
    scores = np.abs(regressor.coef_)
    return(scores[0])

def Lasso_score(X,y,**l1_parameters):
    """
    Score predictor based on `scikit-learn`_ Lasso regression.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **l1_parameters: Named parameters for sklearn Lasso regression

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        sklearn Lasso regressor to the regulatory relationship between
        the target gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = Lasso_score(tfs,tg, alpha=0.01)
        >>> scores
        array([0.13825495, 0.94939204, 0.19118214])
    """
    regressor = Lasso(**l1_parameters)
    regressor.fit(X, y)
    scores = np.np.abs(regressor.coef_)
    return(scores)

def LassoLars_score(X,y,**l1_parameters):
    """
    Score predictor based on `scikit-learn`_ LassoLars regression.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **l1_parameters: Named parameters for sklearn Lasso regression

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        sklearn LassoLars regressor to the regulatory relationship between the
        target gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = LassoLars_score(tfs,tg, alpha=0.01)
        >>> scores
        array([0.12179406, 0.92205553, 0.15503451])
    """
    regressor = LassoLars(**l1_parameters)
    regressor.fit(X, y)
    scores = np.abs(regressor.coef_)
    return(scores)

def stability_randomizedlasso(X,y,**rl_parameters):
    """
    Score predictor based on `scikit-learn`_ randomizedlasso stability selection.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **rl_parameters: Named parameters for sklearn randomizedlasso
    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        sklearn randomizedlasso stability selection to the regulatory
        relationship between the target gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = stability_randomizedlasso(tfs,tg)
        >>> scores
        array([0.11 , 0.17 , 0.085])
    """
    regressor = RandomizedLasso(**rl_parameters)
    regressor.fit(X,y)
    scores = np.abs(regressor.scores_)
    return(scores)

def TIGRESS(X,
            y,
            nsplit=100,
            nstepsLARS=5,
            alpha=0.4,
            scoring="area"):
    """
    TIGRESS score predictor based on stability selection.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        nsplit (int): number of splits applied,
            i.e., randomization tests, the highest the best
        nstepsLARS (int): number of steps of LARS algorithm,
            i.e., number of non zero coefficients to keep (Lars parameter)
        alpha: Noise multiplier coefficient,
            Each transcription factor expression is multiplied by a
            random variable $\in [\alpha,1]$
        scoring (str): option used to score each possible link
            only "area" and "max" options are available

    Returns:
        numpy.array: co-regulation scores

        The i-th element of the score array represents the score assigned by the
        sklearn randomizedlasso stability selection to the regulatory
        relationship between the target gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = TIGRESS(tfs,tg)
        >>> scores
        array([349.   , 312.875, 588.125])
    """
    n,p = X.shape
    halfsize = int(n/2)
    if nstepsLARS > p:
        nstepsLARS = p-1
    freq = np.zeros((p, nstepsLARS))
    i = 0
    while i < nsplit:
        # Randomly reweight each variable (TF expression)
        random_perturbation = np.random.uniform(low=alpha, high=1.0, size=p)
        X *= random_perturbation
        # Randomly split the sample in two sets
        X_1,X_2,y_1,y_2 = train_test_split(X,y,test_size=halfsize, shuffle=True)
        for X_i,y_i in [[X_1, y_1],[X_2,y_2]]:
            if y_i.std() > 0:
                # run LARS on each subsample and collect variables are selected
                lars = Lars(normalize=False, n_nonzero_coefs=nstepsLARS)
                lars.fit(X_i,y_i)
                # collect the presence of the coefficients along the path
                path = lars.coef_path_
                if path.shape[1] < nstepsLARS+1:
                    path_add = np.tile(path[:,-1],(nstepsLARS+1 - path.shape[1],1)).T
                    path = np.hstack((path,path_add))
                freq += np.abs(np.sign(path[:,1:]))
                i += 1
        X /= random_perturbation
    # normalize frequence in [0,1] to get stability curves
    freq /= 2*halfsize
    if (scoring=="area"):
        score = np.cumsum(freq,axis=1)/np.arange(1,nstepsLARS+1,1)
    if (scoring=="max"):
        score = np.maximum.accumulate(freq,axis=1)
    return(score[:,nstepsLARS - 1])

def AdaBoost_regressor (X, y,  **adab_parameters):
    """
    AdaBoost regressor, score predictor function based on `scikit-learn`_
    AdaBoostRegressor.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **adab_parameters: Named parameters for the sklearn AdaBoostRegressor

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        AdaBoostRegressor to the regulatory relationship between the target
        gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                       index =["c1","c2","c3","c4","c5"],
                       columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = AdaBoost_regressor(tfs,tg)
        >>> scores
        array([0.32978247, 0.3617295 , 0.28896647])
    """
    regressor = AdaBoostRegressor(**adab_parameters)
    regressor.fit(X, y)
    scores = regressor.feature_importances_
    return scores


def my_crazy_function(arg1):
    """Does nothing useful

    Args:
        arg1 (int): useless arg

    Returns:
        str : nonsense output

    Examples:
        my example is so nice

        >>> my_crazy_function(10)
        "eeeee"
    """
    return("eeeee")
