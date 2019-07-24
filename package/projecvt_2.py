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


def the_function_42():
    """
    The function 42 returns 42
    """
    return(42)


def public_fn_with_sphinxy_docstring(name, state=None):
    """This function does something.

    :param name: The name to use.
    :type name: str.
    :param state: Current state to be in.
    :type state: bool.
    :returns: int -- the return code.
    :raises: AttributeError, KeyError

    """
    return(0)

def random_number_generator(arg1, arg2):
    """Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    """
    return(42)

def add_two_numbers(arg1, arg2):
    """Summary line for add two nbs.

    Extended description of function. :math:`\mu = 10`

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    Examples
    -------
    My dump example

    >>> add_two_numbers(10,10)
    20
    >>> add_two_numbers(20,10)
    30

    """
    return(arg1+arg2)

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
