"""
This is a title
===============
fezioshgfdihgkjshqfkjlsd
"""

import numpy as np

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
