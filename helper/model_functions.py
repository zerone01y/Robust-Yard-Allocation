
import numpy as np
import pyomo.environ as pe

from helper.data_types import solu_col, DefaultDict

STB_VAR_MAPPING = {"f": 0, "g": 1, "a": 2, "pi": 3}


def Indicator(s, v):
    """ Indicator function I_{set}(value)

    Args:
        s: set or value (int, float, set)
        v: value

    Returns: bool

    """
    if type(s) not in [int, float]:
        return 1 if v in s else 0
    else:  # int or float
        return 1 if v == s else 0



def Indicator_arrival(j, t, Param):
    """Given a schedule M.Schedule, returns if vessel j can arrive at time t
    No wrap_around

    Args:
        j: vessel index
        t: time index
        Param: model param

    Returns: bool

    """
    A = Param.Schedule[j - 1]
    B = min(Param.Schedule[j - 1] + Param.gamma[j - 1], Param.T_cycle)
    return (A <= t <= B)



# Validation: vessel can only arrive within the cycle time
def Demand_validate(M, d, j, t):
    if not t in M.T_cycle:
        return d == 0
    else:
        return True

