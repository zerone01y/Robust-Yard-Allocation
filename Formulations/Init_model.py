'''
Template model (YTG) from
X. Jiang, L. H. Lee,  E. P. Chew,
Y. Han, and B. C. Tan,
 “A container yard storage strategy for
 improving land utilization and operation
 efficiency in a transshipment hub port,”
 Eur. J. Oper. Res., vol. 221, no. 1, pp. 64–73, 2012.
'''

from __future__ import division
import pyomo.environment as pe
from helper.model_functions import Indicator
from helper.Prob_def import *


def rough_cut_model(Param):
    M = pe.AbstractModel("YTP")

    def lambda_initialize(M, j, t):
        if 0 <= t - Param.Schedule[j] <= Param.Loading_periods - 1:
            return 1
        else:
            return 0

    M.cost1 = 1
    M.cost2 = 1
    M.CostDistance = Param.CostDistance
    M.V = pe.RangeSet(Param.V)
    M.T = pe.RangeSet(Param.T)
    M.B = pe.RangeSet(Param.B)
    M.I = pe.RangeSet(Param.I)
    M.lbd = pe.Param(M.V, M.T, initialize=lambda_initialize)

    M.I_b = pe.Set(M.B, within=M.I,
                   initialize=Param.I_b)
    M.Demand = pe.Param(M.V, M.T, within=pe.NonNegativeReals,
                        default=0)  # Incoming containers in shift t
    M.D_T = pe.Param(M.V, M.V, default=0)
    M.Loading_periods = pe.Param(initialize=Param.Loading_periods)  # Loading periods
    M.Capacity = pe.Param(initialize=Param.Capacity)
    M.z = pe.Var(M.I, M.V, within=pe.Binary)
    M.vio1 = pe.Var(M.B, M.T, within=pe.NonNegativeReals)
    M.vio2 = pe.Var(M.I, M.T, within=pe.NonNegativeReals)

    def init_distance(M, k, v):
        if k in Param.distance:
            return Param.distance[k][v - 1]
        elif (k, k + 1) in Param.distance:
            return Param.distance[k, k + 1][v - 1]
        else:
            return Param.distance[k - 1, k][v - 1]

    M.distance = pe.Param(M.B, M.V, initialize=init_distance)

    def violations(M):
        return M.cost1 * pe.summation(M.vio1) + M.cost2 * pe.summation(M.vio2) \
               + M.CostDistance * sum(M.z[i, j] * sum(Indicator(Param.I_b[k], i) * M.distance[k, j]
                                                      for k in M.B)
                                      for i in M.I for j in M.V)

    M.violations = pe.Objective(rule=violations, sense=pe.minimize)

    def c1(M, i):
        return sum(M.z[i, j] for j in M.V) == 1

    M.c1 = pe.Constraint(M.I, rule=c1)

    def c2(M, j):
        return M.Capacity * sum(M.z[i, j] for i in M.I) >=\
               sum(M.Demand[j, t] for t in M.T)\
               +sum(M.D_T[j1,j] for j1 in M.V)


    M.c2 = pe.Constraint(M.V, rule=c2)

    def c3(M, k, t):
        return sum(M.lbd[j, t] * M.z[i, j] for j in M.V for i in M.I_b[k]) \
               - M.vio1[k, t] <= 2

    M.c3 = pe.Constraint(M.B, M.T, rule=c3)

    def c4(M, i, t):
        return sum(M.lbd[j, t] * M.z[i1, j] for i1 in Param.Neighbor_Set[i] for j in M.V) \
               - M.vio2[i, t] <= 1

    M.c4 = pe.Constraint(M.I, M.T, rule=c4)

    return M

