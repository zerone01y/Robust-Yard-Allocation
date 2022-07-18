"""Master problem formulation (dual) for column generation
Uncertainty set: sum(a)=1, sum_{t\in T_j}a_{jt}=1 (Relaxed)
"""

from __future__ import division
import pyomo.environ as pe
from helper.model_functions import *

# DUAL_DUAL problem/RAYTOP (manuscript version)
# a > 0
# alpha < 0
# beta < 0
# e > 0
# f > 0

def relaxed_primal_prob(Param, firststage=False):
    """

    Args:
        Param: Model params
        firststage: set to 0 if only intend to evaluate a given template.
        The template should be provided in Param

    Returns: pyomo problem

    """
    M = pe.ConcreteModel("RAYTOP")

    M.V = pe.Set(initialize = range(1, Param.V+1))
    M.T = pe.Set(initialize = range(1, Param.T+1))
    M.T_cycle = pe.Set(initialize = range(1, Param.T_cycle+1))
    M.B = pe.Set(initialize = range(1, Param.B+1))
    M.I = pe.Set(initialize = range(1, Param.I+1))

    M.VTcycle = M.V * M.T_cycle
    M.VT = M.V * M.T
    M.IV = M.I * M.V
    M.IVT = M.I* M.V * M.T
    M.BT = M.B * M.T
    M.IT =M.I * M.T
    M.VV = M.V * M.V
    M.VVT = M.VV * M.T_cycle

    M.z = pe.Var(M.IV, within=pe.Binary)
    #Vars: second stage
    M.x = pe.Var(M.IVT, within=pe.NonNegativeReals)
    M.y = pe.Var(M.IVT, within=pe.PositiveReals)
    M.vio1 = pe.Var(M.BT, within=pe.NonNegativeReals)
    M.vio2 = pe.Var(M.IT, within=pe.NonNegativeReals)
    M.mu = pe.Var(M.IVT, within=pe.NonNegativeReals)
    M.D_T = pe.Param(M.VV, default=0)
    M.CostDistance = Param.CostDistance
    M.u0 = pe.Var(M.V, within=pe.Reals)
    M.u1 = pe.Var(M.V, within=pe.Reals)
    M.eta1 = pe.Var(M.VVT, within=pe.NonNegativeReals)
    M.eta2 = pe.Var(M.VVT, within=pe.NonNegativeReals)

    M.K = pe.RangeSet(len(Param.KS))



    if firststage:
        def dedicated(M, i):
            return sum(M.z[i, j] for j in M.V) == 1
        M.r1 = pe.Constraint(M.I, rule=dedicated)
    else:
        M.del_component(M.z)
        M.z = pe.Param(M.IV, default=0, mutable=True, initialize=Param.data.data("z"))
        M.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    #
    # Objective
    #
    def _Cost_K(M, k):
        B_k = [Param.KS[k - 1]] if type(Param.KS[k - 1]) == int else Param.KS[k - 1]
        I_k = Param.I_b[Param.KS[k - 1]] if type(Param.KS[k - 1]) == int else \
            Param.I_b[Param.KS[k - 1][0]] + Param.I_b[Param.KS[k - 1][1]]
        return sum(Param.cost1 * sum(M.vio1[b, t] for b in B_k) +
                   Param.cost2 * sum(M.vio2[i, t] for i in I_k)
                   for t in M.T)

    M.Cost_K = pe.Expression(M.K, rule=_Cost_K)


    def total_violation(M):
        return pe.summation(M.u1) + pe.summation(M.u0) + pe.summation(M.Cost_K) \
               + Param.CostDistance * (sum(Param.distance[i][j - 1] * M.z[i, j]
                                       for i in M.I
                                       for j in M.V) + Param.y_distance)

    M.total_cost = pe.Objective(rule=total_violation, sense=pe.minimize)

    #
    # Constraints
    #

    def c1(M, i, j):
        return - sum(M.x[i, j, t] for t in M.T) \
               + M.z[i, j] * Param.Capacity >= 0

    M.e = pe.Constraint(M.I, M.V, rule=c1)

    def c2(M, j, t):#discharging
        return sum(M.x[i, j, t] for i in M.I) - sum(M.eta1[j1, j, t] for j1 in M.V) \
               >= Param.Demand[j, t]

    M.f = pe.Constraint(M.V, M.T_cycle, rule=c2)

    def c3(M, i, j, tau):#loading
        return -sum(M.x[i, j, t] for t in M.T_cycle) + sum(M.y[i, j, t]
            for t in range(tau, tau + Param.Loading_periods)) \
               + M.mu[i, j, tau] >= 0

    M.g = pe.Constraint(M.I, M.V, M.T_cycle, rule=c3)

    def com1(M, j):
        return sum(M.x[i, j, t] for i in M.I for t in M.T) >=\
        sum(Param.Demand[j, t] for t in M.T_cycle)+\
        sum(Param.D_T[j1, j] for j1 in M.V)
    M.rho = pe.Constraint(M.V, rule=com1)

    def h1(M, j1, j2, t):
        return M.eta1[j1, j2, t] + M.eta2[j1, j2, t] >= Param.D_T[j1, j2]

    M.h1 = pe.Constraint(M.VVT, rule=h1)

    def c4(M, k, t):
        return sum(sum((M.x[i, j, t] + M.y[i, j, t])
                       for j in M.V)
                   for i in Param.I_b[k]) - M.vio1[k, t] \
               <= Param.YC

    M.alpha = pe.Constraint(M.B, M.T, rule=c4)

    def c5(M, i, t):
        return sum((sum((M.x[i0, j, t] + M.y[i0, j, t])
                        for j in M.V)
                    - M.vio2[i0, t]) for i0
                   in Param.Neighbor_Set[i]) \
               <= Param.M[i]

    M.beta = pe.Constraint(M.I, M.T, rule=c5)


    # Uncertainty
    def u_0(M, j, t):
        I_sj = 1 if Indicator_arrival(j, t, Param) else 0
        return - (Param.cost1 + Param.cost2) * sum(M.mu[i, j, t] for i in M.I) \
               + M.u0[j] \
               + I_sj * M.u1[j] - 4 * (Param.cost1 + Param.cost2) * sum(M.eta2[j, j1, t] for j1 in M.V) >= 0

    M.a = pe.Constraint(M.V, M.T_cycle, rule=u_0)

    def cutnum(M, i):
        return sum(M.x[i, j, t] - M.y[i, j, t] for j in M.V for t in M.T) == 0

    # M.cut = pe.Constraint(M.I, rule=cutnum)
    def r2(M, i, j):
        return sum(-M.x[i, j, t] + M.y[i, j, t] for t in M.T) >= 0

    M.r2 = pe.Constraint(M.I, M.V, rule=r2)
    return M

