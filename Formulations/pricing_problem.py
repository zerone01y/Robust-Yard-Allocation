import pyomo.environ as pe
from helper.Prob_def import YAP_Prob
from helper.Func import *


def pricing_prob(Param, k, alns=False):  # k: integer starts from 0
    M = pe.AbstractModel("CG_pricing%d" % k)
    M.k = k + 1  # index of block section
    M.ks = Param.KS[Param.G_k[k + 1][0] - 1]  # index of blocks in this section
    M.cost1 = 1
    M.cost2 = 1
    M.CostDistance = Param.CostDistance
    M.V = pe.RangeSet(Param.V)

    if type(M.ks) is int:
        M.B = pe.SetOf([M.ks])
        M.I = pe.SetOf(Param.I_b[M.ks])
    else:
        M.B = pe.SetOf(M.ks)
        M.I = pe.SetOf(Param.I_b[M.ks[0]] + Param.I_b[M.ks[1]])
    M.T = pe.RangeSet(Param.T)
    M.T_cycle = pe.RangeSet(Param.T_cycle)
    M.VV = M.V * M.V
    M.VVT = M.VV * M.T_cycle
    # Incoming containers in shift t

    M.Capacity = pe.Param(initialize=Param.Capacity)
    M.YC = pe.Param(M.B,
                    default=Param.YC)
    M.M = pe.Param(M.I, initialize={i: Param.M[i] for i in M.I})
    # self.k              = pe.Param(self.I, self.B)

    M.Loading_periods = pe.Param(initialize=Param.Loading_periods)  # Loading periods

    M.f = pe.Param(M.V, M.T, default=0, mutable=True)  #
    #M.a = pe.Param(M.V, M.T, default=0, mutable=True)
    M.g = pe.Param(M.V, M.T, default=0, mutable=True)
    M.pi = pe.Param(default=0, mutable=True)
    M.rho = pe.Param(M.V, default=0, mutable=True)
    M.z = pe.Var(M.I, M.V, within=pe.Binary)
    M.x = pe.Var(M.I, M.V, M.T_cycle, within=pe.NonNegativeReals)
    M.y = pe.Var(M.I, M.V, M.T, within=pe.NonNegativeReals)
    M.vio1 = pe.Var(M.B, M.T, within=pe.NonNegativeReals)
    M.vio2 = pe.Var(M.I, M.T, within=pe.NonNegativeReals)
    M.cutminus = pe.Param(M.V, default=0, mutable=True)
    M.cutplus = pe.Param(M.V, default=0, mutable=True)

    M.Nv = pe.Param(M.V, mutable=True, default=0)

    # -sum(M.Capacity*M.rho[j]*M.z[i,j] for i in M.I for j in M.V)\
    def section_cost(M):
        return M.cost1 * pe.summation(M.vio1) \
               + M.cost2 * pe.summation(M.vio2) \
               + M.CostDistance * sum(Param.distance[i][j - 1] * M.z[i, j]
                                      for i in M.I for j in M.V) \
               - sum(M.f[j, t] * M.x[i, j, t]
                     + M.g[j, t] * (
                    sum(M.y[i, j, t0] for t0 in range(t, t + M.Loading_periods))
                    - sum(M.x[i, j, t0] for t0 in M.T_cycle))
                     for i in M.I for t in M.T_cycle for j in M.V) \
               - M.pi \
               - sum((M.cutplus[j] + M.cutminus[j]) * M.z[i, j] for i in M.I for j in M.V)

    M.section_cost = pe.Objective(rule=section_cost, sense=pe.minimize)

    def r1(M, i):
        return sum(M.z[i, j] for j in M.V) == 1

    M.r1 = pe.Constraint(M.I, rule=r1)

    def r2(M, i, j):
        return - sum(M.x[i, j, t] for t in M.T_cycle) + M.Capacity * M.z[i, j] >= 0

    M.r2 = pe.Constraint(M.I, M.V, rule=r2)

    def alpha(M, b, t):
        if t> Param.T_cycle:
            return - sum(  M.y[i, j, t]
                       for j in M.V
                   for i in Param.I_b[b]) +  M.vio1[b, t] \
                     >= - M.YC[b]
        return - sum(sum((M.x[i, j, t] + M.y[i, j, t])
                       for j in M.V)
                   for i in Param.I_b[b]) +  M.vio1[b, t] \
                     >= - M.YC[b]

    M.alpha = pe.Constraint(M.B, M.T, rule=alpha)

    def demand(M, j, t):
        return sum(M.x[i,j ,t] for i in M.I)<=Param.data.data()["Demand"][j, t]
    #M.demand=pe.Constraint(M.V, M.T_cycle, rule=demand)


    def p6(M, i, t):
        if t> Param.T_cycle:
            return - sum((sum(M.y[i0, j, t]
                              for j in M.V)
                          - M.vio2[i0, t]) for i0
                         in Param.Neighbor_Set[i]) \
                   >= - M.M[i]
        return - sum((sum((M.x[i0, j, t] + M.y[i0, j, t])
                        for j in M.V)
                    - M.vio2[i0, t]) for i0
                   in Param.Neighbor_Set[i]) \
                      >= - M.M[i]

    M.beta = pe.Constraint(M.I, M.T, rule=p6)

    def p4(M, i, j):
        return -sum(M.x[i, j, t] for t in M.T_cycle) + sum(M.y[i, j, t] for t in M.T) >= 0

    M.u4 = pe.Constraint(M.I, M.V, rule=p4)

    def positive_constraint(M, i, j, t):
        return sum(M.x[i, j, t0] for t0 in M.T_cycle) \
               - sum(M.y[i, j, t0] for t0 in range(t, t + M.Loading_periods)) >= 0

    M.u2 = pe.Constraint(M.I, M.V, M.T_cycle, rule=positive_constraint)


    if alns:
        M.unassignedNumber = pe.Param(M.V, default=len(M.I), mutable=True)
        def unassignedCons(M, j):
            return sum(M.z[i, j] for i in M.I) <= M.unassignedNumber[j]
        M.unassignedCons = pe.Constraint(M.V,rule=unassignedCons)


    return M
