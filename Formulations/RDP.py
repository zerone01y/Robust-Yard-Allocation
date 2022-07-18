import pyomo.environ as pe

from helper.Prob_def import YAP_Prob
from helper.model_functions import *


def robust_dual_points(Param, **kwargs):  # k: integer starts from 0
    """
    A p
    """
    M = pe.AbstractModel("Robust_DMP")

    M.V = pe.RangeSet(Param.V)
    M.T = pe.RangeSet(Param.T)
    M.T_cycle = pe.RangeSet(Param.T_cycle)
    B = [b for q in Param.G_k for b in Param.B_k[Param.G_k[q][0]]]
    M.B = pe.SetOf(B)

    M.G_k = pe.SetOf(Param.G_k)
    M.VTcycle = M.V * M.T_cycle
    M.VT = M.V * M.T
    M.I = pe.RangeSet(Param.I)
    I = [i for b in B for i in Param.I_b[b]]
    M.Iq = pe.SetOf(I)
    YAP_Prob.Param_anouncement(M, Param)
    M.cost1 = 1
    M.cost2 = 1
    M.CostDistance = Param.CostDistance

    M.IV = M.Iq * M.V
    M.IVT = M.IV * M.T
    M.IVTcycle = M.IV * M.T_cycle
    M.KVTcycle = M.G_k * M.VTcycle
    M.KVT = M.G_k * M.VT
    M.VV = M.V * M.V
    M.VVT = M.VV * M.T_cycle
    M.D_T = pe.Param(M.VV, default=0)
    M.h = pe.Var(M.VVT, within=pe.NonNegativeReals)

    def param_init_NG(M, g):
        return len(Param.G_k[g])

    M.N_g = pe.Param(M.G_k, default=1, initialize=param_init_NG)

    # M.pi = pe.Var(M.G_k, within=pe.Reals)
    M.rho = pe.Var(M.V, within=pe.NonNegativeReals)
    M.f = pe.Var(M.VTcycle, within=pe.NonNegativeReals)
    M.g = pe.Var(M.KVTcycle, within=pe.NonNegativeReals)
    M.a = pe.Var(M.VTcycle, within=pe.NonNegativeReals)

    # self.k              = pe.Param(self.I, self.B)

    M.r1 = pe.Var(M.Iq, within=pe.Reals)
    M.r2 = pe.Var(M.IVT, within=pe.NonNegativeReals)
    M.e = pe.Var(M.IV, within=pe.NonNegativeReals)
    M.alpha = pe.Var(M.B, M.T, within=pe.NonNegativeReals)
    M.beta = pe.Var(M.Iq, M.T, within=pe.NonNegativeReals)

    def plan_cost(M, q):
        k = Param.G_k[q][0]
        B = Param.B_k[k]
        return sum(M.r1[i] for i in Param.I_k[k]) \
               - sum(sum(M.YC[b] * M.alpha[b, t] for b in B) + sum(M.M[i] * M.beta[i, t] for i in Param.I_k[k])
                     for t in M.T)
        # -M.pi[q] >=0

    M.pi = pe.Expression(M.G_k, rule=plan_cost)

    def dual_obj(M):
        return pe.sum_product(M.N_g, M.pi) \
               + pe.sum_product(M.Demand, M.f) \
               + sum(M.D_T[j1, j2] * M.h[j1, j2, t] for (j1, j2, t) in M.VVT) \
               + sum((sum(M.Demand[j, t] for t in M.T) + sum(M.D_T[j1, j] for j1 in M.V)) * M.rho[j] for j in M.V) \
               + Param.y_distance * Param.CostDistance \
               + M.cut_penalty

    #

    M.total_cost = pe.Objective(rule=dual_obj, sense=pe.maximize)

    # M.lbd = pe.Constraint(M.G_k, rule=plan_cost)

    def eta1(M, j1, j2, t):
        return -M.f[j2, t] + M.h[j1, j2, t] <= 0

    def eta2(M, j1, j2, t):
        return M.h[j1, j2, t] - 4 * (M.cost1 + M.cost2) * M.a[j1, t] <= 0

    M.eta1 = pe.Constraint(M.VVT, rule=eta1)
    M.eta2 = pe.Constraint(M.VVT, rule=eta2)

    def v1(M, k, t):  # BT
        return M.cost1 - M.alpha[k, t] >= 0

    def v2(M, i, t):  # IT
        return M.cost2 - sum(M.beta[i0, t] for i0 in Param.Neighbor_Set[i]) >= 0

    def z(M, i, j):  #
        if cut:
            cut_info = sum((M.cutminus[c, j]
                            + M.cutplus[c, j]) * Indicator(cut_group[c], q) * Indicator(Param.G_k[q], k) * Indicator(
                Param.I_k[k], i)
                           for q in M.G_k for k in Param.G_k[q] for c in M.CUT_GROUP)
        else:
            cut_info = 0
        return M.r1[i] \
               + M.Capacity * M.e[i, j] + cut_info + M.Capacity * M.rho[j] \
               <= M.CostDistance * Param.distance[i][j - 1]

    #

    def x(M, i, j, t):
        return -M.e[i, j] - sum(Indicator(Param.I_b[b], i) * M.alpha[b, t] for b in M.B) \
               - sum(M.beta[i0, t] for i0 in Param.Neighbor_Set[i]) + sum(M.r2[i, j, t1] for t1 in M.T) \
               + M.f[j, t] \
               - sum(
            Indicator(Param.I_k[k], i) * Indicator(Param.G_k[q], k) * sum(M.g[q, j, t1] for t1 in M.T_cycle) for q in
            M.G_k for k in Param.G_k[q]) \
               <= 0

    def y(M, i, j, t):
        return - sum(Indicator(Param.I_b[k], i) * M.alpha[k, t] for k in M.B) \
               - sum(M.beta[i0, t] for i0 in Param.Neighbor_Set[i]) - sum(
            M.r2[i, j, t1] for t1 in range(max(1, t - Param.Loading_periods + 1), t + 1)) \
               <= - sum(Indicator(Param.I_k[k], i) * Indicator(Param.G_k[q], k) * sum(
            M.g[q, j, t1] for t1 in M.T_cycle if Loading_Set(M, t, t1))
                        for q in M.G_k for k in Param.G_k[q])

    M.v1 = pe.Constraint(M.B, M.T, rule=v1)
    M.v2 = pe.Constraint(M.Iq, M.T, rule=v2)
    M.z = pe.Constraint(M.IV, rule=z)
    M.x = pe.Constraint(M.IVTcycle, rule=x)
    M.y = pe.Constraint(M.IVT, rule=y)

    def u0(M, j):  # theta
        if Param.U_set == 'U_g':
            A = M.Schedule[j] - M.gamma[j]
        elif Param.U_set == 'U_l':
            A = M.Schedule[j]
        else:
            print("Wrong Uncertainty Type!")
            return None
        B = min(Param.Schedule[j] + M.gamma[j], len(M.T_cycle))
        return sum(M.a[j, tau] for tau in range(A, B + 1)) == 1
        # + M.l[i, j]
        # return g

    def mu(M, q, j, t):
        return M.g[q, j, t] - (M.cost1 + M.cost2) * M.a[j, t] <= 0

    def u1(M, j):
        return sum(M.a[j, t] for t in M.T_cycle) == 1

    M.u0 = pe.Constraint(M.V, rule=u0)
    M.mu = pe.Constraint(M.KVTcycle, rule=mu)
    M.u1 = pe.Constraint(M.V, rule=u1)

    return M
