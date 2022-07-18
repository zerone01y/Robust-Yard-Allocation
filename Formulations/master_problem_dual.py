"""Master problem formulation (dual) for the set partitioning formulation
allow cuts (i.e. columns in the primal view), allow stabilization
Uncertainty set: sum(a)=1, sum_{t\in T_j}a_{jt}=1
"""

from __future__ import division
import pyomo.environ as pe
from helper.model_functions import *
from helper.Prob_def import YAP_Prob


def master_problem_dual(Param, delta=0.001, penaltyweight=None, cut=None, **kwargs):
    name = "Stabilized_RMPDual"
    model = pe.ConcreteModel(name)
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.slack = pe.Suffix(direction=pe.Suffix.IMPORT)

    YAP_Prob.Set_announcement(model, Param)
    YAP_Prob.Param_anouncement(model, Param)
    model.cost1 = 1
    model.cost2 = 1

    model.G_k = pe.SetOf(Param.G_k) #G_k: the set of of block section group

    def param_init_NG(M, g):
        return len(Param.G_k[g])

    model.N_g = pe.Param(model.G_k, default=1, initialize=param_init_NG)
    # N_g: number of block sections in each block section group

    model.KVTcycle = model.G_k * model.VTcycle
    model.KVT = model.G_k * model.VT
    model.VV = model.V * model.V
    model.VVT = model.VV * model.T_cycle
    model.D_T = pe.Param(model.VV, default=0)



    model.rho = pe.Var(model.V, within=pe.NonNegativeReals)
    model.pi = pe.Var(model.G_k, within=pe.Reals)
    model.f = pe.Var(model.VTcycle, within=pe.NonNegativeReals)
    model.g = pe.Var(model.KVTcycle, within=pe.NonNegativeReals)
    model.a = pe.Var(model.VTcycle, within=pe.NonNegativeReals)

    model.h = pe.Var(model.VVT, within=pe.NonNegativeReals)

    if penaltyweight:
        model.drct = pe.SetOf(["+", "-"])

        model.stbf = pe.Var(model.VTcycle, model.drct, within=pe.NonNegativeReals)
        model.stbg = pe.Var(model.KVTcycle, model.drct, within=pe.NonNegativeReals)
        model.penaltyweightf = pe.Param(default=penaltyweight, mutable=True)
        model.penaltyweightg = pe.Param(default=penaltyweight, mutable=True)
        model.scf = pe.Param(model.VTcycle, default=0, mutable=True)
        model.scg = pe.Param(model.KVTcycle, default=0, mutable=True)
        model.deltaf = pe.Param(model.VTcycle, default=delta, mutable=True)
        model.deltag = pe.Param(model.KVTcycle, default=delta, mutable=True)

        def penalty(M):
            return M.penaltyweightf * (pe.summation(M.stbf)) \
                   + M.penaltyweightg * (pe.summation(M.stbg))

        model.penalty = pe.Expression(rule=penalty)

        def sigmaf(M, j, t, dr):
            if dr == "+":
                return M.f[j, t] - M.stbf[j, t, dr] <= M.scf[j, t] + M.deltaf[j, t]
            else:
                return M.f[j, t] + M.stbf[j, t, dr] >= M.scf[j, t] - M.deltaf[j, t]

        def sigmag(M, q, j, t, dr):
            if dr == "+":
                return M.g[q, j, t] - M.stbg[q, j, t, dr] <= M.scg[q, j, t] + M.deltag[q, j, t]
            else:
                return M.g[q, j, t] + M.stbg[q, j, t, dr] >= M.scg[q, j, t] - M.deltag[q, j, t]

        model.sigmaf = pe.Constraint(model.stbf_index, rule=sigmaf)
        model.sigmag = pe.Constraint(model.stbg_index, rule=sigmag)
    else:
        model.penalty=pe.Expression(expr=0)

    def dual_obj(M):
        """Model objective
        """
        return pe.sum_product(M.N_g, M.pi) \
               + pe.sum_product(M.Demand, M.f) \
               + sum(M.D_T[j1, j2] * M.h[j1, j2, t] for (j1, j2, t) in M.VVT) \
               + Param.y_distance * Param.CostDistance \
               - M.penalty

    model.total_cost = pe.Objective(rule=dual_obj, sense=pe.maximize)

    #decision variables: the first constraints; contents will be added later
    model.plan_index = pe.Set(within=pe.PositiveIntegers, ordered=True)
    model.lbd = pe.ConstraintList()

    def mu(M, q, j, t):
        """Dual constraint for variable \mu

        Args:
            M:
            q: block section
            j: vessel
            t: time

        Returns:

        """
        return M.g[q, j, t] - (M.cost1 + M.cost2) * M.a[j, t] <= 0

    model.mu = pe.Constraint(model.KVTcycle, rule=mu)



    def eta1(M, j1, j2, t):
        """Dual constraint for variable \eta1

        Args:
            M:
            j1: vessel 1
            j2: vessel 2
            t: time

        Returns:

        """
        return -M.f[j2, t] + M.h[j1, j2, t] <= 0

    model.eta1 = pe.Constraint(model.VVT, rule=eta1)

    def eta2(M, j1, j2, t):
        return M.h[j1, j2, t] - 4 * (M.cost1 + M.cost2) * M.a[j1, t] <= 0

    model.eta2 = pe.Constraint(model.VVT, rule=eta2)

    def u0(M, j):
        """ Constraint: a vessel arrives within gamma time epoch after the schedule arrival time
        """
        A = M.Schedule[j]
        B = min(Param.Schedule[j] + M.gamma[j], len(M.T_cycle))
        return sum(M.a[j, tau] for tau in range(A, B + 1)) == 1

    model.u0 = pe.Constraint(model.V, rule=u0)

    def u1(M, j):
        """Constraint: a vessel only arrives once
        """
        return sum(M.a[j, t] for t in M.T_cycle) == 1

    model.u1 = pe.Constraint(model.V, rule=u1)  # eta

    return model



