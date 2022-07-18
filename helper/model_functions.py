
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



def ind_SubTemp(subtemp, st, number, vessel):
    if subtemp[st][number] == str(vessel):
        return 1
    else:
        return 0


# Validation: vessel can only arrive within the cycle time
def Demand_validate(M, d, j, t):
    if not t in M.T_cycle:
        return d == 0
    else:
        return True


def Schedule_validate(M, v, j):
    return v <= len(M.T_cycle)


def Update_Q_k(self):
    for k in self.KS_I:
        for q in self.Q_I:
            if not any(v in q
                       for v in self.forbidden_vessels.get(k, (0,))):
                self.Q_I_k[k].add(q)
    for k in self.KS_N:
        for q in self.Q_N:
            if not any(v in q
                       for v in self.forbidden_vessels.get(k, (0,))):
                self.Q_N_k[k].add(q)


def Print_workload(M):
    for i in M.I:
        for j in M.V:
            if M.z[i, j] == 1:
                for tau in M.T:
                    if tau in M.T_cycle:
                        message = "(%d %d %d)\t %.2f\t %.2f\t %.2f\t %.2f" % (
                            i, j, tau, round(M.x[i, j, tau](), 2), round(M.y[i, j, tau](), 2),
                            round((-sum(M.x[i, j, t] for t in M.T) + sum(M.y[i, j, t]
                                                                         for t in
                                                                         [tau, tau + M.Loading_periods - 1]))(),
                                  2), M.mu[i, j, tau]())
                    else:
                        message = "(%d %d %d)\t %.2f\t %.2f" % (
                            i, j, tau, round(M.x[i, j, tau](), 2), round(M.y[i, j, tau](), 2))
                    print(message)

def sparsed_vars(*args, **kwargs):
    from helper.Data_structure import DefaultDict

    value_filter = kwargs.pop("filter", lambda v: v)

    if len(args) >= 2:
        inst = args[0]
        vars = args[1:]

        lbd = {}
        for var in vars:
            lbd[var] = inst.component(var).extract_values()
            lbd[var] = DefaultDict({i: j for (i, j) in lbd[var].items() if value_filter(j)})
        if len(vars) == 1:
            return lbd[vars[0]]
    elif len(args) == 1:
        import pyomo.core.base as pcb
        var = args[0]
        if isinstance(var, pcb.var.Var):
            lbd = DefaultDict({i: j for (i, j) in var.extract_values().items() if value_filter(j)})
        elif isinstance(var, pcb.param.Param):
            lbd = DefaultDict(var.extract_values_sparse())
        else:
            raise Exception("Wrong value type for sparsed_vars")
    return lbd


def add_pricing_sol_to_col(Param, inst, *args, **kwargs):
    import pyomo.core as pc
    Cols = args[0] if len(args) else kwargs.pop("Cols", [])
    counter = args[1] if len(args) else kwargs.pop("counter", dict())
    X, Y = DefaultDict(), DefaultDict()
    if isinstance(inst, pc.ConcreteModel):
        # logging.debug("Receive an instance. Add columns to the set")
        for j in inst.V:
            for t in inst.T:
                X[j, t] = round(sum(inst.x[i, j, t].value for i in inst.I), 5)
                Y[j, t] = round(sum(inst.y[i, j, t].value for i in inst.I), 5)
            # sol.Z[j] = round(sum(inst.z[i, j].value for i in inst.I), 5)
        sol = solu_col(X=X, Y=Y,
                       z=[i[1] for i in sparsed_vars(inst.z, filter=lambda v: v > 0.5)],
                       ks=inst.k,
                       C=round(inst.cost1 * pe.summation(inst.vio1)() + inst.cost2 * pe.summation(inst.vio2)(), 5),
                       )
        z_str = str(sol.z).replace("[", "").replace("]", "").replace(",", "").replace(" ", "")
        index = str(sol.ks) + "_" + z_str + "_"
        i = 0

        if 0 in sol.z:
            print("debug")
        while index + str(i) in counter[inst.k]:
            i += 1
        Cols.append(sol)
        counter += [index + str(i)]
    else:
        # logging.debug("Receive an SolverFactory. Add columns to the set")
        p_inst = inst._pyomo_model
        mapping = inst._pyomo_var_to_solver_var_map
        sol = solu_col(X=np.array(
            tuple(mapping[p_inst.x[i, j, t]].Xn for i in p_inst.I for j in p_inst.V for t in p_inst.T_cycle)).reshape(
            len(p_inst.I), Param.V, Param.T_cycle).sum(axis=0, dtype=np.float32),
                       Y=np.array(tuple(mapping[p_inst.y[i, j, t]].Xn for i in p_inst.I for j in p_inst.V for t in
                                        p_inst.T)).reshape(len(p_inst.I), Param.V, Param.T).sum(axis=0,
                                                                                                dtype=np.float32),
                       z=tuple(
                           int(round((sum(mapping[p_inst.z[i, j]].Xn * j for j in p_inst.V)), 0)) for i in p_inst.I),
                       C=round(
                           sum(Param.cost1 * sum(mapping[p_inst.vio1[b, t]].Xn for b in p_inst.B) + Param.cost2 * sum(
                               mapping[p_inst.vio2[i, t]].Xn for i in p_inst.I) for t in p_inst.T), 5),
                       ks=p_inst.k)

        index = str(sol.ks) + "_" + \
                str(sol.z).replace("[", "").replace("]", "").replace(",", "").replace(" ", "") \
                + "_"
        i = 0
        while index + str(i) in counter:
            i += 1
        Cols.append(sol)
        counter += [index + str(i)]


def Initialize_stability_center(c_inst, Param):  # stability center initialization
    a, f = [np.zeros((c_inst.V._len, c_inst.T_cycle._len)) for i in range(2)]
    g = np.zeros((len(Param.G_k), c_inst.V._len, c_inst.T_cycle._len))
    pi = np.zeros((len(Param.G_k)))

    for j in c_inst.V:
        for t in c_inst.T_cycle:
            a[j - 1, t - 1] = c_inst.a[j, t].get_suffix_value("dual")
            f[j - 1, t - 1] = c_inst.f[j, t].get_suffix_value("dual")

    for q in Param.G_k:
        I_q = [i for k in Param.G_k[q] for i in Param.I_k[k]]
        for j in c_inst.V:
            for t in c_inst.T_cycle:
                i = max(I_q, key=lambda sb: c_inst.g[sb, j, t].get_suffix_value("dual"))
                g[q - 1, j - 1, t - 1] = c_inst.g[i, j, t].get_suffix_value("dual")
        pi_k = 0
        for i in I_q:
            pi_k -= sum(c_inst.z[i, j]() * c_inst.e[i, j].get_suffix_value("dual") for j in c_inst.V) * Param.Capacity
            pi_k += sum(c_inst.beta[i, t].get_suffix_value("dual") for t in c_inst.T) * c_inst.M[i]
        pi_k += sum(
            sum(c_inst.alpha[b, t].get_suffix_value("dual") for t in c_inst.T) * c_inst.YC[b]
            for k in Param.G_k[q] for b in Param.B_k[k])
        pi[q - 1] = pi_k / len(Param.G_k[q])
    return [f, g, a, pi]
