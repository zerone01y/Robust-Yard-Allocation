import logging
from helper.data_types import stabilized_strategy, solu_col, sparsed_vars, DefaultDict
from pyomo.opt import SolverFactory
import numpy as np
from Formulations.master_problem_dual import master_problem_dual

logger = logging.getLogger("algorithm")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(module)s :%(message)s ')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


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

def CG_initialization(Param, Node=None, Num_Sol=2, CG_controller=False, solstate=False) -> tuple:
    """Initialize: generate initial columns for a better starting point
    """

    from Formulations.relaxed_primal_problem import relaxed_primal_prob
    from Formulations.Init_model import rough_cut_model

    time = 0
    # Use an estimation model to generate an initial template
    init_opt = SolverFactory("gurobi_direct")
    init_opt.options["TimeLimit"] = 30
    init_opt.options["PoolSearchMode"] = 2
    init_opt.options["PoolSolutions"] = Num_Sol
    model = rough_cut_model(Param)
    inst = model.create_instance(Param.data)
    time += init_opt.solve(inst).solver.Wallclock_time

    mapping = init_opt._pyomo_var_to_solver_var_map
    # Initialize: second stage evaluation;
    if not solstate:
        c_inst = relaxed_primal_prob(Param, firststage=False, relaxed=False).create_instance(Param.data)
        c_opt = SolverFactory("gurobi_direct")

    # Initialize container to store columns and counters
    # We store an index of column in _counter_ for easy control of the number of active columns (constraints)
    if CG_controller:
        cols = CG_controller.Cols  # Cols[k][P_k]: {(k, P_k): column data}
        counter = CG_controller.counter
    else:
        cols = []  # Cols[k][P_k]: {(k, P_k): column data}
        counter = []
    ub = np.inf
    best_temp = None

    for n in range(init_opt.results.Problem().Number_of_solutions):
        init_opt._solver_model.Params.SolutionNumber = n
        if solstate:
            z = tuple(int(round(sum(j * mapping[inst.z[i, j]].Xn for j in inst.V), 0)) for i in inst.I)
            state = solstate(z, generation="init")
            state.Check_Results()
        else:
            for i in c_inst.I:
                for j in c_inst.V:
                    c_inst.z[i, j] = mapping[inst.z[i, j]].Xn
            time += c_opt.solve(c_inst).solver.Wallclock_time
            break_mstr_sol_to_col(Param, c_inst, cols, counter)

            if ub > c_inst.total_cost():
                ub = min(ub, c_inst.total_cost())
                best_temp = c_inst.z.extract_values()

    n = init_opt._solver_model.SolCount
    logger.debug("%d columns are initialized." % (n + 1))
    if Node and Node.update_UB(ub):
        Node.update(best_temp=best_temp)
    if CG_controller:
        CG_controller.UB = ub
    if solstate:
        return state
    print("Initialization finished: %.2f seconds" % time)

    return (cols, counter)  # , dual_initializer

def break_mstr_sol_to_col(Param, inst, *args, **kwargs):
    Cols = args[0] if len(args) else kwargs.pop("Cols")
    counter = args[1] if len(args) else kwargs.pop("counter")
    z = args[2] if len(args) > 2 else kwargs.pop("z", None)

    if getattr(inst, "z", False):
        z = tuple(int(round(sum(inst.z[i, j] * j for j in inst.V)(), 0)) for i in inst.I)
        if len(z) != len(inst.I):
            raise Exception("Wrong template!")
    x = np.array(tuple(inst.x.get_values().values()), dtype=np.float16).reshape(Param.I, Param.V, Param.T)
    y = np.array(tuple(inst.y.get_values().values()), dtype=np.float16).reshape(Param.I, Param.V, Param.T)

    for bsg in Param.G_k.keys():
        for k in Param.G_k[bsg]:
            ks = Param.KS[k - 1]
            I_b = np.array(Param.I_b[ks] if type(ks) == int else Param.I_b[ks[0]] + Param.I_b[ks[1]])
            z_sub = tuple(z[i - 1] for i in I_b)
            sol = solu_col(
                X=x[I_b - 1, :, :Param.T_cycle].sum(axis=0),
                Y=y[I_b - 1].sum(axis=0),
                # Z = DefaultDict({j: sum(inst.z[i, j]() for i in I_b)
                #                     for j in inst.V})
                z=z_sub,
                C=round(inst.Cost_K[k](), 5),  # Cost_K: property of PRIMAL_R
                ks=bsg)

            index = str(bsg) + "_" + \
                    str(sol.z).replace("[", "").replace("]", "").replace(",", "").replace(" ", "") \
                    + "_"
            i = 0
            while index + str(i) in counter:
                i += 1
            Cols.append(sol)
            counter += [index + str(i)]
    return

def add_col_to_mstrdual(mstr_inst, counter, Cols ,Param):  # [0-f, 1-g, a, 3-pi]
    while len(mstr_inst.plan_index) < len(counter):
        i = len(mstr_inst.plan_index)
        mstr_inst.plan_index.add(i + 1)
        key = counter[i]
        plan = Cols[i]
        try:
            mstr_inst.lbd.add(
                mstr_inst.pi[plan.ks]
                # pi
                + sum(Param.Capacity*mstr_inst.rho[j] for j in plan.z)
                + sum(plan.X[j-1, t-1] * mstr_inst.f[j, t] for (j, t) in mstr_inst.VTcycle)
                - sum((
                plan.X[j-1, :].sum()
                - sum(plan.Y[j-1, t1-1] for t1 in range(t, t + mstr_inst.Loading_periods)))
                 * mstr_inst.g[plan.ks, j, t]
                  for (j, t) in mstr_inst.VTcycle) \
            <= plan.C + Param.CostDistance * sum(
                Param.distance[Param.I_k[Param.G_k[plan.ks][0]][0] + z1][z2 - 1] for (z1, z2) in enumerate(plan.z))
                )
        except:
            print("error, ", plan.ks, plan.z)
            counter.remove(key)
            mstr_inst.plan_index.remove(i+1)

class CG_Controller():
    Param = None
    """
    CG controller:
    storing:
        mstr_inst, mstr_opt as master problem and solver
        Cols, counter: master inst columns
            other param: cut_set, cut_groups, Stabilized_strategy(Rheta, etc)
        dual_info_calculate: update&solve master problem, update dual values
        bs_insts, bs_opt: pricing problems

    """

    def __init__(self, Param, QW=(0.8,0.2), **kwargs):
        """Initialize the column generation

        Initialize the parameter, including LB, UB, initial columns,
            set of active columns;
            Stabilized strategy:
                Stb_center_type: available option: ["robust", "smooth"]

        Build master problem and pricing problems from parameters.

        Param: Model parameters
        QW: weights
        Optional arguments:
            Cols:  initial master problem columns
            count:  index for columns
            Stabilized_strategy
        """

        self.cpu_time_cg = 0 #time counter
        CG_Controller.Param=Param
        self.UB=np.inf
        self.LB=0
        self.Cols = kwargs.pop("Cols", [])
        self.counter = kwargs.pop("kwargs", list())
        self.ActivePlans=0
        self.statistics = []
        self.QW=QW

        self.Stabilized_strategy = kwargs.pop("Stabilized_strategy",None)
        if self.Stabilized_strategy:
            self.Stb_Center_Type = kwargs.pop("Stb_Center_Type", "Robust")
            if self.Stb_Center_Type==None:
                self.Stb_Center_Type="Robust"
                print("Stb type = Robust")
            self.Runtype=self.Stb_Center_Type[0]+"S"
            self.mstr_inst = master_problem_dual(Param, **self.Stabilized_strategy.get("init_mstr_options", {})
                                                 ).create_instance(Param.data)
        else:
            self.mstr_inst = master_problem_dual(Param, penaltyweight=None).create_instance(
                Param.data)
            self.Stb_Center_Type = kwargs.pop("Stb_Center_Type", None)
            self.Runtype="NS"

            if self.Stb_Center_Type=="Robust":
                self.Stb_Center_Type="Smooth"

        self.mstr_opt = SolverFactory("gurobi_persistent")
        self.mstr_opt.set_instance(self.mstr_inst)
        self.mstr_opt.options["Method"] = 2
        self.Init_pricing_prob()

    def init_pricing_prob(self):
        """ Build pricing problems

        k: index of blocks section;
        ks: blocks indices
        """

        from Formulations.pricing_problem import pricing_prob

        pricing_solver_options = {"Method": 1,
                                  "PoolSearchMode": 1,
                                  "MIPGap": 0.2,
                                  "PoolSolutions": 8,
                                  "TimeLimit": 10}

        bs_insts = [pricing_prob(self.Param, k, alns=True).create_instance(self.Param.data) for k in
                    range(len(self.Param.G_k))]
        bs_opt = [SolverFactory('gurobi_persistent') for i in range(len(bs_insts))]
        for inst, opt in zip(bs_insts, bs_opt):
            opt.set_instance(inst)
        for k in range(len(bs_insts)):
            for item in pricing_solver_options.items():
                bs_opt[k].options[item[0]] = item[1]
        self.bs_insts = bs_insts
        self.bs_opt = bs_opt

    def init_stb_center(self):
        """Calculate robust dual point
        Update initial lowerbound and upperbound
        Initializae stabilized strategy

        Returns:

        """
        if self.Stb_Center_Type=="Robust":
            self.calculate_Robust_Dual_Point()  # create self.Rheta and LB
        self.QW_weight = 0.1
        if self.UB==np.inf:
            self.UB=self.LB*3

        # determine the initial stabilization center parameters
        self.Stabilized_strategy = self.Stabilized_strategy["Stabilized_strategy"]
        if self.Stabilized_strategy.name == "curvature":
            self.Stabilized_Strategy = stabilized_strategy(
                self.Stabilized_strategy[0], (self.UB-self.LB)/2, self.Stabilized_strategy[2])

    def dual_value_calculate(self):
        """
        Update master problem with new columns,
        Solve master problem,
        Update stb center and update the penalty values for master problem
        :return:
        """

        # if active plans exceeds a threshold: deactivate plans (columns)
        #   that has high slack value
        if self.ActivePlans >= self.Param.V * self.Param.I * 30:
            slack = np.array([(self.mstr_inst.lbd[i].get_suffix_value("slack", -np.inf)) for i in
                              range(1, 1 + len(self.mstr_inst.plan_index))])
            while self.ActivePlans >= self.Param.V * self.Param.I * 30:
                if slack[:int(0.8 * len(slack))].max() <= 100:
                    break
                popitem = np.where(slack == slack.max())
                for index in popitem[0]:
                    self.mstr_inst.lbd[index + 1].deactivate()
                    self.Cols[index] = 0
                    self.mstr_opt.remove_constraint(self.mstr_inst.lbd[index + 1])
                    self.ActivePlans -= 1
                    slack[index] = -np.inf
                    logger.info("Col %d deactived." % index)

        # add columns to master problem
        l1 = len(self.mstr_inst.lbd_index)
        add_col_to_mstrdual(self.mstr_inst, self.counter, self.Cols,
                            Param=self.Param)
        l2 = len(self.mstr_inst.lbd_index)
        for c in range(l1, l2):
            self.mstr_opt.add_constraint(self.mstr_inst.lbd[c + 1])
        self.ActivePlans += l2 - l1

        # solve master problem, and retrieve the dual values
        self.mstr_opt.solve()
        self.statistics.append(self.mstr_inst.total_cost() + self.mstr_inst.penalty())
        self.cpu_time_cg += self.mstr_opt.results.Solver().Wallclock_time
        self._dual_value = [
            np.array(list(self.mstr_inst.f.extract_values().values())).reshape(self.Param.V, self.Param.T_cycle),
            np.array(list(self.mstr_inst.g.extract_values().values())).reshape(len(self.Param.G_k), self.Param.V,
                                                                               self.Param.T_cycle),
            np.array(list(self.mstr_inst.h.extract_values().values())).reshape(self.Param.V, self.Param.V,
                                                                               self.Param.T_cycle),
            np.array(list(self.mstr_inst.pi.extract_values().values())),
            np.array(list(self.mstr_inst.rho.extract_values().values()))
            ]
        #Dual values: [f, g, h, pi, rho]

        return self._dual_value

    def mstr_value_cal(self):
        """
        Returns: the value of master problem value (calculated from dual values)
        Master problem values exclude the penalties.
        = terms associated with pi + Demand * f + distance + h

        """
        # costs involve transmission containers
        if self.Param.D_T_ind:
            dt=(np.array(list(self.Param.data.data()["D_T"].values())).reshape(self.Param.V,self.Param.V).transpose()
                *self._dual_value[2].sum(axis=2)).sum() #h * transmission containers
        else:
            dt=0
        return (self._dual_value[3]*np.array(tuple(len(self.Param.G_k[i]) for i in self.Param.G_k))).sum()\
        + (np.array(list(self.Param.data.data()["Demand"].values())
                    ).reshape(self.Param.T_cycle,self.Param.V
                              ).transpose()[:,:self.Param.T_cycle
           ]*self._dual_value[0]).sum() \
        + self.Param.y_distance * self.Param.CostDistance+dt

    def update_stabilized_param_for_dual_mstr(self, cancel=False,
                                              **kwargs):
        """ Update the stabilization-related parameter in master problem

        Args:
            cancel: set to True to stop stabilization.
            **kwargs:

        Returns:

        """
        if self.Stabilized_strategy is None:
            return
        # if the stabilization is canceled: set all weights to 0
        if cancel:
            self.Stabilized_strategy = None
            self.mstr_inst.penaltyweightf = 0
            self.mstr_inst.penaltyweightg = 0
            self.mstr_opt.set_objective(self.mstr_inst.total_cost)
            for c in self.mstr_inst.sigmaf.itervalues():
                self.mstr_opt.remove_constraint(c)
            for c in self.mstr_inst.sigmag.itervalues():
                self.mstr_opt.remove_constraint(c)
            return
        else:
            for (j, t) in self.mstr_inst.VTcycle:
                self.mstr_inst.scf[j, t] = self.Stb_Center[0][j - 1, t - 1]
            for (q, j, t) in self.mstr_inst.KVTcycle:
                self.mstr_inst.scg[q, j, t] = self.Stb_Center[1][q - 1, j - 1, t - 1]
        # remove all penalties for f and g that added in the previous iterature
        for c in self.mstr_inst.sigmaf.itervalues():
            self.mstr_opt.remove_constraint(c)
        for c in self.mstr_inst.sigmag.itervalues():
            self.mstr_opt.remove_constraint(c)
        # add new stabilization penalties - two optional strategies
        if self.Stabilized_strategy.name == "dynamic":
            self.mstr_inst.penaltyweightf = self.mstr_inst.penaltyweightf() / self.Stabilized_strategy.delta
            self.mstr_inst.penaltyweightg = self.mstr_inst.penaltyweightg() / self.Stabilized_strategy.delta
            pos_f = [i[1:] for i in sparsed_vars(self.mstr_inst.stbf).keys()]
            pos_g = [i[1:] for i in sparsed_vars(self.mstr_inst.stbg).keys()]
            for (j, t) in self.mstr_inst.VTcycle:
                #f
                if (j, t, "+") in pos_f or (j, t, "-") in pos_f:
                    self.mstr_inst.deltaf[j, t] = self.mstr_inst.deltaf[j, t].value * self.Stabilized_strategy.delta
                else:
                    self.mstr_inst.deltaf[j, t] = self.mstr_inst.deltaf[j, t].value / self.Stabilized_strategy.delta
            for (k, j, t) in self.mstr_inst.KVTcycle:
                #g
                if (k, j, t, "+") in pos_g or (k, j, t, "-") in pos_g:
                    self.mstr_inst.deltag[k, j, t] = self.mstr_inst.deltag[
                                                         k, j, t].value * self.Stabilized_strategy.delta
                else:
                    self.mstr_inst.deltag[k, j, t] = self.mstr_inst.deltag[
                                                         k, j, t].value / self.Stabilized_strategy.delta
        elif self.Stabilized_strategy.name == "curvature":
            for index in self.mstr_inst.VTcycle:
                self.mstr_inst.deltaf[index] = self.Stb_Center_diff[0].mean() / self.Stabilized_strategy.delta
            for index in self.mstr_inst.KVTcycle:
                self.mstr_inst.deltag[index] = self.Stb_Center_diff[1].mean() / self.Stabilized_strategy.delta
            self.mstr_inst.penaltyweightf = 4 * self.Stabilized_strategy.penalty * self.Stb_Center_diff[0].mean()
            self.mstr_inst.penaltyweightg = 4 * self.Stabilized_strategy.penalty * self.Stb_Center_diff[1].mean()
        logger.info("penaltyweight %.2f, %.2f" % (self.mstr_inst.penaltyweightf(),
                                                   self.mstr_inst.penaltyweightg()))
        # finally: add back constraints and reset objective functions
        for c in self.mstr_inst.sigmaf.itervalues():
            self.mstr_opt.add_constraint(c)
        for c in self.mstr_inst.sigmag.itervalues():
            self.mstr_opt.add_constraint(c)
        self.mstr_opt.set_objective(self.mstr_inst.total_cost)

    def update_stb_center(self):
        """Three options to update stb center: curvature; robust; smooth

        Returns:

        """
        if getattr(self, "Stb_Center", None):
            if self.Stabilized_strategy is not None:
                if self.Stabilized_strategy.name == "curvature":
                    self.Stb_Center_diff = [np.abs(i - newi) for (i, newi) in zip(self.Stb_Center, self._dual_value)]
            # the parameters: set Qwery weight, e.g. the weight for previous solution and current solution
            if self.Stb_Center_Type=="Robust" and self.Stabilized_strategy:
                Query_weight = max(min(self.QW[0],
                                   (self.mstr_inst.total_cost()+self.mstr_inst.penalty() - self.RLB
                                    ) / (self.RLB/100+1)),
                               self.QW[1])
                #warning: self.RLB can be 0
                logger.debug("Query_weight=%.2f" % Query_weight)

            # if stabilized strategy is not set
            elif self.Stb_Center_Type:
                Query_weight = 0.2
                logger.debug("Query_weight=%.2f" % Query_weight)

            # Stb center is a convex combination of RTheta, previous stb center, current dual vector
            # only for f, g, h
            if self.Stb_Center_Type == "Robust":
                self.Stb_Center = [ri * Query_weight + newi * (0.85 - Query_weight) + pi*0.15
                                       for (ri, newi, pi) in
                                       zip(self.RTheta[:3], self._dual_value[:3], self.Stb_Center[:3])+ self._dual_value[3:]
                                   ]
            elif self.Stb_Center_Type == "Smooth":
                self.Stb_Center = [i * Query_weight + newi * (1 - Query_weight)
                                       for (i, newi) in
                                       zip(self.Stb_Center[:3], self._dual_value[:3])] + self._dual_value[3:]

            else:
                self.Stb_Center = self._dual_value

            self.update_stabilized_param_for_dual_mstr()  # update penalty and stb_center for the next round
        elif self.Stabilized_strategy and self.Stb_Center_Type=="robust":
            self.calculate_Robust_Dual_Point()
        else:
            #Non-stabilized
            self.Stb_Center = self._dual_value

    def pass_dual_to_pricing_prob(self, *state):
        for ks in self.mstr_inst.G_k.__iter__():
            index = ks - 1
            self.bs_insts[index].pi = self._dual_value[3][index]
            for (j, t) in self.mstr_inst.VTcycle.__iter__():
                self.bs_insts[index].f[j, t] = self._dual_value[0][j - 1, t - 1]
                self.bs_insts[index].g[j, t] = self._dual_value[1][index, j - 1, t - 1]
            for j in self.mstr_inst.V.__iter__():
                self.bs_insts[index].rho[j]=self._dual_value[4][j-1]

    def solve_pricing_prob(self, index):
        bs_inst = self.bs_insts[index]
        bs_opt = self.bs_opt[index]
        bs_opt.set_objective(bs_inst.section_cost)
        result = bs_opt.solve()
        self.cpu_time_cg+=result.Solver().Wallclock_time
        z = sparsed_vars(bs_inst.z).template_output
        if bs_opt.results.Solver.Status in [ "warning"]:
            print(bs_opt.results.Solver())
            return False, False
        else:
            for sol in range(bs_opt._solver_model.Solcount):
                bs_opt._solver_model.Params.SolutionNumber=sol
                if bs_opt._solver_model.PoolObjVal<0:
                    add_pricing_sol_to_col(self.Param, bs_opt, Cols=self.Cols, counter=self.counter)
        return z, result.Solver().Wallclock_time

    @property
    def dual_value(self):
        if getattr(self, "_dual_value", False):
            return self._dual_value
        else:
            return self.dual_value_calculate()

    def calculate_Robust_Dual_Point(self):
        """RTheta is a set of vectors of super optimal dual points.
        Solve the robust dual master problem to otain the vectors
        """
        from Formulations.RDP import robust_dual_points
        robust_dmp = robust_dual_points(self.Param).create_instance(self.Param.data)
        mstr_opt = SolverFactory("gurobi_direct")

        mstr_opt.options["Method"] = 2
        LB_result = mstr_opt.solve(robust_dmp)
        self.cpu_time_cg += LB_result.Solver().Wallclock_time
        self.RTheta = (np.array(list(robust_dmp.f.extract_values().values())).reshape(self.Param.V, self.Param.T_cycle),
                       np.array(list(robust_dmp.g.extract_values().values())).reshape(len(self.Param.G_k), self.Param.V,
                                                                                      self.Param.T_cycle),
                       np.array(list(robust_dmp.h.extract_values().values())).reshape(self.Param.V, self.Param.V, self.Param.T_cycle),
                       np.array(list(pi() for pi in robust_dmp.pi.extract_values().values())),
                       np.array(list(robust_dmp.rho.extract_values().values()))
                       )
        self.Stb_Center = self.RTheta
        self.RLB = LB_result.Problem().Upper_bound
        self.LB=self.RLB
        del mstr_opt, robust_dmp
