import numpy as np
from CG_controller import CG_Controller, logger, CG_initialization
from helper.data_types import Parameters
from helper.data_types import stabilized_strategy
import argparse
from pandas import DataFrame

SEED=5432
rnd_state=np.random.RandomState(SEED)

parser = argparse.ArgumentParser(description="CG options")
parser.add_argument('Instance', default="data_csv")
parser.add_argument('Demand', default="6")
parser.add_argument('--solver', default="CG")
parser.add_argument("--Cd", "-c", nargs="?", type=float, const=0, default=5e-3)
parser.add_argument("--Trans", "-t", nargs="?", const=True, default=False)
parser.add_argument("--MaxIter", "-M", nargs="?", type=int, default=200)
parser.add_argument("--gap","-G", nargs="?",type=float, const=0.005, default=0.05)
parser.add_argument("--timelimit", "-T", nargs="?", type=int, default=2400)
parser.add_argument("--Stabilization", "-S", nargs="?", type=bool, const=True, default=False)
parser.add_argument("--SmoothingType", "-s", nargs="?", default=None)





def report_status(t, counter, total_reduced_cost=-np.inf):
    """

    Args:
        t:
        counter:
        total_reduced_cost:

    Returns:

    """
    if t>0:
        logger.info("Iteration %d: LB for current nodes is %.2f. Master problem yields %.2f. " % (t, CG.LB, CG.statistics[-1]))
        pass
    if t>1 and abs(CG.statistics[-2]-CG.statistics[-1])<0.1:
        counter+=1
        logger.debug("The master problem does not improve for %d iterations," % counter)
    else:
        counter=0
    if counter >  MINIP:
        logger.info("Iteration %d: Max no-improve iterations reached."%t)
        t=MaxIter+1
    if (t>=3 and CG.statistics[-1]\
            <= CG.LB * (1.0005 + GAP) + 0.01 ) or t>MaxIter/2\
            or (total_reduced_cost>=-0.01 and t>=3) :
        # stop stabilized CG if reaching a threshold
        if CG.Stabilized_strategy and CG.mstr_inst.penaltyweightf() + CG.mstr_inst.penaltyweightg() < 5:
            logger.debug("Stop stabilized CG")
            CG.update_stabilized_param_for_dual_mstr(cancel=True)
            CG.Stb_Center_Type=None
            CG.Stop_point=t
    return t+1,counter

def run(Param, CG):

    CG_initialization(Param, None, Num_Sol=3)

    if CG.Stabilized_strategy:
        CG.init_stb_center()

    Param.KS_sequence = list(range(len(Param.KS)))
    t,counter=report_status(0,0,-np.inf)
    Convergence = DataFrame(
        columns=['Mstr_result', "LB", "total_time"])

    CG.dual_value_calculate()
    CG.update_stb_center()

    while t<=MaxIter:
        CG.pass_dual_to_pricing_prob()

        total_reduced_cost=0
        for q_index in Param.G_k:
            # if we have enough time ...
            if TimeLimit-CG.cpu_time_cg <= CG.bs_opt[q_index-1].options.TimeLimit:
                # change time limit, if not enough time
                if TimeLimit-CG.cpu_time_cg>=0.2:
                    CG.bs_opt[q_index-1].options["TimeLimit"]=TimeLimit-CG.cpu_time_cg
                else:
                    logger.info("Termination - Exceed time limit.")
                    print(Convergence)
                    CG.Convergence = Convergence
                    return

            CG.pricing_solver(q_index - 1)
            total_reduced_cost += min(CG.bs_opt[q_index-1].results.problem.Lower_bound,0)*len(Param.G_k[q_index])

        CG.LB= CG.mstr_value_cal() + total_reduced_cost
        CG.dual_value_calculate()
        Convergence.loc[t] = [CG.mstr_inst.total_cost() + CG.mstr_inst.penalty(), CG.LB, CG.cpu_time_cg]

        if round(total_reduced_cost, 2) <= -0.005 or CG.Stb_Center_Type or t<=3:
            t, counter=report_status(t, counter, total_reduced_cost)
            CG.update_stb_center()
        else:
            report_status(t, counter)
            logger.info("No positive reduced cost, terminate")
            t=MaxIter+1

    print(Convergence)
    CG.Convergence=Convergence
    return

if __name__ == '__main__':

    args = parser.parse_args()

    print(args)

    MINIP = 30
    TimeLimit = args.timelimit
    MaxIter = args.MaxIter
    GAP = args.gap

    instance = args.Instance
    Param = Parameters(instance) #Prepare parameters
    Param.Demand_Generation(args.Demand, D_T=args.Trans) #generate demand from demand info (Outbound, transship)

    print("# ",Param._name, Param.Utilization,str(int(args.Cd*1000)), str(Param.D_T_ind)[:1])
    Param.display()
    print(args)

    if args.Stabilization:
        Stabilized_strategy = {"init_mstr_options":
                                   {"delta": 0.001,
                                    "penaltyweight": 50
                                    },
                               "Stabilized_strategy":
                                   stabilized_strategy("curvature", 50, 1.5)
                               }
    else:
        Stabilized_strategy=None

    CG = CG_Controller(Param, QW=(0.3,0), Stabilized_strategy=Stabilized_strategy,
                       Stb_Center_Type=args.SmoothingType
                       )
    print(Param._name,CG.Runtype)
    run(Param,CG)
