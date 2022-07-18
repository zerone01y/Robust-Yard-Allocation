# Robust-Yard-Allocation

For the mathematical formulations, please refer to [A robust approximation for yard template optimization under uncertainty](https://doi.org/10.1016/j.trb.2022.03.005).

Five formulations are coded in pyomo, and the correspondance between the formulations and the mathematical version in the manuscript is:

- relaxed primal problem: [M3] primal RAYTOP problem

- master problem dual: dual of [M5], and stabilization-related constraints are also included.

- pricing problem: [M6]

- Init model: A simple model that provides initial templates for the column generation process

- RDP: [M7] to obtain Robust dual points.

Detailed description of these models can be found in the manuscript.

The models are coded in pyomo, so the python built-in data types can be used, and they can be solved by multiple solvers supported by the module. It is suggested to use collections.namedtuples() for the parameters.

Note that the dual version of the master problem is used, because column generation is equivalent to cutting plane for the dual problem, and adding constraints in the fly would be easier for pyomo/gurobi.

# Column generation

YAP\_CG folder contains column generation algorithm with stabilization.
