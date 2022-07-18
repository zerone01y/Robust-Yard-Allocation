import collections
import numpy as np
import json

solu_col = collections.namedtuple("AssignmentPlan",
                                  ["X", "Y", "z", "C", "ks"])

pricingSolverOptions = collections.namedtuple("Pricing_options",
                                              ["Method", "Cutoff", "Threads",
                                               "PoolSearchMode", "MIPGap",
                                               "PoolSolutions", "TimeLimit"])

stabilizedStrategy = collections.namedtuple("Stabilized_Strategy",
                                            ["name", "penalty", "delta"])

cut_element = collections.namedtuple("cut", ["q", "j", "bound", "sign"])
cut_key = collections.namedtuple("cut", ["q", "j", "sign"])
# (right: >=, sign="+", plus; left: <=, sign="-")

class Node_data():
    UB_info = (np.inf, 0)
    Tree = None

    @classmethod
    def update_uniattr(cls, **kwargs):
        for i in kwargs:
            setattr(cls, i, kwargs[i])

    def __init__(self, UB=float("inf"), **kwargs):
        self.Z = None
        self.LB = kwargs.pop("LB", 0)
        self.Cols = kwargs.pop("Cols", None)
        self.counter = kwargs.pop("kwargs", None)
        self.cut = kwargs.pop("cut", (dict(), dict()))
        self.label = kwargs.pop("label", "Active")
        self.Node_id = kwargs.pop("Node_id", None)
        Node_data.Tree = kwargs.pop("Tree", Node_data.Tree)
        self.UB = UB
        for item in kwargs.items():
            self.__setattr__(item[0], item[1])

    def update_UB(self, UB):
        if UB < self.UB:
            self.UB = UB
            if UB < Node_data.UB_info[0]:
                Node_data.UB_info = (UB, self.Node_id)
                if Node_data.Tree is not None:
                    for Node in getattr(Node_data.Tree, "leaves")():
                        if Node.data.LB >= UB:
                            Node.data.prune("bound")
            return True
        else:
            return False

    def update(self, **kwargs):
        self.update_UB(kwargs.pop("UB", float("inf")))
        if "bs_insts" in kwargs:
            Node_data.bs_insts = kwargs.pop("bs_insts")
        for i in kwargs:
            self.__setattr__(i, kwargs[i])
        self.prune()

    def branch(self, k, j, value, direction):
        self.label = "Branched"
        self.child = direction
        self.branch_with = (k, j, value, value)
        # Branching

    def prune(self, *args):
        if "Pruned" in self.label:
            return
        if getattr(self, "int_sol", False) == True:
            self.label = "Pruned/optimality"
        elif len(args):
            self.label = "Pruned/" + str(args[0])
        elif round(self.LB, 2) > round(self.UB):
            self.label = "Pruned/bound"
        if "Pruned" in self.label:
            print(self.label)
            if "optimality" not in self.label:
                self.clear()

    def clear(self):
        for attr in list(vars(self)):
            if attr not in ["LB", "label", "cut", "UB", "best_temp", "Node_id"]:
                delattr(self, attr)

    def display(self):
        for i in vars(self).items():
            if i[0] == "Cols":
                print(i[0], ":\t", len(i[1]))
            else:
                print(i[0], ":\t", i[1])

    @property
    def info(self):
        if len(self.cut[0]):
            return getattr(self, 'Node_id', '') + ' ' + str(list(self.cut[0].items())[-1]) + ' ' + str(
                round(self.LB, 2))
        else:
            return getattr(self, 'Node_id', '') + ' ' + str(round(self.LB, 2))

    @property
    def Upperbound(self):
        ub = [self.__getattribute__(i) for i in vars(self) if "ub" in i and self.__getattribute__(i)]
        if len(ub) > 0:
            return min(ub)
        else:
            return None

class DefaultDict(dict):
    _default = 0

    @property
    def default(self):
        return self._default

    def set_default(self, default):
        self._default = default
        return self

    def __missing__(self, key):
        return self.default

    def __add__(self, other):
        if isinstance(other, dict):
            for i in self.keys() | other.keys():
                if other.get(i):
                    self[i] += other.get(i)
            return self
        elif isinstance(other, (int, float)):
            for i in self.keys():
                self[i] += other
            return self

    def sparsed(self, filtervalue=1e-5):
        return DefaultDict({i: self[i] for i in self if self[i] > filtervalue})

    # dep
    @property
    def template_output(self):
        for i in self.values():
            if round(i, 1) not in [0, 1]:
                raise ValueError()
        z = []
        for i in sorted(self.sparsed()):
            z.append(i[1])
        return tuple(z)

    def template_str(self):
        z = ""
        for i in self.values():
            for i in sorted(self.sparsed()):
                z += str(i[1])
        return z

    # summed = reduce(lambda x, y: x + y, [x, y, z])
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_col = DefaultDict()
            for i in self:
                new_col[i] = other * self[i]
            return new_col

class sub_template(DefaultDict):
    @property
    def template_output(self):
        for i in self.values():
            if round(i, 1) not in [0, 1]:
                raise ValueError()
        z = []
        for i in sorted(self.sparsed()):
            z.append(i[1])
        return tuple(z)

