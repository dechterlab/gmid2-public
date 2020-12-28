"""

Factor class and necessary functions

"""
# from __future__ import annotations
import numpy as np
from sortedcontainers import SortedSet
from collections import namedtuple
from typing import List, Tuple, Union, overload, Optional, Callable, Iterable
from .helper import reduce_tuples
import functools

Variable = namedtuple('Variable', ['vid', 'dim', 'type'])

Assignment = namedtuple('Assignment', ['vid', 'val'])


# def reduce_tuples(input_list: List, pos: int=0)-> List:
#     return [el[pos] for el in input_list]


class Factor:
    fid = 0

    def __init__(self, scope: List[Variable]=None, table: Union[List[float], float]=None, factor_type: str= ''):
        self.fid = Factor.fid         # later maintain this fid in a SortedSet
        Factor.fid += 1
        # self.logscale = log_scale
        self.type = factor_type
        self.scope_vars = SortedSet(scope) if scope else SortedSet()
        self.scope_vids = SortedSet(reduce_tuples(scope, pos=0)) if scope else SortedSet()
        if not scope:
            self.table = np.array(table, dtype=np.float64)
        else:
            table_size = functools.reduce(lambda x, y: x * y, (v.dim for v in self.scope_vars))
            if table is None:
                self.table = np.array([1.0] * table_size, dtype=np.float64)
            elif isinstance(table, float) or isinstance(table, int):
                self.table = np.array([table] * table_size, dtype=np.float64)
            elif table_size == len(table):
                self.table = np.array(table, dtype=np.float64)
            else:
                assert False, "Input table size is not compatible to the scope of function"
            self.table = np.array(
                np.array(self.table, dtype=np.float64).reshape([v.dim for v in scope]).transpose(np.argsort(tuple(reduce_tuples(scope, 0))))
            )
            # if table is None:
            #     self.table = np.array(0.0, dtype=np.float64)        # if one skipped the table only to build gm
            # else:
            #     self.table = np.array(
            #         np.array(table, dtype=np.float64).reshape([v.dim for v in scope]).transpose(np.argsort(tuple(reduce_tuples(scope, 0))))
            #     )

    @classmethod
    def reset_fid(cls):
        cls.fid = 0

    def build(self, new_sorted_scope: SortedSet, new_sorted_scope_vids: SortedSet, new_table: np.array):
        """
        build a Factor object directly from newly created sorted scope and numpy table
        """
        f = Factor()
        f.scope_vars = new_sorted_scope
        # f.scope_vids = SortedSet(reduce_tuples(f.scope_vars))
        f.scope_vids = new_sorted_scope_vids
        f.table = new_table
        f.type = self.type      # this is a string 'P', 'U'
        return f

    @property
    def scope(self) -> SortedSet:
        return self.scope_vars

    @scope.setter
    def scope(self, new_scope: SortedSet) -> None:
    # def scope(self, new_scope: Iterable[Variable]) -> None:
        self.scope_vars = new_scope
        self.scope_vids = SortedSet(reduce_tuples(new_scope, pos=0))

    @property
    def nvar(self) -> int:
        return len(self.scope)

    @property
    def size(self) -> int:
        return self.table.size

    # @property
    # def vars(self) -> SortedSet:
    #     return self.scope_vars
    #
    # @property
    # def vids(self) -> SortedSet:
    #     return self.scope_vids

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.table.shape

    # apply operator to table of factor
    def _apply(self, op: Callable, inplace: bool=True):
        if inplace:
            op(self.table, out=self.table)
            return
        else:
            new_table = op(self.table)
            return self.build(self.scope.copy(), self.scope_vids.copy(), new_table)

    def __abs__(self):
        return self._apply(np.fabs, inplace=False)

    abs = __abs__

    def __neg__(self):
        return self._apply(np.negative, inplace=False)

    def __pow__(self, power: float):       # **
        new_table = np.power(self.table, power)
        return self.build(self.scope.copy(), self.scope_vids.copy(), new_table)

    power = __pow__

    def exp(self):
        return self._apply(np.exp, inplace=False)

    def log(self):
        return self._apply(np.log, inplace=False)

    def log2(self):
        return self._apply(np.log2, inplace=False)

    def log10(self):
        return self._apply(np.log10, inplace=False)

    # to form a chain f.op().op().op() need to return self even if it is inplace; fixme return self?
    def iabs(self) -> None:
        self._apply(np.fabs, inplace=True)

    def ineg(self) -> None:
        self._apply(np.negative, inplace=True)

    def ipow(self, power: float) -> None:  # **
        np.power(self.table, power, out=self.table)

    def iexp(self) -> None:
        self._apply(np.exp, inplace=True)

    def ilog(self) -> None:
        self._apply(np.log, inplace=True)

    def ilog2(self) -> None:
        self._apply(np.log2, inplace=True)

    def ilog10(self) -> None:
        self._apply(np.log10, inplace=True)

    # combine with other factor and return a new factor object
    def _combine(self, other, op: Callable, inplace: bool=False):
        if not isinstance(other, Factor):               # scalar value
            if inplace:
                op(self.table, other, out=self.table)
                return self
            else:
                out = np.array(1.0) if len(self.scope) == 0 else None       # add this line for scalar conversion
                new_table = op(self.table, other, out=out)
                return self.build(self.scope.copy(), self.scope_vids.copy(), new_table)

        if self.scope_vids == other.scope:      # same scope for both functions no expandsion
            t1 = self.table
            t2 = other.table
            if inplace:
                op(t1, t2, out=t1)  # store result in the table of self, t1
                return self
            else:
                new_table = op(t1, t2)
                return self.build(self.scope.copy(), self.scope_vids.copy(), new_table)
        else:
            new_scope = self.scope | other.scope
            t1 = np.array(np.expand_dims(self.table, axis=[i for i, v in enumerate(new_scope) if v not in self.scope]))
            t2 = np.array(
                np.expand_dims(other.table, axis=[i for i, v in enumerate(new_scope) if v not in other.scope]))
            if inplace:
                self.table = op(t1, t2)
                self.scope = new_scope  # scope is expanded, also change scope_vids
                return self
            else:
                new_table = op(t1, t2)
                new_scope_vids = SortedSet(reduce_tuples(new_scope, pos=0))
                return self.build(new_scope, new_scope_vids, new_table)

    def __add__(self, other):
        return self._combine(other, op=np.add)

    def __sub__(self, other):
        return self._combine(other, op=np.subtract)

    def __mul__(self, other):
        return self._combine(other, op=np.multiply)

    def __truediv__(self, other):
        return self._combine(other, op=np.divide)

    def __radd__(self, other):
        return self._combine(other, op=np.add)

    def __rsub__(self, other):
        if not isinstance(other, Factor):
            return self.__neg__()._combine(other, op=np.add)        # 1 -f == -f + 1
        else:
            return self._combine(other, op=np.subtract)

    def __rmul__(self, other):
        return self._combine(other, op=np.multiply)

    def __rtruediv__(self, other):
        if not isinstance(other, Factor):
            f = Factor([], table = other, factor_type=self.type)  # log_scale=self.logscale)
            return f._combine(self, op=np.divide)
        else:
            return other._combine(self, op=np.divide)

    def __iadd__(self, other) :
        return self._combine(other, op=np.add, inplace=True)

    def __isub__(self, other) :
        return self._combine(other, op=np.subtract, inplace=True)

    def __imul__(self, other) :
        return self._combine(other, op=np.multiply, inplace=True)

    def __itruediv__(self, other) :
        return self._combine(other, op=np.divide, inplace=True)

    # marginalize varaibles by operators, max, min, sum
    def _marginal(self, eliminator: SortedSet, op: Callable, inplace: bool=True):
        if len(eliminator) == len(self.scope_vars):     # marginalzie all vars to float
            return op(self.table)       # this is not Factor object but float
        else:
            axis = tuple(i for i, v in enumerate(self.scope) if v in eliminator)
            if inplace:
                self.table = op(self.table, axis=axis)
                # self.scope = self.scope - eliminator        # change this discard element
                self.scope_vars.difference_update(eliminator)
                self.scope_vids.difference_update(reduce_tuples(eliminator, pos=0))
                return self
            else:
                new_table = op(self.table, axis=axis)
                new_scope = self.scope - eliminator         # fixme change this avoid unnecessary creations??
                new_scope_vids = SortedSet(reduce_tuples(new_scope, pos=0))
                return self.build(new_scope, new_scope_vids, new_table)

    # def _marginal_old(self, eliminator: SortedSet, op: Callable, inplace: bool = True) -> Optional[Factor]:
    #     # add below line for scalar conversion
    #     out = np.array(1.0) if len(eliminator) == len(self.scope_vars) else None
    #     axis = tuple(i for i, v in enumerate(self.scope) if v in eliminator)
    #     if inplace:
    #         self.table = op(self.table, axis=axis, out=out)
    #         self.scope = self.scope - eliminator
    #         return self
    #     else:
    #         new_table = op(self.table, axis=axis, out=out)
    #         new_scope = self.scope - eliminator
    #         new_scope_vids = SortedSet(reduce_tuples(new_scope, pos=0))
    #         return self.build(new_scope, new_scope_vids, new_table)

    def max_marginal(self, eliminator: SortedSet, inplace: bool=True):
        return self._marginal(eliminator, np.amax, inplace)

    def min_marginal(self, eliminator: SortedSet, inplace: bool=True):
        return self._marginal(eliminator, np.amin, inplace)

    def sum_marginal(self, eliminator: SortedSet, inplace: bool=True):
        return self._marginal(eliminator, np.sum, inplace)

    # def lse_marginal_old(self, eliminator: SortedSet, inplace: bool =True) -> Optional[Factor]:
    #
    #     """
    #     log sum_{eliminator} exp ( F(X) )
    #     """
    #     if inplace:
    #         self.iexp()
    #         self.sum_marginal(eliminator, inplace=True)
    #         self.ilog()
    #         return self
    #     else:
    #         f = self.exp()
    #         f.sum_marginal(eliminator, inplace=True)
    #         f.ilog()
    #         return f

    def lse_marginal(self, eliminator: SortedSet, inplace: bool =True):
        # op = np.logaddexp.reduce
        # if len(eliminator) == len(self.scope_vars):
        #     out = np.array(1.0)
        #     new_table = op(np.ravel(self.table), out=out)       # send output to out
        #     new_scope = SortedSet([])
        #     if inplace:
        #         self.table = new_table
        #         self.scope = new_scope      # empty set
        #         return self
        #     else:
        #         return self.build(new_scope, new_table)
        # else:
        #     axis = tuple(i for i, v in enumerate(self.scope) if v in eliminator)
        #     src = self.table
        #     while len(axis) > 1:
        #         src =op(src, axis=axis[-1])
        #         axis = axis[:-1]
        #     new_table = op(src, axis=axis)
        #     new_scope = self.scope - eliminator
        #     if inplace:
        #         self.table = new_table
        #         self.scope = new_scope
        #         return self
        #     else:
        #         return self.build(new_scope, new_table)
        return self._marginal(eliminator, np.logaddexp.reduce, inplace)         # follow common interface; return scalar

    # def lse_pnorm_marginal_old(self, eliminator: SortedSet, p, inplace:bool =True) -> Optional[Factor]:
    #     """
    #     1/p * log sum_{eliminator} [ exp( F(x) ) ]^p
    #     log of Lp norm of exp( F(X) )
    #     """
    #     if inplace:
    #         self.iexp()
    #         self.ipow(p)
    #         self.sum_marginal(eliminator, inplace=True)
    #         self.ilog()
    #         self.__imul__(1.0/p)
    #         return self
    #     else:
    #         f = self.exp()
    #         f.ipow(p)
    #         f.sum_marginal(eliminator, inplace=True)
    #         f.ilog()
    #         f.__imul__(1.0 / p)
    #         return f

    def lse_pnorm_marginal(self, eliminator: SortedSet, p, inplace:bool =True) :
        if p == np.inf:
            return self.max_marginal(eliminator, inplace)
        elif p == -np.inf:
            return self.min_marginal(eliminator, inplace)
        elif p == 1.0:
            return self.lse_marginal(eliminator, inplace)
        else:
            if inplace:
                # self *= p               # this line is in-place operation
                self.__imul__(p)          # not showing error same as *= p (float)
                # self.lse_pnorm_marginal(eliminator, inplace)      # this was bug!
                # self *= (1.0/p)
                # return self
                f = self.lse_marginal(eliminator, inplace=True)     # self was Factor, result may not factor so put it f
                f *= (1.0/p)                                        # if f is factor self and f is identical object
                return f
            else:
                f = self * p                # this line creates a new Factor f from self * p
                f = f.lse_marginal(eliminator, inplace=True)  # create new obj, inplace lse_marginal returns self
                f *= (1.0/p)
                return f

    def __str__(self) -> str:
        return "{}{}({}):={}".format(type(self).__name__, self.fid,
                                     ",".join([str(var.vid)+'[' + str(var.dim) + ']' for var in self.scope]),
                                   str(self.table.ravel()))

    def __repr__(self) -> str:
        return "{}{}({})".format(type(self).__name__, self.fid, ",".join([str(var) for var in self.scope_vids]))

    def __len__(self) -> int:
        return self.size

    def __float__(self) -> Optional[float]:
        if self.nvar == 0:
            return float(self.table)


    # @overload
    def __contains__(self, var: Union[Variable, int]) -> bool:
        """
        return True if var is in the scope of the function

        :param var: Variable
        :return: bool
        """
        if isinstance(var, Variable):
            return var in self.scope_vars
        return var in self.scope_vids

    # def __contains__(self, var: int) -> bool:
    #     """
    #     return True if var is in the scope of the function
    #
    #     :param var: variable id
    #     :return: bool
    #     """
    #     return var in self.scope_vars

    # @overload                 # fixme @overload cause error in runtime why skipping the right signature?
    # def __getitem__(self, values: Union[Tuple[Assignment, ...], Tuple[int, ...]]) -> float:
    #     if isinstance(values[0], Assignment):
    #         return self.table[tuple(reduce_tuples(values, pos=1))]
    #     return self.table[values]

    def __getitem__(self, values: Tuple[int, ...]) -> float:
        """
        get a view of table subject to the values
        assume it is full assignments. for partial assignment use _marginal

        :param values: assignment of values to the variables following the order in the scope
        :return: view of numpy array by numpy indexing scheme
        """
        return self.table[values]

    def __setitem__(self, values: Tuple[int, ...], new_value: float) -> None:
        """
        set assignment to the table on the values
        assume it is full assignments. for partial assignment numpy broadcast but it is not intended behavior

        :param values: assignment of values to the variables following the order in the scope
        :param new_value: floating value assigned to the table
        """
        self.table[values] = new_value

    def __copy__(self) :
        """
        fully copy a Factor with new factor id

        :return: Factor object copied
        """
        return self.build(self.scope.copy(), self.scope_vids.copy(), np.array(self.table))

    __deepcopy__ = __copy__

    def __hash__(self):
        return hash(repr(self))

    # compare objects by fid to use with sortedcontainer
    def __lt__(self, other) -> bool:
        return self.fid < other.fid

    def __gt__(self, other) -> bool:
        return self.fid > other.fid

    def __eq__(self, other) -> bool:
        return self.fid == other.fid

    def __le__(self, other) -> bool:
        return self.fid <= other.fid

    def __ge__(self, other) -> bool:
        return self.fid > - other.fid

    def __ne__(self, other) -> bool:
        return self.fid != other.fid

    def entropy(self)->float:
        Z = self.sum_marginal( self.scope_vars, inplace=False)
        if Z == 0:      # assert float(Z) > 0
            return 0.0
        temp = np.ravel(self.table)
        H = -np.dot( temp, np.log(temp.clip(min=1e-300)) )/ float(Z) + np.log( float(Z) )
        return H

    def argmax(self)-> Tuple[np.ndarray]:
        # only find argmax on the whole table
        ind = self.table.argmax()       # ind = np.ravel(self.table).argmax() doing ravel by default
        return np.unravel_index(ind, self.shape)



def argmax(f , eliminator: SortedSet):
    raise NotImplementedError

def argmin(f, eliminator: SortedSet):
    raise NotImplementedError

def norm(f, kind='L2'):
    raise NotImplementedError


def distance(f1, f2, kind='L2'):
    raise NotImplementedError
