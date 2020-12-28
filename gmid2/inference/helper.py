from typing import List, Iterable
import functools
import numpy as np
from copy import copy

from gmid2.global_constants import *
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.basics.factor import Variable, Factor


def variables_from_vids(gm: GraphicalModel, scope_vids: List[int])-> List[Variable]:
    return [gm.vid2var[vid] for vid in scope_vids]


def table_length(gm: GraphicalModel, scope_vids: List[int])-> int:
    return functools.reduce(lambda x,y: x*y, (gm.vid2var[vid].dim for vid in scope_vids) )


def const_factor(gm: GraphicalModel, scope_vids: List[int], constant:float, type:str)->Factor:
    if gm.is_log:
        table = [np.log(constant)]
    else:
        table = [constant]
    f = Factor(variables_from_vids(gm, scope_vids), table * table_length(gm, scope_vids), type)
    return f


def combine_factor_list(factor_set:List[Factor], is_log:bool)->Factor:
    """
    combine into the first element
    """
    f = copy(factor_set[0])
    if is_log:
        for i in range(1, len(factor_set)):
            f += factor_set[i]
    else:
        for i in range(1, len(factor_set)):
            f *= factor_set[i]
    return f


def combine_factors(factor: Factor, other: Factor, is_log:bool) ->Factor:
    if is_log:
        factor += other     # inplace
    else:
        factor *= other     # inplace
    return factor


def extract_max(f:Factor, is_log)->float:
    m = np.max(f.table)
    if is_log:
        f.table = f.table - m       # subtract max
    else:
        f.table = f.table /m        # divide max
    return m                        # return extracted value    this should be stored somewhere as constant


def uniform_policy_factor(gm: GraphicalModel, scope_vids: Iterable[int], dec_vid:int)-> Factor:
    dec_dim = gm.vid2var[dec_vid].dim
    dims = np.prod( [gm.vid2var[vid].dim for vid in scope_vids] )
    table = [1.0/dec_dim]*dims
    sc = [gm.vid2var[vid] for vid in scope_vids if vid != dec_vid]
    sc.append(gm.vid2var[dec_vid])
    f = Factor(sc, table, TYPE_POLICY_FUNC)
    if gm.is_log:
        f.ilog()
    return f


def uniform_factor(gm: GraphicalModel, scope_vids: List[int], type:str)-> Factor:
    dims = np.prod( [gm.vid2var[vid].dim for vid in scope_vids])
    table = [1.0/dims]*dims
    sc = [gm.vid2var[vid] for vid in scope_vids]
    f = Factor(sc, table, type)
    if gm.is_log:
        f.ilog()
    return f

def zero_factor(gm: GraphicalModel, scope_vids:List[int], type:str)-> Factor:
    sc = [gm.vid2var[vid] for vid in scope_vids]
    f = Factor(sc, ZERO, type)
    return f


def decode_policy(f:Factor, dec_vid:int)->Factor:
    """ return a policy function from cluster factor the factor is of the form dec given observations """
    dec_ind = f.scope_vids.bisect_left(dec_vid)
    sc = list(f.scope_vars)
    dec_dim = sc[dec_ind].dim
    sc[dec_ind], sc[-1] = sc[-1], sc[dec_ind]                   # send decision var to the last
    policy_table = [EPSILON]*f.size # np.zeros(f.size)          # all zero table with the same size
    temp = f.table.swapaxes(dec_ind, -1).reshape(-1)

    dec_max, prev_ind = -INF, -1
    for ind in range(f.size):
        if ind % dec_dim == 0:  # one cycle passed
            dec_max, prev_ind = -INF, -1
        if temp[ind] > dec_max:
            policy_table[ind] = 1.0 - EPSILON*(dec_dim-1)
            dec_max = temp[ind]
            policy_table[prev_ind] = EPSILON
            prev_ind = ind
    policy = Factor(sc, policy_table, TYPE_POLICY_FUNC)
    policy.ilog()       # always log scale
    return policy