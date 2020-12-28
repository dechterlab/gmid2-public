"""

GraphicalModel class is a collection of factors

fast processing of collection of factors, this object is singleton

"""
# from __future__ import annotations
from typing import List, Tuple, Union, Iterable, overload, Text, Optional
from copy import copy
from sortedcontainers import SortedDict, SortedSet, SortedList
from .uai_files import FileInfo
from .factor import Factor, Variable
from ..global_constants import *

class GraphicalModel:
    def __init__(self):
        self.fid2f = SortedDict()

        self.vid2var = SortedDict()
        self.vid2type = SortedDict()       # list of types lookup by vid
        self.vid2weight = SortedDict()
        self.vid2fids = SortedDict()      # Fs is SortedSet of Factors

        self.prob_fids_ = None
        self.policy_fids_ = None
        self.util_fids_ = None

        self.is_log = False

    def convert_to_log(self):
        if not self.is_log:
            self.is_log = True
            for fid in self.fid2f:
                self.fid2f[fid].ilog()

    def convert_to_linear(self):
        if self.is_log:
            self.is_log = False
            for fid in self.fid2f:
                self.fid2f[fid].iexp()

    def convert_prob_to_log(self):
        self.is_log = True
        for fid in self.fid2f:
            if self.fid2f[fid].type != TYPE_UTIL_FUNC:
                self.fid2f[fid].ilog()

    def convert_prob_to_linear(self):
        self.is_log = False
        for fid in self.fid2f:
            if self.fid2f[fid].type != TYPE_UTIL_FUNC:
                self.fid2f[fid].iexp()

    def convert_util_to_alpha(self, a:float=ALPHA):
        for fid in self.fid2f:
            if self.fid2f[fid].type == TYPE_UTIL_FUNC:
                self.fid2f[fid] *= a

    def convert_util_to_linear(self, a:float=ALPHA):
        for fid in self.fid2f:
            if self.fid2f[fid].type == TYPE_UTIL_FUNC:
                self.fid2f[fid] /= a

    def convert_util_to_log(self):
        for fid in self.fid2f:
            if self.fid2f[fid].type == TYPE_UTIL_FUNC:
                self.fid2f[fid].ilog()

    def convert_util_to_exp(self):
        for fid in self.fid2f:
            if self.fid2f[fid].type == TYPE_UTIL_FUNC:
                self.fid2f[fid].iexp()

    def add_factor(self, f: Factor) -> None:
        self.fid2f[f.fid] = f
        for ind, vid in enumerate(f.scope_vids):
            if vid not in self.vid2var:     # first variable shown
                var = f.scope_vars[ind]
                self.add_variable(var.vid, var.dim, var.type)
            if vid not in self.vid2fids:
                self.vid2fids[vid] = SortedSet()
            self.vid2fids[vid].add(f.fid)

    def remove_factor(self, f: Factor) -> None:
        for vid in f.scope_vids:                # remove fids from vid2fids
            self.vid2fids[vid].discard(f.fid)
        del self.fid2f[f.fid]  # remove reference to factor object

    def remove_fid(self, fid: int)-> None:
        self.remove_factor(self.fid2f[fid])

    def add_variable(self, vid: int, dim: int, var_type: str) -> None:
        if vid not in self.vid2var:
            self.vid2var[vid] = Variable(vid=vid, dim=dim, type=var_type)
            if var_type == TYPE_CHANCE_VAR:
                self.vid2type[vid] = TYPE_CHANCE_VAR
                self.vid2weight[vid] = 1.0
            if var_type == TYPE_DECISION_VAR:
                self.vid2type[vid] = TYPE_DECISION_VAR
                self.vid2weight[vid] = 0.0

    def build(self, file_info: FileInfo) -> None:
        Factor.reset_fid()  # this build will call only at the beginning and only once, if not this line cause error
        # add variables
        for vid in range(file_info.nvar):
            self.add_variable(vid, file_info.domains[vid], file_info.var_types[vid])

        # add factors one by one
        for i in range(file_info.nfunc):
            sc = [self.vid2var[vid] for vid in file_info.scopes[i]]
            table = file_info.tables[i]
            f_type = file_info.factor_types[i]
            f = Factor(sc, table, f_type)
            self.add_factor(f)

    def __getitem__(self, fid: Union[int, Tuple[int], Iterable[int]]) -> Union[Factor, List[Factor]]:
        """ retrieve factor or list of factors by fid  """
        if isinstance(fid, int):
            return self.fid2f[fid]
        return [self[i] for i in fid]

    def project(self, project_fids: Iterable[int]):
        gm = GraphicalModel()

        for fid in project_fids:
            f = self.fid2f[fid]
            for vid in f.scope_vids:
                if vid not in gm.vid2var:
                    gm.vid2var[vid] = self.vid2var[vid]
                    gm.vid2type[vid] = self.vid2type[vid]
                    gm.vid2weight[vid] = self.vid2weight[vid]
        gm.is_log = self.is_log

        for fid in project_fids:
            gm.add_factor(self.fid2f[fid])
        return gm

    def exclude(self, exclude_fids: Iterable[int]):
        gm = GraphicalModel()

        for fid in self.fid2f:
            f = self.fid2f[fid]
            for vid in f.scope_vids:
                if vid not in gm.vid2var:
                    gm.vid2var[vid] = self.vid2var[vid]
                    gm.vid2type[vid] = self.vid2type[vid]
                    gm.vid2weight[vid] = self.vid2weight[vid]
        gm.is_log = self.is_log

        for fid in self.fid2f:
            if fid not in exclude_fids:
                gm.add_factor(self.fid2f[fid])
        return gm

    def copy_subset(self, copy_fids):
        """ copy gm and only subset of factors are changed """
        gm = GraphicalModel()
        for vid in self.vid2var:
            gm.vid2var[vid] = self.vid2var[vid]
            gm.vid2type[vid] = self.vid2type[vid]
            gm.vid2weight[vid] = self.vid2weight[vid]
        gm.is_log = self.is_log

        for fid in self.fid2f:
            if copy_fids and fid in copy_fids:    # create
                f = self.fid2f[fid]
                gm.add_factor(copy(f))
            else:
                gm.add_factor(self.fid2f[fid])
        return gm

    def change_var_type(self, vid:int, type:Text)->None:
        var = self.vid2var[vid]
        if var.type == type:
            return
        self.vid2var[vid] = Variable(vid, var.dim, type)
        self.vid2type[vid] = type

        for fid in self.vid2fids[vid]:
            f = self.fid2f[fid]
            if vid in f.scope_vids:
                f.scope_vars.discard(var)
                f.scope_vars.add(self.vid2var[vid])

    @property
    def scope_vids(self)->List[SortedSet]:
        return [ self.fid2f[fid].scope_vids for fid in self.fid2f]

    @property
    def prob_fids(self)-> SortedSet:
        if self.fid2f is not None:
            if self.prob_fids_ is None:
                self.prob_fids_ = SortedSet([fid for fid in self.fid2f if self.fid2f[fid].type == TYPE_PROB_FUNC])
            return self.prob_fids_

    @property
    def policy_fids(self)-> SortedSet:
        if self.fid2f is not None:
            if self.policy_fids_ is None:
                self.policy_fids_ = SortedSet([fid for fid in self.fid2f if self.fid2f[fid].type == TYPE_POLICY_FUNC])
            return self.policy_fids_

    @property
    def util_fids(self)-> SortedSet:
        if self.fid2f is not None:
            if self.util_fids_ is None:
                self.util_fids_ = SortedSet([fid for fid in self.fid2f if self.fid2f[fid].type == TYPE_UTIL_FUNC])
            return self.util_fids_

    def reset_prob_fids(self):
        self.prob_fids_ = None

    def reset_policy_fids(self):
        self.policy_fids_ = None

    def reset_util_fids(self):
        self.util_fids_ = None




